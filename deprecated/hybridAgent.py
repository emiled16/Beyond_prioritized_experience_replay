import sys
import os
import random
from typing import Dict, List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from SegmentTree import MinSegmentTree, SumSegmentTree
from buffer import PrioritizedReplayBuffer, ReplayBuffer
from SegmentTree2 import MinSegmentTree2, SumSegmentTree2
from buffer2 import PrioritizedReplayBuffer2, ReplayBuffer2
from model import Network


class hybridDQNAgent:
    """hybridDQN Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        epsilon (float): parameter for epsilon greedy policy
        epsilon_decay (float): step size to decrease epsilon
        max_epsilon (float): max value of epsilon
        min_epsilon (float): min value of epsilon
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
        beta (float): determines how much importance sampling is used
        prior_eps (float): guarantees every transition can be sampled
    """

    def __init__(
        self, 
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            epsilon_decay (float): step size to decrease epsilon
            lr (float): learning rate
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            gamma (float): discount factor
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
        """
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        self.env = env
        
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilons = []
        self.target_update = target_update
        self.gamma = gamma
        
        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)
        
        # PER
        # In DQN, We used "ReplayBuffer(obs_dim, memory_size, batch_size)"
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory_priority = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha
        )
        self.memory_rank = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha
        )
        

        # networks: dqn, dqn_target
        self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        # self.transition = list()
        
        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.dqn(
                torch.FloatTensor(state).to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        
        # if not self.is_test:
        #     self.transition = [state, selected_action]
        
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)

        # if not self.is_test:
        #     self.transition += [reward, next_state, done]
        #     self.memory.store(*self.transition)
    
        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""

        # Priority
        # PER needs beta to calculate weights
        samples_priority = self.memory_priority.sample_batch(self.beta)
        weights_priority = torch.FloatTensor(
            samples_priority["weights"].reshape(-1, 1)
        ).to(self.device)
        indices_priority = samples_priority["indices"]

        # PER: importance sampling before average
        elementwise_loss_priority = self._compute_dqn_loss(samples_priority)
        loss_priority = torch.mean(elementwise_loss_priority * weights_priority)

        # Rank
        # PER needs beta to calculate weights
        samples_rank = self.memory_rank.sample_batch(self.beta)
        weights_rank = torch.FloatTensor(
            samples_rank["weights"].reshape(-1, 1)
        ).to(self.device)
        indices_rank = samples_rank["indices"]

        # PER: importance sampling before average
        elementwise_loss_rank = self._compute_dqn_loss(samples_rank)
        loss_rank = torch.mean(elementwise_loss_rank * weights_rank)

        # Entire loss
        loss = loss_priority + loss_rank



        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # priority
        # PER: update priorities
        loss_for_prior_priority = elementwise_loss_priority.detach().cpu().numpy()
        new_priorities_priority = loss_for_prior_priority + self.prior_eps
        self.memory_priority.update_priorities(indices_priority, new_priorities_priority)
        # print(self.memory.sum_tree.tree)
        # Rank
        # PER: update priorities
        loss_for_prior_rank = elementwise_loss_rank.detach().cpu().numpy()
        new_priorities_rank = loss_for_prior_rank + self.prior_eps
        self.memory_rank.update_priorities(indices_rank, new_priorities_rank)

        return loss.item()
        
    def update_beta(self,frame_idx):
        fraction = min(frame_idx / self.num_frames, 1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)

    def update_eps(self):
        self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                        self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )
        self.epsilons.append(self.epsilon)
    # def train(self, num_frames: int, plotting_interval: int = 200):
    def train(self, num_episodes: int, plotting_interval: int = 20):
        """Train the agent."""
        self.is_test = False
        self.num_frames = num_episodes # num_frames
        episode = 1
        state = self.env.reset()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0

        # for frame_idx in range(1, num_frames + 1):
        while True:
            transition = dict()
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            transition['obs'] = state
            transition['next_obs'] = next_state
            transition['action'] = action
            transition['reward'] = reward
            transition['done'] = done
            # transition['last_played'] = frame_idx
            transition['last_played'] = episode
            self.memory_priority.store(transition)
            self.memory_rank.store(transition)
            # transition['td_err'] = 0
            state = next_state
            score += reward

            
            # PER: increase beta
            # self.update_beta(frame_idx)
            self.update_beta(episode)
            # fraction = min(frame_idx / num_frames, 1.0)
            # self.beta = self.beta + fraction * (1.0 - self.beta)

            # if episode ends
            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0
                episode += 1
                # print(episode)


            # if training is ready
            if len(self.memory_priority) >= self.batch_size:
                loss = self.update_model()
                # print(loss)
                losses.append(loss)
                update_cnt += 1

                 # linearly decrease epsilon
                self.update_eps()

                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            # plotting
            # if frame_idx % plotting_interval == 0:
            if episode % plotting_interval == 0:
                # print(scores)
                # print(np.mean(scores[-10:]))
                # self._plot(frame_idx, scores, losses, self.epsilons)
                self._plot(episode, scores, losses, self.epsilons)

            if episode == num_episodes:
                break                
        self.env.close()
                
    def test(self) -> List[np.ndarray]:
        """Test the agent."""
        self.is_test = True
        
        state = self.env.reset()
        done = False
        score = 0
        
        frames = []
        while not done:
            frames.append(self.env.render(mode="rgb_array"))
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        
        print("score: ", score)
        self.env.close()
        
        return frames

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(
            next_state
        ).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # calculate element-wise dqn loss
        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")

        return elementwise_loss
    
    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
                
    def _plot(
        self, 
        frame_idx: int, 
        scores: List[float], 
        losses: List[float], 
        epsilons: List[float],
    ):
        """Plot the training progresses."""
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.show()