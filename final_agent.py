from typing import Dict, List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output

from final_buffer import PrioritizedReplayBuffer
from final_model import Network


class DQNAgent:
    """DQN Agent interacting with environment.

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
        staleness=0.0001,
        positive_reward=0.0001,
        differential=False,
        priority_based='rank',  # or 'priority' or 'hybrid'
        episodic=False
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
        assert priority_based in ['rank', 'priority', 'hybrid'], "priority_based should be in ['rank', 'priority', 'hybrid']"
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.env = env
        if priority_based == 'hybrid':
            self.batch_size = batch_size//2
        else:
            self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilons = []
        self.target_update = target_update
        self.gamma = gamma

        self.staleness = staleness
        self.positive_reward = positive_reward
        self.differential = differential
        self.priority_based = priority_based
        self.episodic = episodic

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # PER
        # In DQN, We used "ReplayBuffer(obs_dim, memory_size, batch_size)"
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = {}
        if priority_based in ['rank', 'priority']:
            self.memory[priority_based] = PrioritizedReplayBuffer(
                obs_dim, memory_size, batch_size, alpha, priority_based)
        else:
            self.memory['priority'] = PrioritizedReplayBuffer(
                        obs_dim, memory_size, batch_size, alpha, 'priority')
            self.memory['rank'] = PrioritizedReplayBuffer(
                        obs_dim, memory_size, batch_size, alpha, 'rank')

        # networks: dqn, dqn_target
        self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

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

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)

        return next_state, reward, done

    def update_model(self):

        loss_p = 0
        loss_r = 0

        if self.priority_based in ['priority', 'hybrid']:
            samples_p, weights_p, indices_p = self.memory_sample(
                self.memory['priority'])
            # PER: importance sampling before average
            elementwise_loss_p, pos_neg_loss_p = self._compute_dqn_loss(samples_p)
            loss_p = torch.mean(elementwise_loss_p * weights_p)

        if self.priority_based in ['rank', 'hybrid']:
            samples_r, weights_r, indices_r = self.memory_sample(
                self.memory['rank'])
            # PER: importance sampling before average
            elementwise_loss_r, pos_neg_loss_r = self._compute_dqn_loss(samples_r)
            loss_r = torch.mean(elementwise_loss_r * weights_r)

        loss = loss_r + loss_p
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.priority_based in ['priority', 'hybrid']:
            self.update_priority_(self.memory['priority'], samples_p, weights_p, indices_p, elementwise_loss_p, pos_neg_loss_p)

        if self.priority_based in ['rank', 'hybrid']:
            self.update_priority_(self.memory['rank'], samples_r, weights_r, indices_r, elementwise_loss_r, pos_neg_loss_r)

        return loss.item()

    def memory_sample(self, selected_memory):
        """Samples a batch from a specific memory"""
        # PER needs beta to calculate weights
        samples = selected_memory.sample_batch(self.beta)

        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]
        return samples, weights, indices

    def update_priority_(self, selected_memory, samples, weights, indices, elementwise_loss, pos_neg_loss):
        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        pos_neg_loss_np = pos_neg_loss.detach().cpu().numpy()
        new_deltas = np.squeeze(loss_for_prior)

        if self.positive_reward != 0:
            positive_indices = np.where(np.squeeze(pos_neg_loss_np) > 0)
            # neg_indices = np.where(np.squeeze(pos_neg_loss_np) <= 0)
            new_deltas[positive_indices] += self.positive_reward

        if self.staleness != 0:
            new_deltas = np.abs(np.squeeze(loss_for_prior) - (self.staleness * self.global_step_count))

        selected_memory.delta_buf[indices] = new_deltas
        selected_memory.last_played_buf[indices] = [self.global_step_count] * len(indices)

        new_priorities = new_deltas

        if self.differential:
            new_priorities = np.abs(selected_memory.delta_buf[indices] - np.squeeze(loss_for_prior))

        new_priorities += self.prior_eps

        selected_memory.update_priorities(indices, new_priorities)

    def update_beta(self, frame_idx):
        fraction = min(frame_idx / self.num_frames, 1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)

    def update_eps(self):
        self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                        self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )
        self.epsilons.append(self.epsilon)

    def train(self, num_episodes: int, plotting_interval: int = 20):
        """Train the agent."""
        self.is_test = False
        self.num_frames = num_episodes  # num_frames
        episode = 1
        self.global_step_count = 0
        state = self.env.reset()
        update_cnt = 0
        losses = []
        scores = []
        score = 0

        while True:
            self.global_step_count += 1
            transition = dict()

            # select the action to take
            action = self.select_action(state)
            # step
            next_state, reward, done = self.step(action)

            transition['obs'] = state
            transition['next_obs'] = next_state
            transition['action'] = action
            transition['reward'] = reward
            transition['done'] = done
            transition['last_played'] = episode
            transition['td_err'] = 0
            transition['episode'] = episode

            if self.priority_based == 'hybrid':
                self.memory['priority'].store(transition)
                self.memory['rank'].store(transition)
            else:
                self.memory[self.priority_based].store(transition)

            state = next_state
            score += reward

            # PER: increase beta
            self.update_beta(episode)

            # if episode ends
            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0
                episode += 1

            # if training is ready
            if self.priority_based in ['rank', 'hybrid']:
                len_memory = len(self.memory['rank'])

            if self.priority_based in ['priority', 'hybrid']:
                len_memory = len(self.memory['priority'])

            if len_memory >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

                self.update_eps()

                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            # plotting
            if episode % plotting_interval == 0:
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
        pos_neg_loss = target - curr_q_value

        return elementwise_loss, pos_neg_loss

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
