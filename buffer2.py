import numpy as np
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
from SegmentTree2 import MinSegmentTree2, SumSegmentTree2
import os


class ReplayBuffer2:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.last_played_buf = np.zeros([size], dtype=np.float32)
        self.delta_buf = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        transition: Dict
    ):
        self.obs_buf[self.ptr] = transition['obs']
        self.next_obs_buf[self.ptr] = transition['next_obs']
        self.acts_buf[self.ptr] = transition['action']
        self.rews_buf[self.ptr] = transition['reward']
        self.done_buf[self.ptr] = transition['done']
        self.last_played_buf[self.ptr] = transition['last_played']
        self.delta_buf[self.ptr] = transition['td_err']
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer2(ReplayBuffer2):
    """Prioritized Replay buffer.
    
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        
    """
    
    def __init__(
        self, 
        obs_dim: int,
        size: int, 
        batch_size: int = 32, 
        alpha: float = 0.6
    ):
        """Initialization."""
        assert alpha >= 0
        
        super(PrioritizedReplayBuffer2, self).__init__(obs_dim, size, batch_size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        self.offset = 1
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree2(tree_capacity)
        self.min_tree = MinSegmentTree2(tree_capacity)
        
    def store(
        self, 
        transition: Dict
    ):
        """Store experience and priority."""
        super().store(transition)
        
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        self.sum_tree.update_tree()
        self.min_tree.update_tree()


    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0
        
        indices = self._sample_proportional()
        
        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )


    def update_priorities(self, indices: List[int], deltas: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)
        temp = deltas.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(deltas)) + self.offset
        priorities = (1/ranks)**self.alpha
        max_priority_local = np.max(priorities)
        self.max_priority = max(self.max_priority, max_priority_local)

        for idx, delta, priority in zip(indices, deltas, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            # added by me ------------------
            self.delta_buf[idx] = delta
            self.sum_tree[idx] = priority
            self.min_tree[idx] = priority

        self.sum_tree.update_tree()
        self.min_tree.update_tree()
            
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices

    def compute_ranks(self):
        temp = self.delta_buf.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(self.delta_buf))
        return ranks
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight