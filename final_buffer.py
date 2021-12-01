import numpy as np
import random
from typing import Dict, List

from final_SegmentTree import MinSegmentTree, SumSegmentTree


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.last_played_buf = np.zeros([size], dtype=np.float32)
        self.delta_buf = np.zeros([size], dtype=np.float32)
        self.episode_buf = np.zeros([size], dtype=np.float32)
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
        self.episode_buf[self.ptr] = transition['episode']
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


class PrioritizedReplayBuffer(ReplayBuffer):
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
        alpha: float = 0.6,
        priority_based: str = 'rank'
    ):
        """Initialization."""
        assert priority_based in ['rank', 'priority', 'hybrid']
        assert alpha >= 0

        tree_capacity = 1
        while tree_capacity < size:
            tree_capacity *= 2

        super(PrioritizedReplayBuffer, self).__init__(obs_dim, tree_capacity, batch_size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        self.offset = 1
        self.priority_based = priority_based

        # capacity must be positive and a power of 2.

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

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

    def update_priorities(self, indices: List[int], quantities: np.ndarray):
        if self.priority_based == 'rank':
            self.update_priorities_rank(indices, quantities)
        if self.priority_based == 'priority':
            self.update_priorities_priority(indices, quantities)

    def update_priorities_priority(self, indices: List[int], priorities: np.ndarray):
        assert len(indices) == len(priorities)

        max_priority_local = np.max(priorities)
        self.max_priority = max(self.max_priority, max_priority_local)

        t = len(self.sum_tree.tree)/2
        indices_ = [int(o) + int(t) for o in indices]
        # indices_ = np.array(indices).astype(int) + int(t)

        for i, idx in enumerate(indices):
          self.sum_tree.tree[idx] = priorities[i] ** self.alpha
          self.min_tree.tree[idx] = priorities[i] ** self.alpha

        self.sum_tree.update_tree()
        self.min_tree.update_tree()

    def update_priorities_rank(self, indices: List[int], deltas: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(deltas)

        deltas = deltas.reshape(-1)
        self.delta_buf[indices] = deltas
        temp = self.delta_buf.argsort()[::-1]  # from biggest = 1 ....
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(self.delta_buf)) + self.offset

        priorities = (1/ranks)**self.alpha
        max_priority_local = np.max(priorities)
        self.max_priority = max(self.max_priority, max_priority_local)
        t = len(self.sum_tree.tree)/2

        self.sum_tree.tree[int(t):] = priorities
        self.min_tree.tree[int(t):] = priorities

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
