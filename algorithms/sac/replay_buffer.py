# algorithms/sac/replay_buffer.py
from __future__ import annotations
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device

        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.acts = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rews = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)

        obs = torch.as_tensor(self.obs[idx], device=self.device)
        acts = torch.as_tensor(self.acts[idx], device=self.device)
        rews = torch.as_tensor(self.rews[idx], device=self.device)
        next_obs = torch.as_tensor(self.next_obs[idx], device=self.device)
        dones = torch.as_tensor(self.dones[idx], device=self.device)

        return obs, acts, rews, next_obs, dones
