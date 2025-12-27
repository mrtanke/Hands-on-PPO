# algorithms/sac/networks.py
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_STD_MIN = -20
LOG_STD_MAX = 2


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """Q(s,a)"""
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, *hidden_sizes, 1])

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


class GaussianPolicy(nn.Module):
    """
    Squashed Gaussian policy:
      u ~ N(mu, std), a = tanh(u) * act_scale + act_bias
    Returns:
      action, log_prob, mean_action
    """
    def __init__(self, obs_dim: int, act_dim: int, act_low, act_high, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden_sizes, 2 * act_dim])
        self.act_dim = act_dim

        act_low = np.array(act_low, dtype=np.float32)
        act_high = np.array(act_high, dtype=np.float32)
        self.register_buffer("act_scale", torch.tensor((act_high - act_low) / 2.0))
        self.register_buffer("act_bias", torch.tensor((act_high + act_low) / 2.0))

    def forward(self, obs: torch.Tensor):
        out = self.net(obs)
        mu, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs: torch.Tensor):
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)

        # reparameterization trick
        u = dist.rsample()
        a_tanh = torch.tanh(u)
        action = a_tanh * self.act_scale + self.act_bias

        # log_prob correction for tanh squash
        # log pi(a) = log N(u|mu,std) - sum log(1 - tanh(u)^2)
        log_prob_u = dist.log_prob(u).sum(dim=-1, keepdim=True)
        correction = torch.log(1 - a_tanh.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        log_prob = log_prob_u - correction

        mean_action = torch.tanh(mu) * self.act_scale + self.act_bias
        return action, log_prob, mean_action
