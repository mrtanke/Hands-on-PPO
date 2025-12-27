# sac_agent.py
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# MLP helper
def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class ReplayBuffer:
    """
    A simple FIFO replay buffer for SAC (continuous control).
    Stores transitions: (obs, act, rew, next_obs, done).
    """
    def __init__(self, obs_dim: int, act_dim: int, size: int, device: torch.device):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)

        self.max_size = size
        self.ptr = 0
        self.size = 0
        self.device = device

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.obs2_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size # circular buffer
        self.size = min(self.size + 1, self.max_size) # store range

    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size) # dimension: [batch_size]
        batch = dict(
            obs=torch.as_tensor(self.obs_buf[idxs], device=self.device), # dimension: [batch_size, obs_dim]
            act=torch.as_tensor(self.act_buf[idxs], device=self.device),
            rew=torch.as_tensor(self.rew_buf[idxs], device=self.device),
            obs2=torch.as_tensor(self.obs2_buf[idxs], device=self.device),
            done=torch.as_tensor(self.done_buf[idxs], device=self.device),
        )
        return batch


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class SquashedGaussianActor(nn.Module):
    """
    SAC actor: outputs tanh-squashed Gaussian actions.
    Includes log-prob correction for tanh squashing.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden_sizes], activation=nn.ReLU, output_activation=nn.ReLU)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim) # use for mean
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim) # use for std

    def forward(self, obs: torch.Tensor, deterministic: bool = False, with_logprob: bool = True):
        hidden = self.net(obs)

        mu = self.mu_layer(hidden)
        log_std = self.log_std_layer(hidden)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX) # [batch_size, act_dim] -> [-20,2]
        std = torch.exp(log_std) # positive std [batch_size, act_dim] -> [2.06e-9, 7.39]

        if deterministic: # test
            pi_action = mu
        else: # train
            pi_action = mu + std * torch.randn_like(mu)

        if with_logprob: # for computing actor loss
            # log_prob of Gaussian (before tanh)
            pre_tanh_logp = (-0.5 * (((pi_action - mu) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))).sum(
                dim=-1, keepdim=True
            ) # [batch_size, 1]
        else:
            pre_tanh_logp = None

        # apply tanh squashing
        tanh_action = torch.tanh(pi_action)

        # tanh curshes the distribution, need to correct log prob for entropy calculation
        if with_logprob:
            # tanh correction: log(1 - tanh(x)^2)
            # Add small eps for numerical stability
            eps = 1e-6
            correction = torch.log(1 - tanh_action.pow(2) + eps).sum(dim=-1, keepdim=True)
            logp_pi = pre_tanh_logp - correction
        else:
            logp_pi = None

        return tanh_action, logp_pi


class QCritic(nn.Module):
    """
    Q(s,a) critic for SAC.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.q = mlp([obs_dim + act_dim, *hidden_sizes, 1], activation=nn.ReLU, output_activation=nn.Identity)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.q(x)


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    alpha: float = 0.2              # used if auto_alpha=False
    auto_alpha: bool = True
    target_entropy: float = None    # if None: = -act_dim
    hidden_sizes: Tuple[int, int] = (256, 256)


class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_limit: float,
        device: torch.device,
        cfg: SACConfig,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.device = device
        self.cfg = cfg

        self.total_grad_steps = 0

        self.actor = SquashedGaussianActor(obs_dim, act_dim, hidden_sizes=cfg.hidden_sizes).to(device)
        self.critic1 = QCritic(obs_dim, act_dim, hidden_sizes=cfg.hidden_sizes).to(device)
        self.critic2 = QCritic(obs_dim, act_dim, hidden_sizes=cfg.hidden_sizes).to(device)

        self.critic1_target = QCritic(obs_dim, act_dim, hidden_sizes=cfg.hidden_sizes).to(device)
        self.critic2_target = QCritic(obs_dim, act_dim, hidden_sizes=cfg.hidden_sizes).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=cfg.lr)
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=cfg.lr)

        # Entropy temperature (alpha)
        if cfg.auto_alpha:
            if cfg.target_entropy is None:
                target_entropy = -float(act_dim)
            else:
                target_entropy = float(cfg.target_entropy)

            self.target_entropy = target_entropy
            self.log_alpha = torch.tensor(np.log(cfg.alpha), dtype=torch.float32, device=device, requires_grad=True)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.lr)
        else:
            self.target_entropy = None
            self.log_alpha = None
            self.alpha_optim = None

    @property
    def alpha(self) -> torch.Tensor:
        if self.cfg.auto_alpha:
            return self.log_alpha.exp()
        return torch.tensor(self.cfg.alpha, device=self.device)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _ = self.actor(obs_t, deterministic=deterministic, with_logprob=False)
        a = a.squeeze(0).cpu().numpy()
        # scale from [-1,1] to env action bounds
        return self.act_limit * a

    # soft target network update
    def _polyak_update(self):
        tau = self.cfg.tau
        with torch.no_grad():
            for p, p_target in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                p_target.data.mul_(1 - tau).add_(tau * p.data)
            for p, p_target in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                p_target.data.mul_(1 - tau).add_(tau * p.data)

    def update(self, batch: Dict[str, torch.Tensor], update_policy: bool = True) -> Dict[str, float]:
        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        obs2 = batch["obs2"]
        done = batch["done"]

        # Critics loss
        with torch.no_grad():
            a2, logp_a2 = self.actor(obs2, deterministic=False, with_logprob=True)

            critic1_pi_target = self.critic1_target(obs2, a2)
            critic2_pi_target = self.critic2_target(obs2, a2)
            critic_pi_target = torch.min(critic1_pi_target, critic2_pi_target)

            value = critic_pi_target - self.alpha * logp_a2

            target = rew + self.cfg.gamma * (1 - done) * value

        critic1 = self.critic1(obs, act)
        critic2 = self.critic2(obs, act)
        loss_critic1 = F.mse_loss(critic1, target)
        loss_critic2 = F.mse_loss(critic2, target)

        self.critic1_optim.zero_grad()
        loss_critic1.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        loss_critic2.backward()
        self.critic2_optim.step()

        metrics = {
            "loss_critic1": float(loss_critic1.detach().cpu().item()),
            "loss_critic2": float(loss_critic2.detach().cpu().item()),
            "critic1_mean": float(critic1.detach().mean().cpu().item()),
            "critic2_mean": float(critic2.detach().mean().cpu().item()),
            "loss_action": None,
            "alpha_loss": 0.0,
            "logp_mean": None,
            "alpha": float(self.alpha.detach().cpu().item()),
            "policy_updated": False,
        }

        if update_policy:
            # Actor loss (freeze critics)
            for p in self.critic1.parameters():
                p.requires_grad = False
            for p in self.critic2.parameters():
                p.requires_grad = False

            a_pi, logp_a = self.actor(obs, deterministic=False, with_logprob=True)
            critic1_pi = self.critic1(obs, a_pi)
            critic2_pi = self.critic2(obs, a_pi)
            critic_pi = torch.min(critic1_pi, critic2_pi)
            loss_action = (self.alpha * logp_a - critic_pi).mean()

            self.actor_optim.zero_grad()
            loss_action.backward()
            self.actor_optim.step()

            for p in self.critic1.parameters():
                p.requires_grad = True
            for p in self.critic2.parameters():
                p.requires_grad = True

            metrics["loss_action"] = float(loss_action.detach().cpu().item())
            metrics["logp_mean"] = float(logp_a.detach().mean().cpu().item())
            metrics["policy_updated"] = True

            # Temperature loss (optional)
            if self.cfg.auto_alpha:
                # J(alpha) = E[ alpha * (-log_pi - target_entropy) ]
                alpha_loss = -(self.log_alpha * (logp_a.detach() + self.target_entropy)).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                metrics["alpha_loss"] = float(alpha_loss.detach().cpu().item())
                metrics["alpha"] = float(self.alpha.detach().cpu().item())
            else:
                metrics["alpha_loss"] = 0.0

            # Target networks update
            self._polyak_update()

        return metrics

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "cfg": self.cfg.__dict__,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "act_limit": self.act_limit,
        }
        if self.cfg.auto_alpha:
            payload["log_alpha"] = self.log_alpha.detach().cpu().numpy()
        torch.save(payload, path)

    def load(self, path: str, strict: bool = True):
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(payload["actor"], strict=strict)
        self.critic1.load_state_dict(payload["critic1"], strict=strict)
        self.critic2.load_state_dict(payload["critic2"], strict=strict)
        self.critic1_target.load_state_dict(payload["critic1_target"], strict=strict)
        self.critic2_target.load_state_dict(payload["critic2_target"], strict=strict)
        if self.cfg.auto_alpha and "log_alpha" in payload:
            with torch.no_grad():
                self.log_alpha.copy_(torch.tensor(np.log(payload["log_alpha"] + 1e-8), device=self.device))
