import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import typing as Tuple
import numpy as np

from models import PolicyValueNet
from utils import compute_gae, evaluate_cartpole, save_stats

def collect_trajectories(env, policy, num_steps, device):
    """
    Collect trajectories by interacting with the environment using the current policy.

    :param env: environment object
    :param policy: policy network
    :param num_steps: number of steps to collect
    :param device: torch device

    :return: dictionary containing collected data
    """
    obs_list = []
    actions_list = []
    rewards_list = []
    done_list = []
    value_list = []
    log_probs_list = []

    obs, info = env.reset()
    for _ in range(num_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0) # [1, obs_dim]
    
        with torch.no_grad():
            action, log_prob, value, dist = policy.get_action_and_value(obs_tensor)

        action_np = action.cpu().numpy()[0]
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated

        obs_list.append(obs)
        actions_list.append(action_np)
        rewards_list.append(reward)
        done_list.append(done)
        value_list.append(value.cpu().numpy()[0])
        log_probs_list.append(log_prob.cpu().numpy()[0])

        obs = next_obs
        if done:
            obs, info = env.reset()
    
    data = {
        "obs": torch.tensor(obs_list, dtype=torch.float32, device=device),
        "actions": torch.tensor(actions_list, dtype=torch.int64, device=device),
        "rewards": torch.tensor(rewards_list, dtype=torch.float32, device=device),
        "dones": torch.tensor(done_list, dtype=torch.float32, device=device),
        "values": torch.tensor(value_list, dtype=torch.float32, device=device),
        "log_probs": torch.tensor(log_probs_list, dtype=torch.float32, device=device),
    }
    return data


def ppo_update(policy, optimizer, data, ppo_hyperparams):
    obs = data["obs"]
    actions = data["actions"]
    advantages = data["advantages"]
    returns = data["returns"]
    old_log_probs = data["log_probs"]

    batch_size = ppo_hyperparams["batch_size"]
    clip_range = ppo_hyperparams["clip_range"]
    train_epochs = ppo_hyperparams["train_epochs"]

    N = obs.shape[0]
    idxs = torch.arange(N)

    for epoch in range(train_epochs):
        perm = torch.randperm(N) # shuffle indices, dimension: [N]
        for start in range(0, N, batch_size): # split N into chunks of batch_size
            end = start + batch_size
            mb_idx = perm[start:end] # dimenstion: [batch_size]

            mb_obs = obs[mb_idx] # [batch_size, obs_dim]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_returns = returns[mb_idx]

            # forward pass
            logits, values = policy(mb_obs) # logits: [batch_size, act_dim], values: [batch_size]
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(mb_actions)

            # ratio
            ratio = torch.exp(new_log_probs - mb_old_log_probs)

            # policy loss
            unclipped = ratio * mb_advantages
            clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * mb_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()

            # value loss (MSE)
            value_loss = (mb_returns - values).pow(2).mean()

            # entropy bonus (optional)
            entropy = dist.entropy().mean()
            ent_coef = ppo_hyperparams.get("ent_coef", 0.0)
            vf_coef = ppo_hyperparams.get("vf_coef", 0.5)

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def train_ppo_cartpole():
    env = gym.make("CartPole-v1", render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the neural network that contains both policy and value function
    policy = PolicyValueNet(
        obs_dim=env.observation_space.shape[0], # environment state -> input dimension
        act_dim=env.action_space.n, # number of actions -> output dimension
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    # placeholder config
    ppo_hyperparams = dict(
        gamma=0.99,
        lam=0.95,
        clip_range=0.2,
        train_epochs=4,
        batch_size=64,
        ent_coef=0.0,
        vf_coef=0.5,
    )

    total_timesteps = 50_000
    timesteps_collected = 0
    num_steps_per_rollout = 2048

    episode_rewards = []
    obs, info = env.reset()

    # history for logging
    timesteps_history = []
    rewards_history = []

    while timesteps_collected < total_timesteps:
        # Step 1: collect trajectories
        data = collect_trajectories(env, policy, num_steps_per_rollout, device=device)
        timesteps_collected += num_steps_per_rollout
        
        # Step 2: compute advantages and returns
        advantages, returns = compute_gae(
            rewards=data["rewards"],
            value=data["values"],
            dones=data["dones"],
            gamma=ppo_hyperparams["gamma"],
            gae_lambda=ppo_hyperparams["lam"],
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # normalize advantages
        data["advantages"] = advantages
        data["returns"] = returns
        
        # Step 3: ppo update
        ppo_update(policy, optimizer, data, ppo_hyperparams)

        # Step 4: log progress
        eval_rewards = evaluate_cartpole(env, policy, device)
        episode_rewards.append(eval_rewards)
        timesteps_history.append(timesteps_collected)
        rewards_history.append(eval_rewards)
        print(f"Timesteps: {timesteps_collected}, Eval Reward: {eval_rewards}")

    # Save training statistics
    save_stats("cartpole_training_stats.npz", timesteps_history, rewards_history)

    env.close()

if __name__ == "__main__":
    train_ppo_cartpole()
