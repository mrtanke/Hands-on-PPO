# train_sac.py
import argparse
import os
import time

import gymnasium as gym
import numpy as np
import torch

from agent import ReplayBuffer, SACAgent, SACConfig


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_policy(env_id: str, agent: SACAgent, device: torch.device, seed: int, episodes: int = 5) -> float:
    env = gym.make(env_id)
    returns = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + 1000 + ep)
        done = False
        ep_ret = 0.0
        while not done:
            act = agent.act(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(act)
            done = terminated or truncated
            ep_ret += float(rew)
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="Pendulum-v1")
    parser.add_argument("--total_timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # SAC typical knobs
    parser.add_argument("--buffer_size", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--start_steps", type=int, default=10_000) # steps of random actions at start of training
    parser.add_argument("--update_after", type=int, default=1_000) # number of env steps before starting updates
    parser.add_argument("--update_every", type=int, default=50) # do N gradient steps every N env steps
    parser.add_argument(
        "--policy_delay",
        type=int,
        default=1,
        help="number of critic updates between policy/alpha/target updates",
    )

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto_alpha", action="store_true", default=True)
    parser.add_argument("--no_auto_alpha", dest="auto_alpha", action="store_false")

    parser.add_argument("--eval_interval", type=int, default=10_000)
    parser.add_argument("--eval_episodes", type=int, default=5)

    parser.add_argument("--log_dir", type=str, default="logs")
    args = parser.parse_args()

    device = torch.device(args.device)
    set_seed(args.seed)

    env = gym.make(args.env_id)
    obs_dim = int(np.prod(env.observation_space.shape)) # dimension: [obs_dim]
    act_dim = int(np.prod(env.action_space.shape)) # dimension: [act_dim]
    act_high = float(env.action_space.high[0])  # assume symmetric bounds
    assert np.allclose(env.action_space.high, -env.action_space.low), "This script assumes symmetric action bounds."

    cfg = SACConfig(
        gamma=args.gamma,
        tau=args.tau,
        lr=args.lr,
        alpha=args.alpha,
        auto_alpha=args.auto_alpha,
        target_entropy=-float(act_dim),
        hidden_sizes=(256, 256),
    )
    agent = SACAgent(obs_dim=obs_dim, act_dim=act_dim, act_limit=act_high, device=device, cfg=cfg)

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=args.buffer_size, device=device)

    os.makedirs(args.log_dir, exist_ok=True)
    ckpt_path = os.path.join(args.log_dir, f"{args.env_id}_sac.pt")
    curve_path = os.path.join(args.log_dir, f"{args.env_id}_sac_eval_returns.npy")

    obs, info = env.reset(seed=args.seed)
    ep_ret, ep_len = 0.0, 0
    t0 = time.time()

    eval_returns = []

    # main collection and training loop
    for t in range(1, args.total_timesteps + 1):
        # action selection
        if t <= args.start_steps:
            act = env.action_space.sample()
        else:
            act = agent.act(obs, deterministic=False)

        next_obs, rew, terminated, truncated, info = env.step(act)
        done = terminated or truncated

        # store (note: done as float 0/1)
        replay_buffer.store(obs, act, rew, next_obs, float(done))

        obs = next_obs
        ep_ret += float(rew) # episode return
        ep_len += 1 # episode length

        if done:
            obs, info = env.reset()
            ep_ret, ep_len = 0.0, 0

        # updates
        # update_every: do N updates every N steps
        # policy_delay: update policy/alpha/target every M updates
        if (
            t >= args.update_after
            and replay_buffer.size >= args.batch_size
            and t % args.update_every == 0
        ):
            for _ in range(args.update_every):
                # Increment a persistent counter inside the agent
                agent.total_grad_steps += 1 
                
                # Check delay against the GLOBAL counter
                update_policy = (agent.total_grad_steps % args.policy_delay == 0)
                
                batch = replay_buffer.sample_batch(args.batch_size)
                metrics = agent.update(batch, update_policy=update_policy)

        # evaluation
        if t % args.eval_interval == 0 or t == args.total_timesteps:
            avg_ret = evaluate_policy(
                env_id=args.env_id,
                agent=agent,
                device=device,
                seed=args.seed,
                episodes=args.eval_episodes,
            )
            eval_returns.append([t, avg_ret])

            dt = time.time() - t0
            print(
                f"[SAC] step={t:>8d}  eval_return={avg_ret:>9.2f}  "
                f"alpha={metrics['alpha'] if 'metrics' in locals() else agent.alpha.item():.3f}  "
                f"time={dt:.1f}s"
            )

            # save curve + checkpoint
            np.save(curve_path, np.array(eval_returns, dtype=np.float32))
            agent.save(ckpt_path)

    env.close()
    print(f"Saved SAC checkpoint: {ckpt_path}")
    print(f"Saved eval curve: {curve_path}")


if __name__ == "__main__":
    main()
