import argparse
import os

import gymnasium as gym
import torch

from agent import SACAgent, SACConfig


def load_agent(ckpt_path: str, device: torch.device) -> SACAgent:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = SACConfig(**payload["cfg"])

    agent = SACAgent(
        obs_dim=payload["obs_dim"],
        act_dim=payload["act_dim"],
        act_limit=payload["act_limit"],
        device=device,
        cfg=cfg,
    )
    agent.load(ckpt_path)
    return agent


def main():
    parser = argparse.ArgumentParser(description="Render a trained SAC policy.")
    parser.add_argument("--env_id", type=str, default="Pendulum-v1")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=os.path.join("logs", "Pendulum-v1_sac.pt"),
        help="Path to the saved policy checkpoint.",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render_mode", type=str, default="human")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    agent = load_agent(args.ckpt_path, device)
    env = gym.make(args.env_id, render_mode=args.render_mode)

    print(f"Loaded checkpoint from {args.ckpt_path}. Rendering {args.episodes} episode(s)...")
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        ep_ret = 0.0
        while not done:
            act = agent.act(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(act)
            done = terminated or truncated
            ep_ret += float(rew)
        print(f"Episode {ep + 1}: return={ep_ret:.2f}")

    env.close()


if __name__ == "__main__":
    main()
