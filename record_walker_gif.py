import argparse
from pathlib import Path
from typing import Optional, Sequence

import gymnasium as gym
import imageio
import numpy as np
import torch

from models import ContinuousPolicyValueNet
from utils import ObsNormalizer


def load_policy(model_path: Path, env_id: str, device: torch.device, obs_dim: int, act_dim: int):
    policy = ContinuousPolicyValueNet(obs_dim=obs_dim, act_dim=act_dim).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    state_dict = checkpoint.get("policy_state_dict", checkpoint)
    policy.load_state_dict(state_dict)
    policy.eval()

    obs_normalizer = None
    obs_state = checkpoint.get("obs_normalizer")
    if obs_state is not None:
        obs_normalizer = ObsNormalizer(obs_dim)
        obs_normalizer.mean = np.asarray(obs_state["mean"], dtype=np.float32)
        obs_normalizer.var = np.asarray(obs_state["var"], dtype=np.float32)
        obs_normalizer.count = float(obs_state["count"])

    return policy, obs_normalizer


def rollout_episode(env_id: str,
                    policy: ContinuousPolicyValueNet,
                    device: torch.device,
                    obs_normalizer: Optional[ObsNormalizer],
                    action_low: np.ndarray,
                    action_high: np.ndarray,
                    num_steps: int,
                    seed: Optional[int]) -> tuple[list[np.ndarray], float, int]:
    env = gym.make(env_id, render_mode="rgb_array")
    try:
        obs, _ = env.reset(seed=seed)
        frames: list[np.ndarray] = []
        first_frame = env.render()
        if first_frame is not None:
            frames.append(first_frame)

        steps = 0
        total_reward = 0.0
        done = False

        while not done and steps < num_steps:
            obs_input = (
                obs_normalizer.normalize(obs)
                if obs_normalizer is not None and obs_normalizer.count > 0
                else obs
            )
            obs_tensor = torch.tensor(obs_input, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                mean, _, _ = policy.forward(obs_tensor)
                action = mean

            action_np = action.cpu().numpy()[0]
            action_np = np.clip(action_np, action_low, action_high)

            obs, reward, terminated, truncated, _ = env.step(action_np)
            total_reward += float(reward)

            frame = env.render()
            if frame is not None:
                frames.append(frame)

            done = terminated or truncated
            steps += 1

    finally:
        env.close()

    return frames, total_reward, steps


def record_gif(env_id: str,
               model_path: Path,
               gif_path: Path,
               num_steps: int,
               fps: int,
               seed: Optional[int] = None,
               num_attempts: int = 1) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe_env = gym.make(env_id)
    obs_dim = probe_env.observation_space.shape[0]
    act_dim = probe_env.action_space.shape[0]
    action_low = probe_env.action_space.low
    action_high = probe_env.action_space.high
    probe_env.close()

    policy, obs_normalizer = load_policy(model_path, env_id, device, obs_dim, act_dim)
    if obs_normalizer is None or obs_normalizer.count <= 0:
        print("Warning: checkpoint did not include an observation normalizer; using raw observations.")
    else:
        print(f"Loaded obs normalizer with count={obs_normalizer.count:.0f}")

    attempts = max(1, int(num_attempts))
    seeds: Sequence[Optional[int]]
    if seed is None:
        seeds = [None] * attempts
    else:
        seeds = [seed + i for i in range(attempts)]

    best_result = None

    for attempt_idx in range(attempts):
        attempt_seed = seeds[attempt_idx]
        frames, ep_return, steps = rollout_episode(
            env_id=env_id,
            policy=policy,
            device=device,
            obs_normalizer=obs_normalizer,
            action_low=action_low,
            action_high=action_high,
            num_steps=num_steps,
            seed=attempt_seed,
        )

        if not frames:
            print(f"Attempt {attempt_idx + 1}/{attempts} (seed={attempt_seed}) produced no frames; skipping.")
            continue

        print(
            f"Attempt {attempt_idx + 1}/{attempts} seed={attempt_seed} return={ep_return:.1f} steps={steps}"
        )

        if best_result is None or ep_return > best_result["return"]:
            best_result = {
                "frames": frames,
                "return": ep_return,
                "steps": steps,
                "attempt": attempt_idx + 1,
                "seed": attempt_seed,
            }

    if best_result is None:
        raise RuntimeError("No frames were captured; ensure the environment supports rgb_array rendering.")

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, best_result["frames"], fps=fps)
    print(
        "Saved GIF to "
        f"{gif_path} from attempt {best_result['attempt']}/{attempts} (seed={best_result['seed']}, "
        f"return={best_result['return']:.1f}, steps={best_result['steps']})"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Record a Walker2d rollout GIF using a saved policy.")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the saved policy checkpoint.")
    parser.add_argument("--gif_path", type=Path, default=Path("walker2d_rollout.gif"),
                        help="Where to write the output GIF.")
    parser.add_argument("--env_id", type=str, default="Walker2d-v5", help="Environment ID to load.")
    parser.add_argument("--num_steps", type=int, default=2000, help="Maximum number of steps to record.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the GIF.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the first attempt (subsequent attempts increment by 1).")
    parser.add_argument("--num_attempts", type=int, default=1, help="Number of attempts; best-return episode is saved.")
    return parser.parse_args()


def main():
    args = parse_args()
    record_gif(
        env_id=args.env_id,
        model_path=args.model_path,
        gif_path=args.gif_path,
        num_steps=args.num_steps,
        fps=args.fps,
        seed=args.seed,
        num_attempts=args.num_attempts,
    )


if __name__ == "__main__":
    main()
