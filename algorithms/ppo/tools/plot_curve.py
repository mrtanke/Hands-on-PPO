import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="CartPole-v1")
    return parser.parse_args()

def main():
    args = parse_args()
    env_id = args.env_id

    data = np.load(f"{env_id}_training_stats.npz")
    timesteps = data["timesteps"]
    rewards = data["rewards"]

    plt.plot(timesteps, rewards)
    plt.xlabel("Environment steps")
    plt.ylabel("Eval reward")
    plt.title(f"PPO Training Curve ({env_id})")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
