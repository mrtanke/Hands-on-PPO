import numpy as np
import matplotlib.pyplot as plt

def main():
    data = np.load("cartpole_training_stats.npz")
    timesteps = data["timesteps"]
    rewards = data["rewards"]

    plt.plot(timesteps, rewards)
    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation Reward")
    plt.title("PPO CartPole Training Curve (from-scratch)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()