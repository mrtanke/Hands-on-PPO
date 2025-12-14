import gymnasium as gym
from stable_baselines3 import PPO

def main():
    # 1. Create the CartPole environment
    env = gym.make("CartPole-v1", render_mode="human")

    # 2. Create the PPO agent
    model = PPO("MlpPolicy", env, verbose=1)

    # 3. Train the agent for a specified number of timesteps
    model.learn(total_timesteps=50000)

    # 4. Save the trained model
    model.save("ppo_cartpole_sb3")

    env.close()

if __name__ == "__main__":
    main()