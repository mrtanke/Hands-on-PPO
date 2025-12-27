import gymnasium as gym
from stable_baselines3 import PPO

def main():
    # 1. Create the Pendulum environment
    env = gym.make("Pendulum-v1", render_mode="human")

    # 2. Create model foro continuous actions
    model = PPO(
        "MlpPolicy",
        env, 
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
    )

    # 3. Train the model
    model.learn(total_timesteps=10000)

    # 4. Save the model
    model.save("ppo_pendulum_sb3")

    env.close()

if __name__ == "__main__":
    main()