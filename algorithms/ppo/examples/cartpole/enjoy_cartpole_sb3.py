import gymnasium as gym
from stable_baselines3 import PPO

def main():
    # 1. Create the CartPole environment
    env = gym.make("CartPole-v1", render_mode="human")

    # 2. Load the trained PPO model from file
    model = PPO.load("ppo_cartpole_sb3")

    # 3. Reset the environment to start a new episode
    obs, info = env.reset() # obs -> initial state
    done = False

    # 4. Run the trained agent in the environment
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True) # deterministic -> choose the best move
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            obs, info = env.reset()
    
    env.close()

if __name__ == "__main__":
    main()