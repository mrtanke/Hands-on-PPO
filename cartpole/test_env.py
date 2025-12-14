import gymnasium as gym

# Create the environment without rendering
env = gym.make("CartPole-v1", render_mode=None)
# Reset the environment to start a new episode
obs, info = env.reset()
print("Initial Observation:", obs)

# Take a few random steps in the environment
for t in range(5):
    action = env.action_space.sample()  # Sample a random action
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {t}: action={action}, observation={obs}, reward={reward}, done={terminated or truncated}")

# Close the environment
env.close()