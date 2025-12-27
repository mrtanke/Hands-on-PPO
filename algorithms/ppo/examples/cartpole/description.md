# What I Do
There are two scripts:
- ppo_cartpole_sb3.py (training)
- enjoy_cartpole_sb3.py (rendering)
---
The process of ppo training in gymnasium is:
1. Call gymnasium to create the CartPole environment.
    - gynnasium is a RL environment library that provides standard environments where RL agents can be trained and rendered.
2. Create the PPO agent
    - we just set three types of hyperparameters:
        - "MlpPolicy" -> define the model type for the policy and value network.
        - env -> places thiss PPO agent to the environment for training and rendering.
        - verbose=1 -> controls logging output.
3. Train the agent for a specified number of timesteps
    - if set up "render_mode=human", then the training process will be rendered in the PyGame.
    - Step 1: collect an episode by running the current policy in the env.
    - Step 2: compute the GAE advantage using the reward and critic(value) function.
    - Step 3: optimize the policy and critic network with PPO loss on this episode.
    - Step 4: repeat until the specific number of timesteps is reached.
4. Save the trained model for the next call.

The process in <enjoy_cartpole_sb3.py>:
1. Create the CartPole environment.
2. Load the trained PPO model from previous stored .zip file.
3. Reset the environment to start a new episode
    - return obs as current environment state 
4. Run the trained agent in the environment.