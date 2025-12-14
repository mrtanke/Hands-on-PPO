# PPO Training Summary: Acrobot-v1 vs CartPole-v1

This repo contains a minimal PPO implementation (`ppo_minimal.py`) trained on two classic control tasks.
The plots below are based on the printed **evaluation return** during training.

<p align="center">
  <img src="images/ppo_training_cartpole.png" width="45%">
  <img src="images/ppo_training_acrobot.png" width="45%">
</p>



## Quick interpretation of the score

- **CartPole-v1**: higher is better. Typical “solved” threshold is an average return around **500**.
- **Acrobot-v1**: less negative is better.
  - Reward is approximately **-1 per step** until termination.
  - `-500` usually means the agent **timed out** at the max episode length (500 steps).
  - Around `-100` means the agent is typically solving in ~100 steps.

## Observed learning behavior

### Acrobot-v1 (500k steps)

- Long flat start at **-500** (no solves) for ~100k+ environment steps.
- Clear learning transition: evaluation rapidly improves from **-500 → ~-100**.
- After ~200k steps, performance stabilizes mostly around **~-70 to -110**, with occasional regressions/spikes.

**Takeaway:** the agent learned to reliably solve Acrobot (often in <100 steps), which is a strong outcome for a minimal PPO.

### CartPole-v1 (50k steps)

- Evaluation is **high-variance**: it sometimes spikes to **~350–400**, but also drops back to much lower returns.
- The training curve suggests intermittent good policies, but not consistently stable “solved” behavior within 50k steps.

**Takeaway:** learning happens, but it looks less stable than Acrobot in this particular run (likely due to evaluation variance, short training, and minimal PPO settings).

## Why the curves look different

- **Reward scales & termination rules differ** (`Acrobot` has a hard -500 floor; `CartPole` does not).
- **Training length**: Acrobot was run for 500k steps, CartPole for 50k.
