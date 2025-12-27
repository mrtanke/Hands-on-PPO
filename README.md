# PPO Implementation

PPO implementations for:
- **Discrete action**: `CartPole-v1`, `Acrobot-v1`, ...
- **Continuous action**: `Pendulum-v1`, `Walker2d-v5`, ...

## Install

This repo assumes you already have Python + PyTorch working.
Install the basic dependencies:

```powershell
pip install -r requirements.txt
```

## Run training

`main.py` now dispatches into the reorganized `algorithms/ppo` package, which contains all PPO-related code paths (training loops, models, utilities, and helper tools). SAC-specific work should live under `algorithms/sac`.

### Discrete (CartPole / Acrobot)

```powershell
python main.py --env_id CartPole-v1 --total_timesteps 50000
python main.py --env_id Acrobot-v1 --total_timesteps 500000
```

### Continuous (Pendulum)

```powershell
python main.py --env_id Pendulum-v1 --mode continuous --total_timesteps 300000
python main.py --env_id Walker-v5 --mode continuous --total_timesteps 1000000
```

## Outputs

- Training logs print periodic evaluation returns.
- Curves can be plotted via `plot_curve.py`.
- Notes and plots live under `notes/` and `images/`.

## Files

- `main.py`: lightweight entrypoint that calls the PPO CLI package.
- `algorithms/ppo/cli.py`: PPO training loops + CLI (discrete/continuous).
- `algorithms/ppo/agent.py`: PPO update + rollout collection helpers.
- `algorithms/ppo/models.py`: policy/value networks (discrete + continuous).
- `algorithms/ppo/utils.py`: GAE, evaluation helpers, stats saving.
- `algorithms/ppo/tools/`: ancillary tooling such as plotting curves or recording Walker rollouts.
- `algorithms/ppo/examples/`: environment-specific PPO scripts (e.g., SB3 baselines for CartPole, Pendulum, Walker).
- `algorithms/sac/`: placeholder for future SAC implementation artifacts.
