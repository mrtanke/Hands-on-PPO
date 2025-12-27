"""Project entry point dispatching to the PPO CLI module."""
from algorithms.ppo.cli import main as ppo_main


if __name__ == "__main__":
    ppo_main()
