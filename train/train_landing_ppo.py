"""Train a single-agent PPO baseline for the landing task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
# Allow the script to import the local project package without requiring installation first.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drone_rl.landing_ppo import LandingPPOConfig, show_model_performance, train_landing_agent


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for landing-stage training."""
    parser = argparse.ArgumentParser(description="Train a PPO landing baseline.")
    parser.add_argument("--total-timesteps", type=int, default=10000000, help="Total PPO training timesteps.")
    parser.add_argument("--eval-freq", type=int, default=2000, help="Evaluation frequency in steps.")
    parser.add_argument("--reward-threshold", type=float, default=250.0, help="Early-stop reward threshold.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for training.")
    parser.add_argument("--show-performance", action="store_true", help="Render a post-training rollout.")
    parser.add_argument("--plot", action="store_true", help="Plot the logged rollout state after playback.")
    return parser.parse_args()


def main() -> None:
    """Run landing PPO training and optionally show policy rollout performance."""
    args = parse_args()
    # Override only the CLI-exposed settings and keep other defaults in shared config.
    config = LandingPPOConfig(
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        reward_threshold=args.reward_threshold,
        seed=args.seed,
    )
    run_dir = train_landing_agent(config=config)
    print("[INFO] Training finished. Run directory:", run_dir)

    if args.show_performance:
        best_model_path = run_dir / "best_model.zip"
        final_model_path = run_dir / "final_model.zip"
        model_path = best_model_path if best_model_path.exists() else final_model_path
        print("[INFO] Showing policy performance from:", model_path)
        show_model_performance(
            model_path=model_path,
            config=config,
            output_folder=run_dir,
            gui=True,
            plot=args.plot,
        )


if __name__ == "__main__":
    main()
