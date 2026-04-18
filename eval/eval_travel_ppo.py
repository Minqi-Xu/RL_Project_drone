"""Evaluate a saved single-agent PPO travel model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
# Mirror the training script import setup so evaluation works from the repo root immediately.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drone_rl.travel_ppo import TravelPPOConfig, evaluate_saved_model, show_model_performance


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for travel-stage evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a saved PPO travel model.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to the saved .zip PPO model.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes.")
    parser.add_argument("--gui", action="store_true", help="Run one real-time GUI rollout (single scenario) and show plots.")
    return parser.parse_args()


def main() -> None:
    """Evaluate a saved travel policy and optionally run one GUI rollout."""
    args = parse_args()
    # Reuse one shared config so evaluation stays aligned with the training environment.
    config = TravelPPOConfig(n_eval_episodes=args.episodes)
    mean_reward, std_reward = evaluate_saved_model(
        model_path=args.model_path,
        config=config,
        n_eval_episodes=args.episodes,
    )
    print(f"[INFO] Mean reward: {mean_reward:.3f}")
    print(f"[INFO] Std reward: {std_reward:.3f}")

    if args.gui:
        output_folder = args.model_path.resolve().parent
        print("[INFO] Running one real-time rollout with GUI and plots.")
        show_model_performance(
            model_path=args.model_path,
            config=config,
            output_folder=output_folder,
            gui=True,
            plot=True,
        )


if __name__ == "__main__":
    main()
