"""Evaluate a saved single-agent PPO hover model trained with VEL actions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
# Mirror the training script import setup so evaluation works from the repo root immediately.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drone_rl.hover_ppo_vel import HoverVelPPOConfig, evaluate_saved_vel_model, show_vel_model_performance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved PPO hover model trained with ActionType.VEL.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to the saved .zip PPO model.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument("--gui", action="store_true", help="After fast evaluation, run one real-time GUI rollout and show plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Hover VEL uses explicit 3D velocity actions [vx, vy, vz].
    config = HoverVelPPOConfig(n_eval_episodes=args.episodes)
    mean_reward, std_reward = evaluate_saved_vel_model(
        model_path=args.model_path,
        config=config,
        n_eval_episodes=args.episodes,
    )
    print(f"[INFO] Mean reward: {mean_reward:.3f}")
    print(f"[INFO] Std reward: {std_reward:.3f}")

    if args.gui:
        output_folder = args.model_path.resolve().parent
        print("[INFO] Running one real-time VEL hover rollout with GUI and plots.")
        show_vel_model_performance(
            model_path=args.model_path,
            config=config,
            output_folder=output_folder,
            gui=True,
            plot=True,
        )


if __name__ == "__main__":
    main()
