"""Evaluate a saved single-agent PPO landing model (legacy RPM version)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
# Mirror the training script import setup so evaluation works from the repo root immediately.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drone_rl.custom_envs.project_landing_aviary_rpm_legacy import ProjectLandingAviary as LegacyLandingRPMAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import sync


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for legacy landing evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a saved legacy PPO landing model (RPM action mode).")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to the saved .zip PPO model.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes.")
    parser.add_argument("--gui", action="store_true", help="Run one real-time GUI rollout (single scenario) and show plots.")
    return parser.parse_args()


def main() -> None:
    """Evaluate a saved legacy RPM landing policy and optionally run one GUI rollout."""
    args = parse_args()
    model = PPO.load(str(args.model_path))

    eval_env = LegacyLandingRPMAviary(
        gui=False,
        record=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
    )
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.episodes,
    )
    eval_env.close()
    print(f"[INFO] Mean reward: {mean_reward:.3f}")
    print(f"[INFO] Std reward: {std_reward:.3f}")

    if args.gui:
        print("[INFO] Running one real-time rollout with GUI (legacy RPM mode).")
        test_env = LegacyLandingRPMAviary(
            gui=True,
            record=False,
            obs=ObservationType.KIN,
            act=ActionType.RPM,
        )
        obs, _ = test_env.reset(seed=42, options={})
        start = time.time()
        max_rollout_steps = (test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ
        for step in range(max_rollout_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            print(
                "Obs shape:",
                np.array(obs).shape,
                "\tAction:",
                action,
                "\tReward:",
                reward,
                "\tTerminated:",
                terminated,
                "\tTruncated:",
                truncated,
            )
            test_env.render()
            sync(step, start, test_env.CTRL_TIMESTEP)
            if terminated or truncated:
                print("[INFO] One rollout finished after termination/truncation.")
                break
        test_env.close()


if __name__ == "__main__":
    main()
