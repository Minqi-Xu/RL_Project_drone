"""Plot evaluation reward versus training timesteps from Stable-Baselines3 evaluations.npz.

The script saves the output image in the same directory as the .npz file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def resolve_npz_path(input_path: Path) -> Path:
    """Resolve a file or directory input to an evaluations.npz path."""
    if input_path.is_dir():
        npz_path = input_path / "evaluations.npz"
    else:
        npz_path = input_path

    if not npz_path.exists():
        raise FileNotFoundError(f"Could not find file: {npz_path}")
    return npz_path


def load_eval_data(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load timesteps and reward statistics from an evaluations.npz file."""
    with np.load(npz_path) as data:
        if "timesteps" not in data or "results" not in data:
            raise KeyError("Expected keys 'timesteps' and 'results' in evaluations.npz")

        timesteps = data["timesteps"]
        results = data["results"]

    if results.ndim == 1:
        mean_rewards = results
        std_rewards = np.zeros_like(results)
    else:
        mean_rewards = np.mean(results, axis=1)
        std_rewards = np.std(results, axis=1)

    return timesteps, mean_rewards, std_rewards


def downsample_by_timestep_span(
    timesteps: np.ndarray,
    mean_rewards: np.ndarray,
    std_rewards: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample points so plot density scales with total training timestep span."""
    if max_points <= 1 or timesteps.shape[0] <= max_points:
        return timesteps, mean_rewards, std_rewards

    total_span = float(timesteps[-1] - timesteps[0])
    if total_span <= 0:
        step = int(np.ceil(timesteps.shape[0] / max_points))
        indices = np.arange(0, timesteps.shape[0], step, dtype=int)
        if indices[-1] != timesteps.shape[0] - 1:
            indices = np.append(indices, timesteps.shape[0] - 1)
        return timesteps[indices], mean_rewards[indices], std_rewards[indices]

    target_gap = total_span / float(max_points - 1)
    keep = [0]
    last_kept_ts = float(timesteps[0])

    for i in range(1, timesteps.shape[0] - 1):
        ts = float(timesteps[i])
        if ts - last_kept_ts >= target_gap:
            keep.append(i)
            last_kept_ts = ts

    if keep[-1] != timesteps.shape[0] - 1:
        keep.append(timesteps.shape[0] - 1)

    indices = np.array(keep, dtype=int)
    return timesteps[indices], mean_rewards[indices], std_rewards[indices]


def clip_to_end_timestep(
    timesteps: np.ndarray,
    mean_rewards: np.ndarray,
    std_rewards: np.ndarray,
    end_timestep: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep data up to end_timestep (inclusive), using the latest available record before it."""
    if end_timestep is None:
        return timesteps, mean_rewards, std_rewards

    if end_timestep < int(timesteps[0]):
        raise ValueError(
            f"--end-timestep ({end_timestep}) is smaller than first logged timestep ({int(timesteps[0])})."
        )

    last_idx = np.searchsorted(timesteps, end_timestep, side="right") - 1
    if last_idx < 0:
        raise ValueError("No data points are available at or before the requested end timestep.")

    return timesteps[: last_idx + 1], mean_rewards[: last_idx + 1], std_rewards[: last_idx + 1]


def plot_rewards(
    timesteps: np.ndarray,
    mean_rewards: np.ndarray,
    std_rewards: np.ndarray,
    output_path: Path,
) -> None:
    """Create and save the evaluation reward plot."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc

    plt.figure(figsize=(9, 5))
    plt.plot(timesteps, mean_rewards, linewidth=2, label="Mean eval reward")

    if np.any(std_rewards > 0):
        plt.fill_between(
            timesteps,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.2,
            label="+/-1 std",
        )

    plt.title("Evaluation Reward vs Training Timesteps")
    plt.xlabel("Training timesteps")
    plt.ylabel("Evaluation reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot evaluation reward from Stable-Baselines3 evaluations.npz",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to evaluations.npz or to the run directory containing it.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="evaluation_reward_vs_training_steps.png",
        help="Output image filename (saved beside evaluations.npz).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=200,
        help="Maximum plotted points after automatic downsampling. Default: 200",
    )
    parser.add_argument(
        "--end-timestep",
        type=int,
        default=None,
        help="Optional cutoff timestep n; plot uses data from start up to n (or latest point before n).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = resolve_npz_path(args.input)
    timesteps, mean_rewards, std_rewards = load_eval_data(npz_path)
    timesteps, mean_rewards, std_rewards = clip_to_end_timestep(
        timesteps,
        mean_rewards,
        std_rewards,
        end_timestep=args.end_timestep,
    )
    timesteps, mean_rewards, std_rewards = downsample_by_timestep_span(
        timesteps,
        mean_rewards,
        std_rewards,
        max_points=args.max_points,
    )

    output_path = npz_path.parent / args.output_name
    plot_rewards(timesteps, mean_rewards, std_rewards, output_path)

    print(f"[INFO] Loaded: {npz_path}")
    print(f"[INFO] Plotted points: {timesteps.shape[0]}")
    print(f"[INFO] Saved plot: {output_path}")


if __name__ == "__main__":
    main()
