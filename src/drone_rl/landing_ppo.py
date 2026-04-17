"""Shared PPO landing training utilities built on top of gym-pybullet-drones."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from drone_rl.custom_envs.project_landing_aviary import ProjectLandingAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import sync


@dataclass(frozen=True)
class LandingPPOConfig:
    """Configuration for the single-agent landing baseline."""

    # Use the kinematic state vector (position, attitude, velocities, action history).
    observation: str = "kin"
    # Use PID-guided velocity actions for landing.
    action: str = "vel"
    seed: int = 0
    num_envs: int = 1
    total_timesteps: int = 1e7
    eval_freq: int = 2000
    n_eval_episodes: int = 50
    # Differential shaping reward uses smaller per-step increments than terminal-bonus rewards.
    reward_threshold: float = 52.2
    verbose: int = 1
    deterministic_eval: bool = True


def project_root() -> Path:
    # Resolve paths relative to the repository root so scripts can be run from anywhere.
    return Path(__file__).resolve().parents[2]


def default_results_dir() -> Path:
    return project_root() / "results"


def build_run_dir(base_dir: Path) -> Path:
    # Keep each training run in its own timestamped folder to avoid overwrites.
    run_dir = base_dir / f"landing_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_obs_type(config: LandingPPOConfig) -> ObservationType:
    return ObservationType(config.observation)


def get_action_type(config: LandingPPOConfig) -> ActionType:
    return ActionType(config.action)


def make_train_env(config: LandingPPOConfig):
    # Vectorized env creation matches Stable-Baselines3's expected training interface.
    return make_vec_env(
        ProjectLandingAviary,
        env_kwargs={"obs": get_obs_type(config), "act": get_action_type(config)},
        n_envs=config.num_envs,
        seed=config.seed,
    )


def make_eval_env(config: LandingPPOConfig, gui: bool = False):
    # Evaluation uses a plain ProjectLandingAviary instance so we can optionally render rollouts.
    return ProjectLandingAviary(
        gui=gui,
        record=False,
        obs=get_obs_type(config),
        act=get_action_type(config),
    )


def build_model(train_env, run_dir: Path, config: LandingPPOConfig) -> PPO:
    # The policy and TensorBoard logging setup are centralized here so train scripts stay small.
    return PPO(
        "MlpPolicy",
        train_env,
        tensorboard_log=str(run_dir / "tb"),
        verbose=config.verbose,
    )


def build_eval_callback(eval_env, run_dir: Path, config: LandingPPOConfig) -> EvalCallback:
    # Stop training early once the landing reward is consistently good enough.
    stop_on_reward = StopTrainingOnRewardThreshold(
        reward_threshold=config.reward_threshold,
        verbose=config.verbose,
    )
    return EvalCallback(
        eval_env,
        callback_on_new_best=stop_on_reward,
        verbose=config.verbose,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        eval_freq=config.eval_freq,
        deterministic=config.deterministic_eval,
        render=False,
    )


def print_training_progress(run_dir: Path) -> None:
    """Print saved evaluation rewards in the same format as the hover helper."""
    evaluations_path = run_dir / "evaluations.npz"
    if not evaluations_path.exists():
        print("[WARNING] evaluations.npz was not found, so no training progression is available.")
        return

    with np.load(evaluations_path) as data:
        timesteps = data["timesteps"]
        results = data["results"][:, 0]
        print("Data from evaluations.npz")
        for index in range(timesteps.shape[0]):
            print(f"{timesteps[index]},{results[index]}")


def train_landing_agent(config: LandingPPOConfig, results_dir: Path | None = None) -> Path:
    """Train a PPO policy for the landing task and return the run directory."""
    results_dir = results_dir or default_results_dir()
    run_dir = build_run_dir(results_dir)

    # Keep training and evaluation environments separate so evaluation does not interfere with learning.
    train_env = make_train_env(config)
    eval_env = make_eval_env(config)

    print("[INFO] Action space:", train_env.action_space)
    print("[INFO] Observation space:", train_env.observation_space)
    print("[INFO] Saving run artifacts to:", run_dir)

    model = build_model(train_env, run_dir, config)
    eval_callback = build_eval_callback(eval_env, run_dir, config)

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=eval_callback,
        log_interval=100,
    )

    # Save a final checkpoint even if the best-model callback never fires.
    final_model_path = run_dir / "final_model"
    model.save(str(final_model_path))
    print("[INFO] Final model saved to:", final_model_path.with_suffix(".zip"))
    print_training_progress(run_dir)

    train_env.close()
    eval_env.close()
    return run_dir


def evaluate_saved_model(
    model_path: Path,
    config: LandingPPOConfig,
    n_eval_episodes: int | None = None,
) -> tuple[float, float]:
    """Compute aggregate evaluation metrics for a saved landing policy."""
    eval_env = make_eval_env(config, gui=False)
    model = PPO.load(str(model_path))
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes or config.n_eval_episodes,
    )
    eval_env.close()
    return mean_reward, std_reward


def show_model_performance(
    model_path: Path,
    config: LandingPPOConfig,
    output_folder: Path,
    gui: bool = True,
    plot: bool = True,
) -> tuple[float, float]:
    """Run one rendered rollout for landing and stop at the first episode end."""
    model = PPO.load(str(model_path))
    test_env = make_eval_env(config, gui=gui)
    test_env_nogui = make_eval_env(config, gui=False)
    logger = Logger(
        logging_freq_hz=int(test_env.CTRL_FREQ),
        num_drones=1,
        output_folder=str(output_folder),
        colab=False,
    )

    mean_reward, std_reward = evaluate_policy(
        model,
        test_env_nogui,
        n_eval_episodes=1,
    )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    max_rollout_steps = (test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ
    for step in range(max_rollout_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs_data = obs.squeeze()
        action_data = action.squeeze()
        print(
            "Obs:",
            obs,
            "\tAction",
            action,
            "\tReward:",
            reward,
            "\tTerminated:",
            terminated,
            "\tTruncated:",
            truncated,
        )

        if get_obs_type(config) == ObservationType.KIN and obs_data.shape[0] >= 15:
            logger.log(
                drone=0,
                timestamp=step / test_env.CTRL_FREQ,
                state=np.hstack([obs_data[0:3], np.zeros(4), obs_data[3:15], action_data]),
                control=np.zeros(12),
            )

        test_env.render()
        sync(step, start, test_env.CTRL_TIMESTEP)
        if terminated or truncated:
            print("[INFO] One rollout finished after termination/truncation.")
            break

    test_env.close()
    test_env_nogui.close()

    if plot and get_obs_type(config) == ObservationType.KIN:
        logger.plot()

    return mean_reward, std_reward
