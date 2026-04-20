# Drone RL Project

Reinforcement learning project for quadcopter control in PyBullet using `gym-pybullet-drones` as the simulation environment.

Upstream environment repository: [utiasDSL/gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)

## Project Scope

The project follows a staged approach:

1. Train a baseline agent to reach and maintain a stable hover at a fixed target position.
2. Extend the control problem to safe landing from hover.
3. Explore point-to-point navigation as a separate RL agent.

In this project, the hovering task includes takeoff. The drone starts from the ground or a low-altitude initial state, rises to the target hover position, and then learns to remain stable around that point.

## Repository Structure

```text
Project/
├── RL_Project_drone/
│   ├── configs/                           # experiment and hyperparameter configuration files
│   ├── eval/                              # evaluation and analysis entrypoints
│   │   ├── eval_hover_ppo.py
│   │   ├── eval_hover_ppo_vel.py
│   │   ├── eval_landing_ppo.py
│   │   ├── eval_landing_rpm_ppo.py
│   │   ├── eval_travel_ppo.py
│   │   └── plot_eval_rewards.py
│   ├── results/                           # plots, logs, and generated artifacts
│   ├── src/drone_rl/                      # shared project code for training and evaluation
│   │   ├── __init__.py
│   │   ├── hover_ppo.py
│   │   ├── hover_ppo_vel.py
│   │   ├── landing_ppo.py
│   │   ├── travel_ppo.py
│   │   └── custom_envs/
│   │       ├── __init__.py
│   │       ├── project_base_aviary.py
│   │       ├── project_rl_base_aviary.py
│   │       ├── project_hover_aviary.py
│   │       ├── project_hover_vel_aviary.py
│   │       ├── project_landing_aviary.py
│   │       ├── project_travel_aviary.py
│   │       └── project_landing_aviary_rpm_legacy.py
│   └── train/                             # training entrypoints
│       ├── train_hover_ppo.py
│       ├── train_hover_ppo_vel.py
│       ├── train_landing_ppo.py
│       └── train_travel_ppo.py
└── gym-pybullet-drones/ # sibling upstream repository (set up separately)
```

## Immediate Goal

Current milestones are:

1. Train and evaluate a PPO agent that can take off and maintain stable hover near a target position.
2. Train and evaluate a separate PPO agent that starts near the hover target in an imbalanced state and performs safe landing.
3. Train and evaluate a separate PPO agent that travels from near `(0, 0, 1)` to `(4, 4, 1)` under PID-backed velocity control.

## Current Baselines

The repository now includes multiple single-agent PPO baselines based on the upstream `gym_pybullet_drones/examples/learn.py` example style.

Hover stage:

- `src/drone_rl/hover_ppo.py`: shared PPO configuration, environment builders, training, and evaluation helpers
- `train/train_hover_ppo.py`: trains a single-agent PPO hover policy (default action mode: `one_d_rpm`)
- `eval/eval_hover_ppo.py`: evaluates a saved PPO hover model trained with the default hover setup
- `src/drone_rl/hover_ppo_vel.py`: shared PPO utilities for hover with explicit 3D velocity actions
- `src/drone_rl/custom_envs/project_hover_vel_aviary.py`: hover environment variant where policy directly controls `[vx, vy, vz]` in `vel` mode
- `train/train_hover_ppo_vel.py`: trains a single-agent PPO hover policy with explicit 3D velocity action mode `vel`
- `eval/eval_hover_ppo_vel.py`: evaluates a saved PPO hover model trained with explicit 3D velocity action mode `vel`

Landing stage:

- `src/drone_rl/landing_ppo.py`: shared PPO configuration, environment builders, training, and evaluation helpers for landing (default action mode: `vel`)
- `src/drone_rl/custom_envs/project_landing_aviary.py`: landing environment with randomized starts near hover target `(0, 0, 1)`, including random roll/pitch/yaw and moderate random linear/angular velocity; in `vel` mode, PPO controls only lateral velocity references while vertical descent speed is scheduled by altitude
- `train/train_landing_ppo.py`: trains a single-agent PPO landing policy
- `eval/eval_landing_ppo.py`: evaluates a saved PPO landing model

Travel stage:

- `src/drone_rl/travel_ppo.py`: shared PPO configuration, environment builders, training, and evaluation helpers for traveling (default action mode: `vel`)
- `src/drone_rl/custom_envs/project_travel_aviary.py`: travel environment with randomized starts near `(0, 0, 1)`, target at `(4, 4, 1)`, and VEL PID control where PPO outputs normalized `[vx, vy, vz]`; observation state is `[dx, dy, dz, p, q, r, roll, pitch, yaw]`
- `train/train_travel_ppo.py`: trains a single-agent PPO travel policy
- `eval/eval_travel_ppo.py`: evaluates a saved PPO travel model

Training outputs are written to timestamped folders under `results/`.

## How To Run

### Prerequisite: set up `utiasDSL/gym-pybullet-drones` first

All training in this repository depends on the upstream `gym-pybullet-drones` package.
You must run training only from the same activated conda environment (`drones`) where `gym-pybullet-drones` is installed.

For environment setup details, follow the upstream README:
[utiasDSL/gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)

### Train Hover

From `RL_Project_drone/`, while that same conda environment is activated (`conda activate drones`):

```bash
python train/train_hover_ppo.py
```

Hover default control setup:

- Action mode: `ActionType.ONE_D_RPM` (`one_d_rpm`)
- PPO action: 1D normalized command in `[-1, 1]`
- Control meaning: the single action value is mapped to the thrust command for all 4 motors equally

Training arguments:

- `--total-timesteps`: total number of environment steps PPO will train for before stopping, unless early stopping happens first. Default: `10000000`
- `--eval-freq`: how often the callback pauses training to evaluate the current policy. Default: `2000`
- `--reward-threshold`: target evaluation reward used for early stopping when the policy is good enough. Default: `474.0`
- `--seed`: random seed for reproducible training runs. Default: `0`
- `--show-performance`: after training, open one real-time GUI rollout of the learned policy. Default: off
- `--plot`: when used with `--show-performance`, display the logged state plots after the rollout finishes. Default: off

Evaluate a saved hover model:

```bash
python eval/eval_hover_ppo.py --model-path results/<run-folder>/best_model.zip
```

Evaluation arguments:

- `--model-path`: path to the saved PPO model file, usually `best_model.zip` or `final_model.zip`. Required.
- `--episodes`: number of fast evaluation episodes used to compute mean and standard deviation reward. Default: `10`
- `--gui`: run one normal-speed single-scenario PyBullet rollout and show the plots. Default: off

Without `--gui`, evaluation runs only the fast metric check. With `--gui`, the script first runs the fast evaluation (using `--episodes`) and then shows one normal-speed single-scenario rollout with plots. If `best_model.zip` is not available, you can also evaluate `final_model.zip`.

### Train Hover (VEL Action Mode)

From `RL_Project_drone/`, while that same conda environment is activated (`conda activate drones`):

```bash
python train/train_hover_ppo_vel.py
```

Training arguments:

- `--total-timesteps`: total number of environment steps PPO will train for before stopping, unless early stopping happens first. Default: `1000000`
- `--eval-freq`: how often the callback pauses training to evaluate the current policy. Default: `2000`
- `--reward-threshold`: target evaluation reward used for early stopping when the policy is good enough. Default: `474.0`
- `--seed`: random seed for reproducible training runs. Default: `0`
- `--show-performance`: after training, open one real-time GUI rollout of the learned policy. Default: off
- `--plot`: when used with `--show-performance`, display the logged state plots after the rollout finishes. Default: off

Hover VEL control setup:

- Action mode: `ActionType.VEL` (PID-backed velocity control)
- PPO action: 3D normalized velocity command `[vx, vy, vz]` in `[-1, 1]`
- No scheduled vertical command is injected; `vz` is learned by the policy
- Observation state: `[dx, dy, dz, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]` plus action-history buffer (buffer is reset each episode)
- Reward (current): `alpha*(e_(t-1)-e_t) - beta*e_t^2 + b*1(e_t<0.1) - w_smooth*||a_t-a_(t-1)||_1 - w_tilt*(roll_excess + pitch_excess) - crash_penalty`
- Current weights: `alpha=1.0`, `beta=2.0`, `b=0.5`, `w_smooth=0.2`, `w_tilt=0.1`, `tilt_threshold=0.1`, `crash_penalty=600`
- Termination/truncation: `terminated=True` on crash-like states (`|x|>1.5`, `|y|>1.5`, `z>2.0`, `|roll|>0.4`, or `|pitch|>0.4`), `truncated=True` only at hard episode time limit (`8s`)

Evaluate a saved hover VEL model:

```bash
python eval/eval_hover_ppo_vel.py --model-path results/<run-folder>/best_model.zip
```

Evaluation arguments:

- `--model-path`: path to the saved PPO model file, usually `best_model.zip` or `final_model.zip`. Required.
- `--episodes`: number of fast evaluation episodes used to compute mean and standard deviation reward. Default: `25`
- `--gui`: run one normal-speed single-scenario PyBullet rollout and show the plots. Default: off

Training callback evaluation for hover VEL also uses `25` episodes per evaluation checkpoint.

### Train Landing

From `RL_Project_drone/`, while that same conda environment is activated (`conda activate drones`):

```bash
python train/train_landing_ppo.py
```

Landing control setup (current default):

- Action mode: `ActionType.VEL` (PID-backed velocity control)
- PPO action: 2D normalized lateral command `[ax, ay]` in `[-1, 1]`
- Vertical command: not learned directly; descent speed is altitude-scheduled:
  - altitude `> 0.5 m`: `0.5 m/s`
  - altitude `0.2 m` to `0.5 m`: linearly decreases from `0.5` to `0.2 m/s`
  - altitude `< 0.2 m`: `0.2 m/s`
- Reward shaping includes position/velocity/attitude/angular-rate costs and anti-aggressive terms (`||a_t-a_{t-1}||` and acceleration proxy)

Training arguments:

- `--total-timesteps`: total number of environment steps PPO will train for before stopping, unless early stopping happens first. Default: `10000000`
- `--eval-freq`: how often the callback pauses training to evaluate the current policy. Default: `2000`
- `--reward-threshold`: target evaluation reward used for early stopping when the policy is good enough. Default: `52.2`
- `--seed`: random seed for reproducible training runs. Default: `0`
- `--show-performance`: after training, open one real-time GUI rollout of the learned policy. Default: off
- `--plot`: when used with `--show-performance`, display the logged state plots after the rollout finishes. Default: off

Evaluate a saved landing model:

```bash
python eval/eval_landing_ppo.py --model-path results/<run-folder>/best_model.zip
```

Evaluate a legacy landing model trained with 4-motor RPM actions (previous code version, no longer used for current training):

```bash
python eval/eval_landing_rpm_ppo.py --model-path results/<run-folder>/best_model.zip
```

Note: this legacy evaluation path is only for loading/evaluating historical models saved from the previous RPM-based landing method. It is not part of the current landing training setup.

Evaluation arguments:

- `--model-path`: path to the saved PPO model file, usually `best_model.zip` or `final_model.zip`. Required.
- `--episodes`: number of fast evaluation episodes used to compute mean and standard deviation reward. Default: `50`
- `--gui`: run one normal-speed single-scenario PyBullet rollout and show the plots. Default: off

Without `--gui`, evaluation runs only the fast metric check. With `--gui`, the script first runs the fast evaluation (using `--episodes`) and then shows one normal-speed single-scenario rollout with plots. If `best_model.zip` is not available, you can also evaluate `final_model.zip`.

### Train Travel

From `RL_Project_drone/`, while that same conda environment is activated (`conda activate drones`):

```bash
python train/train_travel_ppo.py
```

Travel control setup (current default):

- Action mode: `ActionType.VEL` (PID-backed velocity control)
- PPO action: 3D normalized velocity command `[vx, vy, vz]` in `[-1, 1]`
- Start condition: random initial position near `(0, 0, 1)` with randomized linear velocity, angular velocity, roll, pitch, and yaw
- Target: fixed point `(4, 4, 1)`
- Observation state: `[dx, dy, dz, p, q, r, roll, pitch, yaw]` where `d*` is position error to target
- GUI visualization: a persistent reference line is drawn from `(0, 0, 1)` to `(4, 4, 1)`

Training arguments:

- `--total-timesteps`: total number of environment steps PPO will train for before stopping, unless early stopping happens first. Default: `10000000`
- `--eval-freq`: how often the callback pauses training to evaluate the current policy. Default: `2000`
- callback evaluation episodes: each callback evaluation runs `50` episodes (`n_eval_episodes=50` in travel PPO config)
- `--reward-threshold`: target evaluation reward used for early stopping when the policy is good enough. Default: `45.0`
- `--seed`: random seed for reproducible training runs. Default: `0`
- `--show-performance`: after training, open one real-time GUI rollout of the learned policy. Default: off
- `--plot`: when used with `--show-performance`, display the logged state plots after the rollout finishes. Default: off

Evaluate a saved travel model:

```bash
python eval/eval_travel_ppo.py --model-path results/<run-folder>/best_model.zip
```

Evaluation arguments:

- `--model-path`: path to the saved PPO model file, usually `best_model.zip` or `final_model.zip`. Required.
- `--episodes`: number of fast evaluation episodes used to compute mean and standard deviation reward. Default: `50`
- `--gui`: run one normal-speed single-scenario PyBullet rollout and show the plots. Default: off

Without `--gui`, evaluation runs only the fast metric check. With `--gui`, the script first runs the fast evaluation (using `--episodes`) and then shows one normal-speed single-scenario rollout with plots. If `best_model.zip` is not available, you can also evaluate `final_model.zip`.

### Plot Evaluation Reward vs Training Steps

Use the standalone plotting script to read `evaluations.npz` and save a reward curve image in the same run folder.

```bash
python eval/plot_eval_rewards.py --input results/<run-folder>
```

You can also point directly to the file:

```bash
python eval/plot_eval_rewards.py --input results/<run-folder>/evaluations.npz
```

Plot arguments:

- `--input`: path to either a run directory containing `evaluations.npz` or the `evaluations.npz` file itself. Required.
- `--output-name`: output image filename saved beside `evaluations.npz`. Default: `evaluation_reward_vs_training_steps.png`
- `--max-points`: maximum plotted points after automatic downsampling based on total training timestep span. Default: `200`
- `--end-timestep`: optional cutoff timestep `n`; plot uses data from the first logged timestep up to `n` (or the latest logged point before `n`).

## Dependencies

- Python
- PyBullet
- `gym-pybullet-drones`
- `stable-baselines3`
- `gymnasium`
- `matplotlib` (required for `eval/plot_eval_rewards.py`)
- TensorBoard (required by the current training logger setup)

## Version Control

This repository stores only the project-specific code and artifacts configuration. The upstream simulator is kept separately in the sibling `gym-pybullet-drones` directory.
