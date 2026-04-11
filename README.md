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
│   ├── configs/         # experiment and hyperparameter configuration files
│   ├── eval/            # evaluation entrypoints
│   ├── models/          # saved model checkpoints
│   ├── results/         # plots, logs, and generated artifacts
│   ├── src/drone_rl/    # shared project code for training and evaluation
│   └── train/           # training entrypoints
└── gym-pybullet-drones/ # sibling upstream repository (set up separately)
```

## Immediate Goal

The first milestone is to train and evaluate a PPO agent that can take off and maintain stable hover near a target position in simulation.

## Current Baseline

The repository now includes a single-agent PPO hover baseline based on the upstream `gym_pybullet_drones/examples/learn.py` example:

- `src/drone_rl/hover_ppo.py`: shared PPO configuration, environment builders, training, and evaluation helpers
- `train/train_hover_ppo.py`: trains a single-agent PPO hover policy
- `eval/eval_hover_ppo.py`: evaluates a saved PPO hover model

Training outputs are written to timestamped folders under `results/`.

## How To Run

### Prerequisite: set up `utiasDSL/gym-pybullet-drones` first

All training in this repository depends on the upstream `gym-pybullet-drones` package.
You must run training only from the same activated conda environment (`drones`) where `gym-pybullet-drones` is installed.

For environment setup details, follow the upstream README:
[utiasDSL/gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)

### Train

From `RL_Project_drone/`, while that same conda environment is activated (`conda activate drones`):

```bash
python train/train_hover_ppo.py
```

Training arguments:

- `--total-timesteps`: total number of environment steps PPO will train for before stopping, unless early stopping happens first. Default: `10000000`
- `--eval-freq`: how often the callback pauses training to evaluate the current policy. Default: `2000`
- `--reward-threshold`: target evaluation reward used for early stopping when the policy is good enough. Default: `474.0`
- `--seed`: random seed for reproducible training runs. Default: `0`
- `--show-performance`: after training, open one real-time GUI rollout of the learned policy. Default: off
- `--plot`: when used with `--show-performance`, display the logged state plots after the rollout finishes. Default: off

Evaluate a saved model:

```bash
python eval/eval_hover_ppo.py --model-path results/<run-folder>/best_model.zip
```

Evaluation arguments:

- `--model-path`: path to the saved PPO model file, usually `best_model.zip` or `final_model.zip`. Required.
- `--episodes`: number of fast evaluation episodes used to compute mean and standard deviation reward. Default: `10`
- `--gui`: after the fast evaluation, run one normal-speed PyBullet rollout and show the plots. Default: off

Without `--gui`, evaluation runs only the fast multi-episode metric check. With `--gui`, the script first runs the fast evaluation and then shows one normal-speed rollout with plots. If `best_model.zip` is not available, you can also evaluate `final_model.zip`.

## Dependencies

- Python
- PyBullet
- `gym-pybullet-drones`
- `stable-baselines3`
- `gymnasium`
- TensorBoard (required by the current training logger setup)

## Version Control

This repository stores only the project-specific code and artifacts configuration. The upstream simulator is kept separately in the sibling `gym-pybullet-drones` directory.
