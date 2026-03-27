# Drone RL Project

Reinforcement learning project for quadcopter control in PyBullet using `gym-pybullet-drones` as the simulation environment.

## Project Scope

The project follows a staged approach:

1. Train a baseline agent to hover at a fixed position.
2. Extend the control problem to takeoff and landing.
3. Explore point-to-point navigation as a separate RL agent.

## Repository Structure

- `src/`: shared project code, wrappers, utilities, and reward logic
- `train/`: training scripts for each task
- `eval/`: evaluation scripts and rollout testing
- `configs/`: experiment and hyperparameter configuration files
- `models/`: saved model checkpoints
- `results/`: plots, logs, and generated artifacts

## Immediate Goal

The first milestone is to train and evaluate a PPO agent that can maintain stable hover near a target position in simulation.

## Dependencies

- Python
- PyBullet
- `gym-pybullet-drones`
- `stable-baselines3`
- `gymnasium`

## Version Control

This repository stores only the project-specific code and artifacts configuration. The upstream simulator is kept separately in the sibling `gym-pybullet-drones` directory.
