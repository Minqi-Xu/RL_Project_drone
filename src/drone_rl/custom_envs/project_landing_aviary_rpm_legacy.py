"""Local project landing aviary task environment.

Initial version structure follows: drone_rl.custom_envs.project_hover_aviary
This environment is maintained separately so landing-stage tuning does not affect
hover-stage training code.
"""

from __future__ import annotations

import numpy as np
import pybullet as p

from drone_rl.custom_envs.project_rl_base_aviary import ProjectBaseRLAviary
from gym_pybullet_drones.utils.enums import ActionType, DroneModel, ObservationType, Physics


class ProjectLandingAviary(ProjectBaseRLAviary):
    """Single-agent RL problem: recover from imbalance and land safely."""

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        gui: bool = False,
        record: bool = False,
        obs: ObservationType = ObservationType.KIN,
        act: ActionType = ActionType.RPM,
    ):
        """Initialization of a single-agent RL environment for landing.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs : ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position.
        initial_rpys : ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing initial roll/pitch/yaw (radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision).
        act : ActionType, optional
            The action type. Landing defaults to full 4-motor RPM control.

        """
        # Hover-stage target used as the center of the randomized landing start.
        self.HOVER_TARGET_POS = np.array([0.0, 0.0, 1.0])
        # Reference touchdown point near the world origin.
        self.TARGET_XY = np.array([0.0, 0.0])
        # Landing can take longer than hover because the agent must stabilize first.
        self.EPISODE_LEN_SEC = 10

        # Reset randomization ranges around the hover target position.
        self.INIT_XY_RANGE_M = 0.12
        self.INIT_Z_OFFSET_RANGE_M = (-0.05, 0.05)

        # Keep initial orientation and velocities "random but not crazy".
        self.INIT_RP_RANGE_RAD = 0.25
        self.INIT_YAW_RANGE_RAD = np.pi
        self.INIT_VEL_XY_RANGE_MPS = 0.35
        self.INIT_VEL_Z_RANGE_MPS = 0.25
        self.INIT_ANG_VEL_RANGE_RADPS = 1.20

        # Success thresholds for a stable touchdown state.
        self.SUCCESS_XY_ERR_M = 0.08
        self.SUCCESS_ALTITUDE_M = 0.08
        self.SUCCESS_SPEED_MPS = 0.18
        self.SUCCESS_TILT_RAD = 0.22
        self.SUCCESS_ANG_SPEED_RADPS = 0.65

        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
        )

        # Do not treat literal contact with the plane as immediate failure.
        # This lets the policy learn to complete controlled touchdowns.
        p.setCollisionFilterPair(
            bodyUniqueIdA=self.PLANE_ID,
            bodyUniqueIdB=self.DRONE_IDS[0],
            linkIndexA=-1,
            linkIndexB=-1,
            enableCollision=0,
            physicsClientId=self.CLIENT,
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Resets the environment and randomizes landing-stage initial conditions.

        The drone starts around (0, 0, 1) with moderate random translational and
        angular velocities to represent imbalanced conditions.

        """
        obs, info = super().reset(seed=seed, options=options)

        # Keep the plane-drone collision disabled after each reset.
        p.setCollisionFilterPair(
            bodyUniqueIdA=self.PLANE_ID,
            bodyUniqueIdB=self.DRONE_IDS[0],
            linkIndexA=-1,
            linkIndexB=-1,
            enableCollision=0,
            physicsClientId=self.CLIENT,
        )

        rng = np.random.default_rng(seed)

        # Random position near the hover target (0, 0, 1).
        init_pos = np.array(
            [
                self.HOVER_TARGET_POS[0] + rng.uniform(-self.INIT_XY_RANGE_M, self.INIT_XY_RANGE_M),
                self.HOVER_TARGET_POS[1] + rng.uniform(-self.INIT_XY_RANGE_M, self.INIT_XY_RANGE_M),
                self.HOVER_TARGET_POS[2] + rng.uniform(self.INIT_Z_OFFSET_RANGE_M[0], self.INIT_Z_OFFSET_RANGE_M[1]),
            ]
        )

        # Random roll/pitch with free yaw.
        init_rpy = np.array(
            [
                rng.uniform(-self.INIT_RP_RANGE_RAD, self.INIT_RP_RANGE_RAD),
                rng.uniform(-self.INIT_RP_RANGE_RAD, self.INIT_RP_RANGE_RAD),
                rng.uniform(-self.INIT_YAW_RANGE_RAD, self.INIT_YAW_RANGE_RAD),
            ]
        )

        # Random linear and angular velocity (bounded to avoid unstable launches).
        init_linear_vel = np.array(
            [
                rng.uniform(-self.INIT_VEL_XY_RANGE_MPS, self.INIT_VEL_XY_RANGE_MPS),
                rng.uniform(-self.INIT_VEL_XY_RANGE_MPS, self.INIT_VEL_XY_RANGE_MPS),
                rng.uniform(-self.INIT_VEL_Z_RANGE_MPS, self.INIT_VEL_Z_RANGE_MPS),
            ]
        )
        init_angular_vel = np.array(
            [
                rng.uniform(-self.INIT_ANG_VEL_RANGE_RADPS, self.INIT_ANG_VEL_RANGE_RADPS),
                rng.uniform(-self.INIT_ANG_VEL_RANGE_RADPS, self.INIT_ANG_VEL_RANGE_RADPS),
                rng.uniform(-self.INIT_ANG_VEL_RANGE_RADPS, self.INIT_ANG_VEL_RANGE_RADPS),
            ]
        )

        # Apply randomized state directly in PyBullet.
        p.resetBasePositionAndOrientation(
            self.DRONE_IDS[0],
            init_pos,
            p.getQuaternionFromEuler(init_rpy),
            physicsClientId=self.CLIENT,
        )
        p.resetBaseVelocity(
            self.DRONE_IDS[0],
            linearVelocity=init_linear_vel,
            angularVelocity=init_angular_vel,
            physicsClientId=self.CLIENT,
        )

        # Refresh cached kinematics and return consistent first observation.
        self._updateAndStoreKinematicInformation()
        return self._computeObs(), info

    def _computeReward(self):
        """Computes reward for stabilization + controlled descent + safe touchdown."""
        state = self._getDroneStateVector(0)

        # Position and attitude terms.
        xy_error = np.linalg.norm(state[0:2] - self.TARGET_XY)
        altitude = state[2]
        tilt_norm = np.linalg.norm(state[7:9])

        # Velocity terms.
        linear_speed = np.linalg.norm(state[10:13])
        angular_speed = np.linalg.norm(state[13:16])

        # Encourage approach to landing point, descending, and balancing.
        reward = 0.0
        reward += 1.8 * np.exp(-4.0 * xy_error)
        reward += 1.4 * np.exp(-2.5 * altitude)
        reward += 1.2 * np.exp(-3.5 * tilt_norm)
        reward += 1.0 * np.exp(-2.0 * linear_speed)
        reward += 0.8 * np.exp(-1.5 * angular_speed)

        # Reward stable behavior near the ground (low tilt + low speeds).
        if altitude < 0.30:
            reward += 1.2 * np.exp(-4.0 * tilt_norm)
            reward += 1.0 * np.exp(-3.0 * linear_speed)
            reward += 0.8 * np.exp(-2.0 * angular_speed)

        # Encourage downward movement while still above touchdown zone.
        if altitude > self.SUCCESS_ALTITUDE_M:
            descent_bonus = np.clip(-state[12], 0.0, 0.40)
            reward += 0.8 * descent_bonus

        # Strong success bonus when all touchdown conditions are met.
        if self._is_successful_landing(state):
            reward += 1000.0

        # Strongly penalize crash/out-of-control states so the agent learns to avoid them.
        if self._is_crash_condition(state):
            reward -= 500.0

        return float(reward)

    def _computeTerminated(self):
        """Episode terminates only when a stable landing state is reached."""
        state = self._getDroneStateVector(0)
        return self._is_successful_landing(state)

    def _computeTruncated(self):
        """Truncates on unsafe flight or time limit."""
        state = self._getDroneStateVector(0)

        out_of_bounds = abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.2
        timeout = self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC

        return out_of_bounds or self._is_crash_condition(state) or timeout

    def _computeInfo(self):
        """Computes the current info dict (kept lightweight for SB3 compatibility)."""
        return {"task": "landing"}

    def _is_successful_landing(self, state: np.ndarray) -> bool:
        """Checks if the drone satisfies stable touchdown criteria."""
        xy_error = np.linalg.norm(state[0:2] - self.TARGET_XY)
        altitude = state[2]
        linear_speed = np.linalg.norm(state[10:13])
        angular_speed = np.linalg.norm(state[13:16])
        tilt_norm = np.linalg.norm(state[7:9])

        return (
            xy_error < self.SUCCESS_XY_ERR_M
            and altitude < self.SUCCESS_ALTITUDE_M
            and linear_speed < self.SUCCESS_SPEED_MPS
            and angular_speed < self.SUCCESS_ANG_SPEED_RADPS
            and tilt_norm < self.SUCCESS_TILT_RAD
        )

    def _is_crash_condition(self, state: np.ndarray) -> bool:
        """Checks if the drone is in a crash or strongly out-of-control state."""
        over_tilted = abs(state[7]) > 1.0 or abs(state[8]) > 1.0
        too_fast = np.linalg.norm(state[10:13]) > 3.0 or np.linalg.norm(state[13:16]) > 8.0
        below_floor = state[2] < -0.05
        return over_tilted or too_fast or below_floor
