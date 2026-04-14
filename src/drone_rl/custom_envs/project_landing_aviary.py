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

        # Reward shaping coefficients adapted from paper index.pdf Eq. (10)-(11).
        self.SHAPING_POS_COEFF = 100.0
        self.SHAPING_VEL_COEFF = 10.0
        self.SHAPING_ACTION_COEFF = 1.0
        # Penalize angular-rate magnitude (roll/pitch/yaw rates) to improve landing stability.
        self.SHAPING_ANGVEL_COEFF = 5.0
        # Explicit altitude and descent shaping to avoid hovering local optimum.
        self.SHAPING_ALTITUDE_COEFF = 30.0
        self.SHAPING_DESCENT_COEFF = 12.0
        self.DESCENT_TARGET_MPS = -0.20
        self.CONTACT_BONUS_COEFF = 10.0
        # Small per-step time cost to encourage timely landing.
        self.STEP_TIME_COST = 0.02
        # Positive terminal bonus when the stable landing condition is met.
        self.TERMINAL_LANDING_BONUS = 300.0
        # Keep an explicit crash penalty for safety during exploration.
        self.CRASH_PENALTY = 10000.0
        # Penalize non-crash truncation reasons to discourage stalling or drifting away.
        self.OUT_OF_BOUNDS_PENALTY = 10000.0
        self.TIMEOUT_PENALTY = 10000.0
        # Larger XY workspace for landing training.
        self.XY_BOUND_M = 5.0
        self.Z_UPPER_BOUND_M = 2.2
        # Differential shaping requires the previous shaping value from the prior step.
        self.previous_shaping = 0.0

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
        # Initialize previous shaping at reset so the first reward step is consistent.
        state = self._getDroneStateVector(0)
        self.previous_shaping = self._computeShaping(state)
        return self._computeObs(), info

    def _computeReward(self):
        """Computes landing reward using differential shaping from the referenced paper."""
        state = self._getDroneStateVector(0)
        shaping = self._computeShaping(state)
        reward = shaping - self.previous_shaping
        self.previous_shaping = shaping

        # Encourage finishing the task quickly instead of hovering indefinitely.
        reward -= self.STEP_TIME_COST

        # Explicitly reward successful stable touchdown.
        if self._is_successful_landing(state):
            reward += self.TERMINAL_LANDING_BONUS

        if self._is_out_of_bounds(state):
            reward -= self.OUT_OF_BOUNDS_PENALTY

        if self._is_timeout():
            reward -= self.TIMEOUT_PENALTY

        if self._is_crash_condition(state):
            reward -= self.CRASH_PENALTY

        return float(reward)

    def _computeTerminated(self):
        """Episode terminates only when a stable landing state is reached."""
        state = self._getDroneStateVector(0)
        return self._is_successful_landing(state)

    def _computeTruncated(self):
        """Truncates on unsafe flight or time limit."""
        state = self._getDroneStateVector(0)

        out_of_bounds = self._is_out_of_bounds(state)
        timeout = self._is_timeout()

        return out_of_bounds or self._is_crash_condition(state) or timeout

    def _computeInfo(self):
        """Computes the current info dict (kept lightweight for SB3 compatibility)."""
        return {"task": "landing"}

    def _is_successful_landing(self, state: np.ndarray) -> bool:
        """Checks if the drone satisfies stable touchdown criteria."""
        altitude = state[2]
        linear_speed = np.linalg.norm(state[10:13])
        angular_speed = np.linalg.norm(state[13:16])
        tilt_norm = np.linalg.norm(state[7:9])

        return (
            altitude < self.SUCCESS_ALTITUDE_M
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

    def _is_out_of_bounds(self, state: np.ndarray) -> bool:
        """Checks whether the drone is outside the allowed training workspace."""
        return (
            abs(state[0]) > self.XY_BOUND_M
            or abs(state[1]) > self.XY_BOUND_M
            or state[2] > self.Z_UPPER_BOUND_M
        )

    def _is_timeout(self) -> bool:
        """Checks whether the current episode exceeded max duration."""
        return self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC

    def _computeShaping(self, state: np.ndarray) -> float:
        """Computes shaping term used in r_t = shaping_t - shaping_{t-1}."""
        # Relative planar position and velocity errors.
        px = state[0] - self.TARGET_XY[0]
        py = state[1] - self.TARGET_XY[1]
        z = state[2]
        vx = state[10]
        vy = state[11]
        vz = state[12]
        wx = state[13]
        wy = state[14]
        wz = state[15]

        # Use normalized action magnitudes from the latest policy command.
        # RPM control has 4 actions; we aggregate them into one smoothness magnitude.
        latest_action = self.action_buffer[-1][0]
        action_l2 = np.linalg.norm(latest_action)
        action_mean_abs = float(np.mean(np.abs(latest_action)))

        # Contact-like indicator adapted for this task:
        # near ground and low vertical speed approximates touchdown state.
        contact_like = 1.0 if (state[2] < 0.10 and abs(state[12]) < 0.20) else 0.0

        shaping = (
            -self.SHAPING_POS_COEFF * np.sqrt(px**2 + py**2)
            -self.SHAPING_ALTITUDE_COEFF * z
            -self.SHAPING_VEL_COEFF * np.sqrt(vx**2 + vy**2)
            -self.SHAPING_DESCENT_COEFF * abs(vz - self.DESCENT_TARGET_MPS)
            -self.SHAPING_ACTION_COEFF * action_l2
            -self.SHAPING_ANGVEL_COEFF * np.sqrt(wx**2 + wy**2 + wz**2)
            +self.CONTACT_BONUS_COEFF * contact_like * (1.0 - action_mean_abs)
        )
        return float(shaping)
