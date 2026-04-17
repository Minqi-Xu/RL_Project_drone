"""Local project landing aviary task environment.

This landing environment is intentionally kept separate from hover code so
landing-stage tuning does not affect hover-stage training.
"""

from __future__ import annotations

import numpy as np
import pybullet as p
from gymnasium import spaces

from drone_rl.custom_envs.project_rl_base_aviary import ProjectBaseRLAviary
from gym_pybullet_drones.utils.enums import ActionType, DroneModel, ObservationType, Physics


class ProjectLandingAviary(ProjectBaseRLAviary):
    """Single-agent RL problem: descend and land near target using PID velocity actions."""

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
        act: ActionType = ActionType.VEL,
    ):
        """Initialization of a single-agent RL environment for landing."""
        # Landing target at origin in world frame.
        self.TARGET_POS = np.array([0.0, 0.0, 0.0])

        # Episode and landing geometry.
        self.EPISODE_LEN_SEC = 8
        self.TOUCHDOWN_ALTITUDE_M = 0.05
        self.Z_DESCENT_SPEED_HIGH_MPS = 0.50
        self.Z_DESCENT_SPEED_LOW_MPS = 0.20
        self.Z_DESCENT_ALT_HIGH_M = 0.50
        self.Z_DESCENT_ALT_LOW_M = 0.20

        # Randomized starts around hover point near z=1.
        self.HOVER_START_POS = np.array([0.0, 0.0, 1.0])
        self.INIT_XY_RANGE_M = 0.12
        self.INIT_Z_OFFSET_RANGE_M = (-0.05, 0.05)
        self.INIT_RP_RANGE_RAD = 0.25
        self.INIT_YAW_RANGE_RAD = np.pi
        self.INIT_VEL_XY_RANGE_MPS = 0.35
        self.INIT_VEL_Z_RANGE_MPS = 0.25
        self.INIT_ANG_VEL_RANGE_RADPS = 1.20

        # Reward shaping weights:
        # shaping_t = -wp*||dpos|| - wv*||vel|| - wa*(|roll|+|pitch|)
        #             - ww*||ang_vel|| - ws*||action_t-action_(t-1)|| - wacc*||acc||.
        # Following the paper's "smooth actions with respect to time" idea, we
        # penalize action variation instead of absolute action magnitude.
        self.W_POS = 2.0
        self.W_VEL = 0.8
        self.W_ATT = 0.5
        self.W_ANG_VEL = 0.05
        self.W_ACTION_SMOOTH = 0.10
        self.W_ACC = 0.02

        # Terminal rewards/penalties.
        self.SUCCESS_REWARD = 50.0
        self.FAILURE_PENALTY = -50.0
        self.previous_shaping = 0.0
        self.previous_landing_state = np.zeros(11, dtype=np.float32)

        # Success / failure thresholds.
        self.SUCCESS_XY_ERR_M = 0.05
        self.SUCCESS_SPEED_MPS = 0.15
        self.SUCCESS_ATT_RAD = 0.25
        self.FAIL_ATT_RAD = 1.0
        self.HARD_LANDING_SPEED_MPS = 0.45
        self.BOUND_X_M = 5.0
        self.BOUND_Y_M = 5.0
        self.BOUND_Z_M = 2.5

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

        # Keep plane collision disabled so touchdown is judged by altitude threshold.
        p.setCollisionFilterPair(
            bodyUniqueIdA=self.PLANE_ID,
            bodyUniqueIdB=self.DRONE_IDS[0],
            linkIndexA=-1,
            linkIndexB=-1,
            enableCollision=0,
            physicsClientId=self.CLIENT,
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Resets environment and randomizes initial state around (0,0,1)."""
        _obs, info = super().reset(seed=seed, options=options)

        # Keep plane collision disabled after each reset.
        p.setCollisionFilterPair(
            bodyUniqueIdA=self.PLANE_ID,
            bodyUniqueIdB=self.DRONE_IDS[0],
            linkIndexA=-1,
            linkIndexB=-1,
            enableCollision=0,
            physicsClientId=self.CLIENT,
        )

        rng = self.np_random

        init_pos = np.array(
            [
                self.HOVER_START_POS[0] + rng.uniform(-self.INIT_XY_RANGE_M, self.INIT_XY_RANGE_M),
                self.HOVER_START_POS[1] + rng.uniform(-self.INIT_XY_RANGE_M, self.INIT_XY_RANGE_M),
                self.HOVER_START_POS[2] + rng.uniform(self.INIT_Z_OFFSET_RANGE_M[0], self.INIT_Z_OFFSET_RANGE_M[1]),
            ]
        )
        init_rpy = np.array(
            [
                rng.uniform(-self.INIT_RP_RANGE_RAD, self.INIT_RP_RANGE_RAD),
                rng.uniform(-self.INIT_RP_RANGE_RAD, self.INIT_RP_RANGE_RAD),
                rng.uniform(-self.INIT_YAW_RANGE_RAD, self.INIT_YAW_RANGE_RAD),
            ]
        )
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

        self._updateAndStoreKinematicInformation()
        landing_state = self._get_landing_state()
        self.previous_landing_state = landing_state.copy()
        self.previous_shaping = self._compute_shaping(landing_state)

        return self._computeObs(), info

    def _observationSpace(self):
        """Returns observation space for landing state."""
        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([[lo, lo, lo, lo, lo, lo, lo, lo, lo, lo, lo]])
        obs_upper_bound = np.array([[hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi]])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def _computeObs(self):
        """Returns current landing observation.

        State:
        [dx, dy, dz, vx, vy, vz, roll, pitch, p, q, r]
        """
        return np.array([self._get_landing_state()], dtype=np.float32)

    def _actionSpace(self):
        """Returns action space for landing.

        In landing VEL mode, the policy commands only lateral velocity references
        `[vx_ref, vy_ref]` in normalized range [-1, 1]. A constant downward z
        reference is injected in `_preprocessAction()`.
        """
        if self.ACT_TYPE == ActionType.VEL:
            size = 2
            act_lower_bound = np.array([-1 * np.ones(size) for _ in range(self.NUM_DRONES)])
            act_upper_bound = np.array([+1 * np.ones(size) for _ in range(self.NUM_DRONES)])
            for _ in range(self.ACTION_BUFFER_SIZE):
                self.action_buffer.append(np.zeros((self.NUM_DRONES, size)))
            return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
        return super()._actionSpace()

    def _preprocessAction(self, action):
        """Pre-processes action into motor RPMs.

        In landing VEL mode, action `[ax, ay]` commands lateral velocity while a
        scheduled negative z velocity reference drives descent.
        """
        if self.ACT_TYPE != ActionType.VEL:
            return super()._preprocessAction(action)

        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES, 4))
        for k in range(action.shape[0]):
            state = self._getDroneStateVector(k)
            planar_cmd = np.clip(action[k, :], -1.0, 1.0)
            planar_norm = np.linalg.norm(planar_cmd)
            if planar_norm > 1.0:
                planar_cmd = planar_cmd / planar_norm
                planar_norm = 1.0
            planar_unit = planar_cmd / planar_norm if planar_norm > 0 else np.zeros(2)
            altitude = float(state[2] - self.TARGET_POS[2])
            z_descent_speed = self._get_scheduled_descent_speed(altitude)
            target_vel = np.array(
                [
                    self.SPEED_LIMIT * planar_norm * planar_unit[0],
                    self.SPEED_LIMIT * planar_norm * planar_unit[1],
                    -z_descent_speed,
                ]
            )
            rpm_k, _, _ = self.ctrl[k].computeControl(
                control_timestep=self.CTRL_TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=state[0:3],
                target_rpy=np.array([0, 0, state[9]]),
                target_vel=target_vel,
            )
            rpm[k, :] = rpm_k
        return rpm

    def _get_scheduled_descent_speed(self, altitude_m: float) -> float:
        """Returns descent speed schedule as a function of altitude.

        - altitude > 0.5 m: constant 0.5 m/s
        - 0.2 m <= altitude <= 0.5 m: linearly decreases from 0.5 to 0.2 m/s
        - altitude < 0.2 m: constant 0.2 m/s
        """
        if altitude_m > self.Z_DESCENT_ALT_HIGH_M:
            return self.Z_DESCENT_SPEED_HIGH_MPS
        if altitude_m < self.Z_DESCENT_ALT_LOW_M:
            return self.Z_DESCENT_SPEED_LOW_MPS
        span = self.Z_DESCENT_ALT_HIGH_M - self.Z_DESCENT_ALT_LOW_M
        alpha = (altitude_m - self.Z_DESCENT_ALT_LOW_M) / span
        return self.Z_DESCENT_SPEED_LOW_MPS + alpha * (
            self.Z_DESCENT_SPEED_HIGH_MPS - self.Z_DESCENT_SPEED_LOW_MPS
        )

    def _computeReward(self):
        """Computes r_t = shaping_t - shaping_(t-1) + r_success + r_fail."""
        landing_state = self._get_landing_state()
        shaping = self._compute_shaping(landing_state)
        reward = shaping - self.previous_shaping
        self.previous_shaping = shaping
        self.previous_landing_state = landing_state.copy()

        if self._is_success(landing_state):
            reward += self.SUCCESS_REWARD
        elif self._is_failure(landing_state):
            reward += self.FAILURE_PENALTY
        # Landed but not success: no terminal bonus/penalty.
        return float(reward)

    def _computeTerminated(self):
        """Computes episode termination."""
        landing_state = self._get_landing_state()
        return bool(
            self._is_success(landing_state)
            or self._is_landed(landing_state)
            or self._is_failure(landing_state)
        )

    def _computeTruncated(self):
        """No separate truncation."""
        return False

    def _computeInfo(self):
        """Computes current info dictionary."""
        landing_state = self._get_landing_state()
        return {
            "task": "landing",
            "success": self._is_success(landing_state),
            "landed": self._is_landed(landing_state),
            "failure": self._is_failure(landing_state),
        }

    def _get_landing_state(self) -> np.ndarray:
        """Returns [dx, dy, dz, vx, vy, vz, roll, pitch, p, q, r]."""
        state = self._getDroneStateVector(0)
        delta_pos = state[0:3] - self.TARGET_POS
        vel = state[10:13]
        roll = state[7]
        pitch = state[8]
        ang_vel = state[13:16]
        return np.array(
            [
                delta_pos[0],
                delta_pos[1],
                delta_pos[2],
                vel[0],
                vel[1],
                vel[2],
                roll,
                pitch,
                ang_vel[0],
                ang_vel[1],
                ang_vel[2],
            ],
            dtype=np.float32,
        )

    def _compute_shaping(self, landing_state: np.ndarray) -> float:
        """Computes paper-style shaping_t for landing."""
        dx, dy, dz, vx, vy, vz, roll, pitch, p_rate, q_rate, r_rate = landing_state

        # Penalize aggressive command changes over time (paper-inspired smooth-action term).
        latest_action = self.action_buffer[-1][0]
        prev_action = self.action_buffer[-2][0] if len(self.action_buffer) > 1 else np.zeros_like(latest_action)
        action_delta_l2 = float(np.linalg.norm(latest_action - prev_action))

        # Penalize aggressive physical motion using an acceleration proxy.
        vel = np.array([vx, vy, vz], dtype=np.float32)
        prev_vel = self.previous_landing_state[3:6]
        acc_l2 = float(np.linalg.norm((vel - prev_vel) / self.CTRL_TIMESTEP))

        shaping = (
            -self.W_POS * np.sqrt(dx**2 + dy**2 + dz**2)
            -self.W_VEL * np.sqrt(vx**2 + vy**2 + vz**2)
            -self.W_ATT * (np.abs(roll) + np.abs(pitch))
            -self.W_ANG_VEL * np.sqrt(p_rate**2 + q_rate**2 + r_rate**2)
            -self.W_ACTION_SMOOTH * action_delta_l2
            -self.W_ACC * acc_l2
        )
        return float(shaping)

    def _is_success(self, landing_state: np.ndarray) -> bool:
        """Checks success landing condition."""
        dx, dy, _dz, vx, vy, vz, roll, pitch, _p_rate, _q_rate, _r_rate = landing_state
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        return bool(
            self._is_landed(landing_state)
            and np.abs(dx) < self.SUCCESS_XY_ERR_M
            and np.abs(dy) < self.SUCCESS_XY_ERR_M
            and speed < self.SUCCESS_SPEED_MPS
            and np.abs(roll) < self.SUCCESS_ATT_RAD
            and np.abs(pitch) < self.SUCCESS_ATT_RAD
        )

    def _is_landed(self, landing_state: np.ndarray) -> bool:
        """Checks touchdown from altitude threshold."""
        dz = landing_state[2]
        return bool(dz <= self.TOUCHDOWN_ALTITUDE_M)

    def _is_failure(self, landing_state: np.ndarray) -> bool:
        """Checks failure conditions."""
        dx, dy, dz, vx, vy, vz, roll, pitch, _p_rate, _q_rate, _r_rate = landing_state
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        out_of_bounds = (
            np.abs(dx) > self.BOUND_X_M
            or np.abs(dy) > self.BOUND_Y_M
            or np.abs(dz) > self.BOUND_Z_M
        )
        hard_landing = self._is_landed(landing_state) and speed > self.HARD_LANDING_SPEED_MPS
        flip_too_much = np.abs(roll) > self.FAIL_ATT_RAD or np.abs(pitch) > self.FAIL_ATT_RAD
        timeout = self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC
        return bool(out_of_bounds or hard_landing or flip_too_much or timeout)
