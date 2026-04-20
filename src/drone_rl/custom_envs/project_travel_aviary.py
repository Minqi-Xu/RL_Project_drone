"""Local project travel aviary task environment.

This travel environment is intentionally separate from hover/landing code so
travel-stage tuning does not affect existing training stages.
"""

from __future__ import annotations

import numpy as np
import pybullet as p
from gymnasium import spaces

from drone_rl.custom_envs.project_rl_base_aviary import ProjectBaseRLAviary
from gym_pybullet_drones.utils.enums import ActionType, DroneModel, ObservationType, Physics


class ProjectTravelAviary(ProjectBaseRLAviary):
    """Single-agent RL problem: travel from near (0,0,1) to (4,4,1) with VEL PID."""

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
        """Initialization of a single-agent RL environment for traveling."""
        self.START_POS = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.TARGET_POS = np.array([4.0, 4.0, 1.0], dtype=np.float32)
        self.EPISODE_LEN_SEC = 15

        # Randomized starts around (0,0,1), aligned with landing-stage ranges.
        self.INIT_XY_RANGE_M = 0.12
        self.INIT_Z_OFFSET_RANGE_M = (-0.05, 0.05)
        self.INIT_RP_RANGE_RAD = 0.25
        self.INIT_YAW_RANGE_RAD = np.pi
        self.INIT_VEL_XY_RANGE_MPS = 0.35
        self.INIT_VEL_Z_RANGE_MPS = 0.25
        self.INIT_ANG_VEL_RANGE_RADPS = 1.20

        # Travel velocity scaling over the default VEL speed limit.
        self.TRAVEL_SPEED_SCALE = 4.0

        # Reward shaping weights.
        self.W_POS = 5.0
        self.W_ATT = 0.3
        self.W_ALT = 0.5
        self.W_ANG_VEL = 0.05
        self.W_ACTION_SMOOTH = 0.08
        self.W_OVERSPEED = 1.0
        self.SAFE_SPEED_MPS = 1.0
        self.TIME_PENALTY = 0.02

        # Terminal rewards/penalties.
        self.SUCCESS_REWARD = 2000.0
        self.FAILURE_PENALTY = -2000.0
        self.previous_shaping = 0.0
        self.previous_travel_state = np.zeros(9, dtype=np.float32)

        # Success / failure thresholds.
        self.SUCCESS_DIST_M = 0.20
        self.SUCCESS_SPEED_MPS = 0.30
        self.SUCCESS_ATT_RAD = 0.30
        self.FAIL_ATT_RAD = 1.00
        self.BOUND_MIN_X_M = -2.0
        self.BOUND_MAX_X_M = 6.0
        self.BOUND_MIN_Y_M = -2.0
        self.BOUND_MAX_Y_M = 6.0
        self.BOUND_MIN_Z_M = 0.0
        self.BOUND_MAX_Z_M = 3.0

        # Travel-specific GUI camera framing so the entire path is visible.
        self.CAMERA_TARGET = np.array([2.0, 2.0, 1.0], dtype=np.float32)
        self.CAMERA_DISTANCE = 5.5
        self.CAMERA_YAW_DEG = 45.0
        self.CAMERA_PITCH_DEG = -32.0

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

        self._set_travel_camera_view()
        self._draw_reference_line()

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Resets environment and randomizes initial state near (0,0,1)."""
        _obs, info = super().reset(seed=seed, options=options)

        rng = self.np_random
        init_pos = np.array(
            [
                self.START_POS[0] + rng.uniform(-self.INIT_XY_RANGE_M, self.INIT_XY_RANGE_M),
                self.START_POS[1] + rng.uniform(-self.INIT_XY_RANGE_M, self.INIT_XY_RANGE_M),
                self.START_POS[2] + rng.uniform(self.INIT_Z_OFFSET_RANGE_M[0], self.INIT_Z_OFFSET_RANGE_M[1]),
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
        travel_state = self._get_travel_state()
        self.previous_travel_state = travel_state.copy()
        self.previous_shaping = self._compute_shaping(travel_state)
        self._set_travel_camera_view()
        self._draw_reference_line()
        return self._computeObs(), info

    def _observationSpace(self):
        """Returns observation space for travel state."""
        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([[lo, lo, lo, lo, lo, lo, lo, lo, lo]])
        obs_upper_bound = np.array([[hi, hi, hi, hi, hi, hi, hi, hi, hi]])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def _computeObs(self):
        """Returns current travel observation.

        State:
        [dx, dy, dz, p, q, r, roll, pitch, yaw]
        """
        return np.array([self._get_travel_state()], dtype=np.float32)

    def _actionSpace(self):
        """Returns action space for traveling.

        In travel VEL mode, the policy commands normalized [vx, vy, vz] in [-1, 1].
        """
        if self.ACT_TYPE == ActionType.VEL:
            size = 3
            act_lower_bound = np.array([-1 * np.ones(size) for _ in range(self.NUM_DRONES)])
            act_upper_bound = np.array([+1 * np.ones(size) for _ in range(self.NUM_DRONES)])
            for _ in range(self.ACTION_BUFFER_SIZE):
                self.action_buffer.append(np.zeros((self.NUM_DRONES, size)))
            return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
        return super()._actionSpace()

    def _preprocessAction(self, action):
        """Pre-processes action into motor RPMs using velocity PID control."""
        if self.ACT_TYPE != ActionType.VEL:
            return super()._preprocessAction(action)

        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES, 4))
        for k in range(action.shape[0]):
            state = self._getDroneStateVector(k)
            vel_cmd = np.clip(action[k, :], -1.0, 1.0)
            target_vel = self.TRAVEL_SPEED_SCALE * self.SPEED_LIMIT * vel_cmd
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

    def _computeReward(self):
        """Computes r_t = shaping_t - shaping_(t-1) - time_penalty + terminal reward."""
        travel_state = self._get_travel_state()
        shaping = self._compute_shaping(travel_state)
        reward = shaping - self.previous_shaping - self.TIME_PENALTY
        self.previous_shaping = shaping
        self.previous_travel_state = travel_state.copy()

        if self._is_success(travel_state):
            reward += self.SUCCESS_REWARD
        elif self._is_failure(travel_state):
            reward += self.FAILURE_PENALTY
        return float(reward)

    def _computeTerminated(self):
        """Computes episode termination."""
        travel_state = self._get_travel_state()
        return bool(self._is_success(travel_state) or self._is_failure(travel_state))

    def _computeTruncated(self):
        """No separate truncation."""
        return False

    def _computeInfo(self):
        """Computes current info dictionary."""
        travel_state = self._get_travel_state()
        delta_pos = travel_state[0:3]
        return {
            "task": "travel",
            "success": self._is_success(travel_state),
            "failure": self._is_failure(travel_state),
            "distance_to_target_m": float(np.linalg.norm(delta_pos)),
        }

    def _get_travel_state(self) -> np.ndarray:
        """Returns [dx, dy, dz, p, q, r, roll, pitch, yaw]."""
        state = self._getDroneStateVector(0)
        delta_pos = state[0:3] - self.TARGET_POS
        ang_vel = state[13:16]
        roll = state[7]
        pitch = state[8]
        yaw = state[9]
        return np.array(
            [
                delta_pos[0],
                delta_pos[1],
                delta_pos[2],
                ang_vel[0],
                ang_vel[1],
                ang_vel[2],
                roll,
                pitch,
                yaw,
            ],
            dtype=np.float32,
        )

    def _compute_shaping(self, travel_state: np.ndarray) -> float:
        """Computes shaping_t for traveling."""
        dx, dy, dz, p_rate, q_rate, r_rate, roll, pitch, _yaw = travel_state
        state = self._getDroneStateVector(0)
        speed = float(np.linalg.norm(state[10:13]))
        overspeed = max(0.0, speed - self.SAFE_SPEED_MPS)

        latest_action = self.action_buffer[-1][0]
        prev_action = self.action_buffer[-2][0] if len(self.action_buffer) > 1 else np.zeros_like(latest_action)
        action_delta_l2 = float(np.linalg.norm(latest_action - prev_action))

        shaping = (
            -self.W_POS * np.sqrt(dx**2 + dy**2 + dz**2)
            -self.W_ALT * np.abs(dz)
            -self.W_ATT * (np.abs(roll) + np.abs(pitch))
            -self.W_ANG_VEL * np.sqrt(p_rate**2 + q_rate**2 + r_rate**2)
            -self.W_OVERSPEED * (overspeed**2)
            -self.W_ACTION_SMOOTH * action_delta_l2
        )
        return float(shaping)

    def _is_success(self, travel_state: np.ndarray) -> bool:
        """Checks successful travel condition near target with stable motion."""
        dx, dy, dz, _p_rate, _q_rate, _r_rate, roll, pitch, _yaw = travel_state
        state = self._getDroneStateVector(0)
        speed = float(np.linalg.norm(state[10:13]))
        dist = float(np.sqrt(dx**2 + dy**2 + dz**2))
        return bool(
            dist < self.SUCCESS_DIST_M
            and speed < self.SUCCESS_SPEED_MPS
            and np.abs(roll) < self.SUCCESS_ATT_RAD
            and np.abs(pitch) < self.SUCCESS_ATT_RAD
        )

    def _is_failure(self, travel_state: np.ndarray) -> bool:
        """Checks failure conditions."""
        x, y, z = self._getDroneStateVector(0)[0:3]
        roll = float(travel_state[6])
        pitch = float(travel_state[7])
        out_of_bounds = (
            x < self.BOUND_MIN_X_M
            or x > self.BOUND_MAX_X_M
            or y < self.BOUND_MIN_Y_M
            or y > self.BOUND_MAX_Y_M
            or z < self.BOUND_MIN_Z_M
            or z > self.BOUND_MAX_Z_M
        )
        flip_too_much = np.abs(roll) > self.FAIL_ATT_RAD or np.abs(pitch) > self.FAIL_ATT_RAD
        timeout = self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC
        return bool(out_of_bounds or flip_too_much or timeout)

    def _draw_reference_line(self) -> None:
        """Draws GUI reference line from start to target for visualization."""
        if not self.GUI:
            return
        p.addUserDebugLine(
            lineFromXYZ=self.START_POS.tolist(),
            lineToXYZ=self.TARGET_POS.tolist(),
            lineColorRGB=[0.1, 0.9, 0.1],
            lineWidth=3.0,
            lifeTime=0,
            physicsClientId=self.CLIENT,
        )

    def _set_travel_camera_view(self) -> None:
        """Sets a travel-only GUI camera view covering the full route."""
        if not self.GUI:
            return
        p.resetDebugVisualizerCamera(
            cameraDistance=self.CAMERA_DISTANCE,
            cameraYaw=self.CAMERA_YAW_DEG,
            cameraPitch=self.CAMERA_PITCH_DEG,
            cameraTargetPosition=self.CAMERA_TARGET.tolist(),
            physicsClientId=self.CLIENT,
        )
