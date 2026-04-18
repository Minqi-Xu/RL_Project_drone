"""Project-local hover aviary variant with explicit 3D VEL actions.

This environment keeps hover reward/termination/truncation identical to
`ProjectHoverAviary`, but changes VEL action semantics so the policy directly
commands `[vx, vy, vz]` (all learned), without any scheduled z-velocity.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from drone_rl.custom_envs.project_hover_aviary import ProjectHoverAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


class ProjectHoverVelAviary(ProjectHoverAviary):
    """Hover task with independent velocity commands on x, y, and z axes."""

    ALPHA = 1.0
    BETA = 1.0
    BONUS = 0.25
    LAMBDA_VEL = 0.1
    W_SMOOTH = 0.1
    CRASH_PENALTY = 600.0

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset env and reinitialize action history buffer every episode."""
        _obs, info = super().reset(seed=seed, options=options)
        self._reset_action_buffer()
        self.prev_error = self._get_position_error()
        return self._computeObs(), info

    def _computeReward(self):
        """Computes hover reward for VEL mode with error-delta shaping."""
        state = self._getDroneStateVector(0)
        error = self._get_position_error()
        prev_error = self.prev_error if hasattr(self, "prev_error") else error
        vel_norm = float(np.linalg.norm(state[10:13]))

        reward = (
            self.ALPHA * (prev_error - error)
            - self.BETA * (error**2)
            + self.BONUS * float(error < 0.1)
            - self.LAMBDA_VEL * (vel_norm**2) * float(error < 0.2)
            - self.W_SMOOTH * self._action_delta_l1()
        )
        self.prev_error = error

        # Add explicit terminal penalties for VEL-hover failure modes.
        if self._is_crash_state(state):
            reward -= self.CRASH_PENALTY

        return float(reward)

    def _computeTerminated(self):
        """Terminate only on crash-like states for VEL hovering."""
        state = self._getDroneStateVector(0)
        return self._is_crash_state(state)

    def _computeTruncated(self):
        """Truncate only at hard episode time limit."""
        return self._is_timeout_state()

    def _is_crash_state(self, state: np.ndarray) -> bool:
        """Crash-like states aligned with hover RPM safety limits."""
        return bool(
            abs(state[0]) > 1.5
            or abs(state[1]) > 1.5
            or state[2] > 2.0
            or abs(state[7]) > 0.4
            or abs(state[8]) > 0.4
        )

    def _is_timeout_state(self) -> bool:
        """Hard endpoint at episode length."""
        return bool(self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC)

    def _get_position_error(self) -> float:
        """Returns Euclidean distance to hover target."""
        state = self._getDroneStateVector(0)
        return float(np.linalg.norm(self.TARGET_POS - state[0:3]))

    def _reset_action_buffer(self) -> None:
        """Clears and refills action history with zeros for deterministic resets."""
        if self.ACT_TYPE == ActionType.VEL:
            action_dim = 3
        else:
            action_dim = int(self.action_space.shape[1])
        self.action_buffer.clear()
        for _ in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES, action_dim), dtype=np.float32))

    def _action_delta_l1(self) -> float:
        """Returns L1 action change ||a_t - a_{t-1}||_1 for smoothness penalty."""
        if len(self.action_buffer) < 2:
            return 0.0
        a_t = self.action_buffer[-1][0]
        a_prev = self.action_buffer[-2][0]
        return float(np.sum(np.abs(a_t - a_prev)))

    def _actionSpace(self):
        """Returns action space.

        For VEL mode, policy action is normalized `[vx, vy, vz]` in `[-1, 1]`.
        """
        if self.ACT_TYPE != ActionType.VEL:
            return super()._actionSpace()

        size = 3
        act_lower_bound = np.array([-1 * np.ones(size) for _ in range(self.NUM_DRONES)])
        act_upper_bound = np.array([+1 * np.ones(size) for _ in range(self.NUM_DRONES)])
        for _ in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES, size)))
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    def _observationSpace(self):
        """Returns observation space.

        For VEL mode, state uses delta position [dx, dy, dz] and appends
        3 action-history values per buffer slot instead of 4.
        """
        if self.OBS_TYPE != ObservationType.KIN or self.ACT_TYPE != ActionType.VEL:
            return super()._observationSpace()

        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([[lo, lo, lo, lo, lo, lo, lo, lo, lo, lo, lo, lo] for _ in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi] for _ in range(self.NUM_DRONES)])

        act_lo = -1
        act_hi = +1
        for _ in range(self.ACTION_BUFFER_SIZE):
            obs_lower_bound = np.hstack(
                [obs_lower_bound, np.array([[act_lo, act_lo, act_lo] for _ in range(self.NUM_DRONES)])]
            )
            obs_upper_bound = np.hstack(
                [obs_upper_bound, np.array([[act_hi, act_hi, act_hi] for _ in range(self.NUM_DRONES)])]
            )
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def _computeObs(self):
        """Returns current observation.

        KIN state layout:
        [dx, dy, dz, roll, pitch, yaw, vx, vy, vz, wx, wy, wz] + action history.
        """
        if self.OBS_TYPE != ObservationType.KIN or self.ACT_TYPE != ActionType.VEL:
            return super()._computeObs()

        obs_12 = np.zeros((self.NUM_DRONES, 12))
        for i in range(self.NUM_DRONES):
            obs = self._getDroneStateVector(i)
            delta_pos = obs[0:3] - self.TARGET_POS
            obs_12[i, :] = np.hstack([delta_pos, obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)

        ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype("float32")
        for i in range(self.ACTION_BUFFER_SIZE):
            ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
        return ret

    def _preprocessAction(self, action):
        """Converts normalized `[vx, vy, vz]` into PID velocity targets and RPMs."""
        if self.ACT_TYPE != ActionType.VEL:
            return super()._preprocessAction(action)

        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES, 4))

        for k in range(action.shape[0]):
            state = self._getDroneStateVector(k)
            vel_cmd = np.clip(action[k, :], -1.0, 1.0)
            target_vel = self.SPEED_LIMIT * vel_cmd
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
