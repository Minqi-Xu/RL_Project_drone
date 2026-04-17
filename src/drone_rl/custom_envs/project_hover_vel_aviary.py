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

    def _computeReward(self):
        """Computes hover reward for VEL mode using cubic distance shaping."""
        state = self._getDroneStateVector(0)
        return max(0, 2 - np.linalg.norm(self.TARGET_POS - state[0:3]) ** 3)

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

        For VEL mode, append 3 action-history values per buffer slot instead of 4.
        """
        if self.OBS_TYPE != ObservationType.KIN or self.ACT_TYPE != ActionType.VEL:
            return super()._observationSpace()

        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([[lo, lo, 0, lo, lo, lo, lo, lo, lo, lo, lo, lo] for _ in range(self.NUM_DRONES)])
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
        """Returns current observation."""
        if self.OBS_TYPE != ObservationType.KIN or self.ACT_TYPE != ActionType.VEL:
            return super()._computeObs()

        obs_12 = np.zeros((self.NUM_DRONES, 12))
        for i in range(self.NUM_DRONES):
            obs = self._getDroneStateVector(i)
            obs_12[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)

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
