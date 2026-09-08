"""Command terms for the HelixNav."""

from dataclasses import MISSING
from typing import Sequence
from isaaclab.managers import CommandTerm
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from .events import MAX_PATH_LENGTH

import torch


class NavigationWaypointCommand(CommandTerm):

    # fixed sample distances along the A* path (metres)
    SAMPLE_DISTANCES = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    NUM_LOOKAHEADS = len(SAMPLE_DISTANCES)

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        N = env.num_envs
        D = env.device

        # obtain the robot from the scene
        self.robot: Articulation = env.scene[cfg.asset_name]

        # per-env waypoint tracking
        self.current_waypoint_idx = torch.zeros(N, dtype=torch.long, device=D)

        # path in world frame, copied from env at each reset
        #    shape: (N, MAX_PATH_LENGTH, 2)
        self.path_world = torch.zeros(N, MAX_PATH_LENGTH, 2, device=D)
        self.path_lengths = torch.zeros(N, dtype=torch.long, device=D)

        # lookahead output: what obs functions read
        #    each slot: (unit_dx, unit_dy, norm_dist)
        self.lookaheads = torch.zeros(N, self.NUM_LOOKAHEADS, 3, device=D)
        self.snap_flags = torch.zeros(N, self.NUM_LOOKAHEADS, dtype=torch.float, device=D)

        # path remaining for PBRS
        self.path_remaining = torch.zeros(N, device=D)
        self.prev_path_remaining = torch.zeros(N, device=D)

        # sample distances as tensor for vectorized ops
        self._sample_dists = torch.tensor(
            self.SAMPLE_DISTANCES, device=D, dtype=torch.float
        )

    def _resample(self, env_ids: Sequence[int]):
        """Called by command manager after _reset_idx. 
        Grid and A* path already computed by env."""

        self.current_waypoint_idx[env_ids] = 0

        # copy from env (env owns the path, we just track state on it)
        self.path_world[env_ids] = (
            self._env._paths_local[env_ids]    # [N, P, 2]
            + self._env.scene.env_origins[env_ids, :2].unsqueeze(1)   # [N, 1, 2]
        )
        self.path_lengths[env_ids] = self._env._path_lengths[env_ids]

        # compute initial lookaheads and path_remaining
        self._compute_lookaheads(env_ids)
        self._compute_path_remaining(env_ids)
        self.prev_path_remaining[env_ids] = self.path_remaining[env_ids]

    def _update_command(self, dt: float):
        """Called every step. Advance waypoint idx, recompute lookaheads."""

        robot_pos_w = self.robot.data.root_pos_w[:, :2]

        # advance waypoint idx if robot is close enough
        current_wp = self._get_current_waypoint()  # (N, 2)
        dist_to_wp = torch.norm(robot_pos_w - current_wp, dim=-1)

        advance_mask = dist_to_wp < self.cfg.waypoint_advance_threshold
        self.current_waypoint_idx[advance_mask] += 1

        # clamp to path length
        self.current_waypoint_idx = torch.clamp(
            self.current_waypoint_idx,
            max=self.path_lengths - 1
        )

        # recompute everything downstream
        all_ids = torch.arange(self._env.num_envs, device=self._env.device)
        self.prev_path_remaining[:] = self.path_remaining
        self._compute_lookaheads(all_ids)
        self._compute_path_remaining(all_ids)

    def _compute_lookaheads(self, env_ids):
        """Sample 8 points at fixed distances along path from current position.
        Convert to robot-relative frame."""
        # TODO: walk path accumulating segment lengths,
        #       sample at each SAMPLE_DISTANCE,
        #       snap to goal if path ends before sample distance,
        #       rotate into robot frame
        pass

    def _compute_path_remaining(self, env_ids):
        """Sum segment lengths from current waypoint idx to end of path."""
        # TODO: vectorized cumulative segment length from current_idx to path end
        pass

    def _get_current_waypoint(self) -> torch.Tensor:
        """Gather current waypoint for each env."""
        idx = self.current_waypoint_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)
        return self.path_world.gather(1, idx).squeeze(1)   # N,2