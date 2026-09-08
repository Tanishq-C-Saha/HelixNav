"""Command terms for HelixNav CP7.

Owns navigation tracking state only:
  waypoint index, lookaheads, snap flags, path_remaining.

Does NOT own scene state — env's events.py handles:
  obstacles, grid, A*, goal sampling, robot pose.
"""

from __future__ import annotations
from collections.abc import Sequence

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.configclass import configclass
import isaaclab.utils.math as math_utils

from .events import MAX_PATH_LENGTH

# ──────────────────────────────────────────────
#  Command Term
# ──────────────────────────────────────────────


class NavigationWaypointCommand(CommandTerm):
    """Pure state tracker for navigation along an A* path."""

    SAMPLE_DISTANCES = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    NUM_LOOKAHEADS = len(SAMPLE_DISTANCES)

    def __init__(self, cfg: NavigationWaypointCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        N = env.num_envs
        D = env.device

        self.robot: Articulation = env.scene[cfg.asset_name]

        # per-env waypoint tracking
        self.current_waypoint_idx = torch.zeros(N, dtype=torch.long, device=D)

        # path in TRUE WORLD frame (local + env_origins, set at reset)
        self.path_world = torch.zeros(N, MAX_PATH_LENGTH, 2, device=D)
        self.path_lengths = torch.zeros(N, dtype=torch.long, device=D)

        # cumulative distances along each path
        self.cum_dist = torch.zeros(N, MAX_PATH_LENGTH, device=D)

        # goal in world frame (copied from env, source of truth)
        self.goal_pos_w = torch.zeros(N, 2, device=D)

        # lookahead outputs — obs functions read these
        self.lookaheads = torch.zeros(N, self.NUM_LOOKAHEADS, 3, device=D)
        self.snap_flags = torch.zeros(N, self.NUM_LOOKAHEADS, device=D)

        # PBRS state — TWO potentials, kept separate
        self.path_remaining_euclidean = torch.zeros(N, device=D)
        self.prev_path_remaining_euclidean = torch.zeros(N, device=D)

        self.path_remaining_along = torch.zeros(N, device=D)
        self.prev_path_remaining_along = torch.zeros(N, device=D)

        # constants as tensor
        self._sample_dists = torch.tensor(
            self.SAMPLE_DISTANCES, device=D, dtype=torch.float
        )

    # ── required abstract property ──

    @property
    def command(self) -> torch.Tensor:
        """Flattened lookaheads as the 'command'. Shape: (N, 24)."""
        return self.lookaheads.reshape(self._env.num_envs, -1)

    # ── _resample_command (called by base _resample after timer reset) ──

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_t = (
            torch.tensor(env_ids, device=self._env.device, dtype=torch.long)
            if not isinstance(env_ids, torch.Tensor)
            else env_ids
        )

        self.current_waypoint_idx[env_ids_t] = 0

        # copy path from env (LOCAL → WORLD by adding env origin)
        origins_xy = self._env.scene.env_origins[env_ids_t, :2]
        self.path_world[env_ids_t] = self._env._paths_local[
            env_ids_t
        ] + origins_xy.unsqueeze(1)
        self.path_lengths[env_ids_t] = self._env._path_lengths[env_ids_t]

        # goal from env (source of truth - works even if A* failed)
        self.goal_pos_w[env_ids_t] = (
            self._env._goal_positions_local[env_ids_t, :2] + origins_xy
        )

        # handle A* failure: zero everything for envs with no path
        failed = self.path_lengths[env_ids_t] == 0
        if failed.any():
            f_ids = env_ids_t[failed]
            self.path_remaining_euclidean[f_ids] = 0.0
            self.prev_path_remaining_euclidean[f_ids] = 0.0
            self.path_remaining_along[f_ids] = 0.0
            self.prev_path_remaining_along[f_ids] = 0.0
            self.lookaheads[f_ids] = 0.0
            self.snap_flags[f_ids] = 0.0

        # compute only for envs with valid paths
        valid = ~failed
        if valid.any():
            v_ids = env_ids_t[valid]
            self._recompute_cum_dist(v_ids)
            self._compute_lookaheads(v_ids)
            self._compute_path_remaining(v_ids)
            self.prev_path_remaining_euclidean[v_ids] = self.path_remaining_euclidean[v_ids].clone()
            self.prev_path_remaining_along[v_ids] = self.path_remaining_along[v_ids].clone()

    # ── _update_command (called every step, NO dt param) ──

    def _update_command(self):
        N = self._env.num_envs
        D = self._env.device

        robot_pos_w = self.robot.data.root_pos_w[:, :2]  # (N, 2)

        # nearest-point-on-path projection (replaces threshold-based advance)
        self._update_nearest_waypoint(robot_pos_w)

        # recompute downstream (only valid paths)
        all_ids = torch.arange(N, device=D)
        valid = self.path_lengths > 0
        if valid.any():
            v_ids = all_ids[valid]
            self.prev_path_remaining_euclidean[v_ids] = self.path_remaining_euclidean[v_ids].clone()
            self.prev_path_remaining_along[v_ids] = self.path_remaining_along[v_ids].clone()
            self._compute_lookaheads(v_ids)
            self._compute_path_remaining(v_ids)

    def _update_nearest_waypoint(self, robot_pos_w: torch.Tensor):
        """Snap current_waypoint_idx to the path segment the robot is closest to.
        
        Uses point-to-line-segment distance. Enforces monotonic forward progress
        so the policy cannot farm PBRS by walking backward along the path.
        """
        path = self.path_world           # (N, MAX, 2)
        N, MAX, _ = path.shape

        # segments: start[i] = path[i], end[i] = path[i+1]
        seg_start = path[:, :-1, :]              # (N, MAX-1, 2)
        seg_end = path[:, 1:, :]                 # (N, MAX-1, 2)
        seg_vec = seg_end - seg_start            # (N, MAX-1, 2)
        seg_len_sq = (seg_vec ** 2).sum(dim=-1).clamp(min=1e-8)  # (N, MAX-1)

        # project robot onto each segment
        robot_expanded = robot_pos_w.unsqueeze(1)                     # (N, 1, 2)
        to_robot = robot_expanded - seg_start                          # (N, MAX-1, 2)
        t = (to_robot * seg_vec).sum(dim=-1) / seg_len_sq              # (N, MAX-1)
        t = t.clamp(0.0, 1.0)

        # closest point on each segment, distance from robot
        closest = seg_start + t.unsqueeze(-1) * seg_vec                # (N, MAX-1, 2)
        dist_to_seg = torch.norm(robot_expanded - closest, dim=-1)     # (N, MAX-1)

        # mask out invalid segments (beyond path length - 1)
        indices = torch.arange(MAX - 1, device=path.device).unsqueeze(0)  # (1, MAX-1)
        valid_seg = indices < (self.path_lengths - 1).unsqueeze(1)
        dist_to_seg = torch.where(valid_seg, dist_to_seg, torch.full_like(dist_to_seg, float('inf')))

        # monotonic forward progress: index cannot decrease
        monotonic = indices >= self.current_waypoint_idx.unsqueeze(1)
        dist_to_seg = torch.where(monotonic, dist_to_seg, torch.full_like(dist_to_seg, float('inf')))

        # argmin → new segment index (= new current_waypoint_idx)
        new_idx = torch.argmin(dist_to_seg, dim=1)
        self.current_waypoint_idx = new_idx.clamp(max=(self.path_lengths - 1).clamp(min=0))

    # ── cumulative distance precomputation ──

    def _recompute_cum_dist(self, env_ids: torch.Tensor):
        path = self.path_world[env_ids]
        MAX = path.shape[1]

        seg = path[:, 1:, :] - path[:, :-1, :]
        seg_len = torch.norm(seg, dim=-1)

        indices = torch.arange(MAX - 1, device=path.device).unsqueeze(0)
        valid_mask = indices < (self.path_lengths[env_ids] - 1).unsqueeze(1)
        seg_len = seg_len * valid_mask.float()

        cum = torch.cumsum(seg_len, dim=1)
        self.cum_dist[env_ids, 0] = 0.0
        self.cum_dist[env_ids, 1:] = cum

    # ── distance-based lookahead sampling ──

    def _compute_lookaheads(self, env_ids: torch.Tensor):
        D = self._env.device
        n = env_ids.shape[0]
        MAX = self.path_world.shape[1]
        K = self.NUM_LOOKAHEADS

        path = self.path_world[env_ids]
        cum = self.cum_dist[env_ids]
        wp_idx = self.current_waypoint_idx[env_ids]
        p_len = self.path_lengths[env_ids]

        robot_pos_w = self.robot.data.root_pos_w[env_ids, :2]

        # cumulative distance at current waypoint
        base_cum = cum.gather(1, wp_idx.unsqueeze(1)).squeeze(1)

        # target cumulative distances per sample slot
        target_cum = base_cum.unsqueeze(1) + self._sample_dists.unsqueeze(0)

        # total path distance
        last_idx = (p_len - 1).clamp(min=0)
        total_cum = cum.gather(1, last_idx.unsqueeze(1)).squeeze(1)

        # snap flags
        snap = target_cum >= total_cum.unsqueeze(1)
        target_cum_clamped = torch.clamp(target_cum, max=total_cum.unsqueeze(1))

        # find segment index for each target distance
        le_mask = cum.unsqueeze(2) <= target_cum_clamped.unsqueeze(1)  # (n, MAX, K)
        seg_idx = le_mask.sum(dim=1) - 1  # (n, K)
        seg_idx = seg_idx.clamp(min=0, max=MAX - 2)

        # interpolation fraction
        seg_start_cum = cum.gather(1, seg_idx)
        seg_end_cum = cum.gather(1, (seg_idx + 1).clamp(max=MAX - 1))
        seg_length = (seg_end_cum - seg_start_cum).clamp(min=1e-6)
        frac = ((target_cum_clamped - seg_start_cum) / seg_length).clamp(0.0, 1.0)

        # interpolate positions
        idx_2d = seg_idx.unsqueeze(-1).expand(-1, -1, 2)
        idx_2d_next = (seg_idx + 1).clamp(max=MAX - 1).unsqueeze(-1).expand(-1, -1, 2)
        wp_start = path.gather(1, idx_2d)
        wp_end = path.gather(1, idx_2d_next)
        sampled_w = wp_start + frac.unsqueeze(-1) * (wp_end - wp_start)

        # override snapped slots with goal
        goal_expanded = self.goal_pos_w[env_ids].unsqueeze(1).expand(-1, K, -1)
        sampled_w = torch.where(snap.unsqueeze(-1), goal_expanded, sampled_w)

        # convert sampled lookahead points from world frame to robot frame
        rel_w = sampled_w - robot_pos_w.unsqueeze(1)  # (N, K, 2)

        # Flatten to (N*K, 3) because quat_apply_inverse does not
        # broadcast (N, 4) against (N, K, 3).
        rel_w_3d = torch.zeros(
            n * K,
            3,
            device=D,
            dtype=rel_w.dtype,
        )
        rel_w_3d[:, :2] = rel_w.reshape(n * K, 2)

        # Expand each robot's yaw quaternion across its K lookaheads:
        # (N, 4) -> (N, K, 4) -> (N*K, 4)
        robot_yaw_quat = math_utils.yaw_quat(
            self.robot.data.root_quat_w[env_ids]
        )
        robot_yaw_quat = (
            robot_yaw_quat
            .unsqueeze(1)
            .expand(-1, K, -1)
            .reshape(n * K, 4)
        )

        # Rotate world-frame vectors into robot frame
        rel_local = math_utils.quat_apply_inverse(
            robot_yaw_quat,
            rel_w_3d,
        )

        # Back to (N, K, 2)
        rel_local = rel_local[:, :2].reshape(n, K, 2)

        # Euclidean distance and unit direction
        eucl = torch.norm(rel_local, dim=-1).clamp(min=1e-6)  # (N, K)

        unit_dx = rel_local[..., 0] / eucl
        unit_dy = rel_local[..., 1] / eucl

        # Distance normalization
        norm_dist = eucl / self._sample_dists.unsqueeze(0)

        self.lookaheads[env_ids] = torch.stack(
            [unit_dx, unit_dy, norm_dist],
            dim=-1,
        )

        self.snap_flags[env_ids] = snap.float()

    # ── path remaining for PBRS ──

    def _compute_path_remaining(self, env_ids: torch.Tensor):
        """Compute Euclidean and continuous nearest-path remaining distances."""

        robot_pos_w = self.robot.data.root_pos_w[env_ids, :2]
        goal_pos_w = self.goal_pos_w[env_ids]

        # ---------------------------------------------------------
        # Euclidean remaining distance
        # ---------------------------------------------------------
        self.path_remaining_euclidean[env_ids] = torch.norm(
            robot_pos_w - goal_pos_w,
            dim=-1,
        )

        # ---------------------------------------------------------
        # Continuous remaining distance along the A* path
        #
        # Find the nearest point on any valid path segment, then
        # measure how far along the path that point lies.
        # ---------------------------------------------------------
        path = self.path_world[env_ids]
        cum = self.cum_dist[env_ids]
        p_len = self.path_lengths[env_ids]

        n = env_ids.shape[0]
        MAX = path.shape[1]

        # Segments
        seg_start = path[:, :-1, :]       # (n, MAX-1, 2)
        seg_end = path[:, 1:, :]          # (n, MAX-1, 2)
        seg_vec = seg_end - seg_start

        seg_len_sq = (
            (seg_vec ** 2).sum(dim=-1)
            .clamp(min=1e-8)
        )

        # Robot relative to every segment start
        robot_expanded = robot_pos_w.unsqueeze(1)

        to_robot = robot_expanded - seg_start

        # Projection fraction on each segment
        t = (
            (to_robot * seg_vec).sum(dim=-1)
            / seg_len_sq
        )

        t = t.clamp(0.0, 1.0)

        # Nearest point on every segment
        closest = (
            seg_start
            + t.unsqueeze(-1) * seg_vec
        )

        # Robot → nearest point distance
        dist_to_seg = torch.norm(
            robot_expanded - closest,
            dim=-1,
        )

        # ---------------------------------------------------------
        # Ignore segments that don't exist
        # ---------------------------------------------------------
        indices = torch.arange(
            MAX - 1,
            device=path.device,
        ).unsqueeze(0)

        valid_seg = (
            indices < (p_len - 1).unsqueeze(1)
        )

        dist_to_seg = torch.where(
            valid_seg,
            dist_to_seg,
            torch.full_like(
                dist_to_seg,
                float("inf"),
            ),
        )

        # ---------------------------------------------------------
        # Find nearest path segment
        # ---------------------------------------------------------
        nearest_seg = torch.argmin(
            dist_to_seg,
            dim=1,
        )

        # Projection fraction on the selected segment
        nearest_t = t.gather(
            1,
            nearest_seg.unsqueeze(1),
        ).squeeze(1)

        # Length of selected segment
        nearest_seg_len = torch.sqrt(
            seg_len_sq.gather(
                1,
                nearest_seg.unsqueeze(1),
            ).squeeze(1)
        )

        # ---------------------------------------------------------
        # Distance traveled along the A* path to the nearest point
        # ---------------------------------------------------------
        cum_at_seg = cum.gather(
            1,
            nearest_seg.unsqueeze(1),
        ).squeeze(1)

        distance_along_path = (
            cum_at_seg
            + nearest_t * nearest_seg_len
        )

        # Total A* path length
        last_idx = (p_len - 1).clamp(min=0)

        total_path_length = cum.gather(
            1,
            last_idx.unsqueeze(1),
        ).squeeze(1)

        # ---------------------------------------------------------
        # Remaining path distance
        # ---------------------------------------------------------
        self.path_remaining_along[env_ids] = (
            total_path_length - distance_along_path
        ).clamp(min=0.0)

    # ── helpers ──

    def _extract_yaw(self, env_ids: torch.Tensor) -> torch.Tensor:
        quat = self.robot.data.root_quat_w[env_ids]  # (n, 4) wxyz
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _gather_waypoints(path: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        gi = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)
        return path.gather(1, gi).squeeze(1)

    # ── required overrides (no-ops for now) ──

    def _update_metrics(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


# ──────────────────────────────────────────────
#  Cfg (MUST be after class definition)
# ──────────────────────────────────────────────


@configclass
class NavigationWaypointCommandCfg(CommandTermCfg):
    """Configuration for NavigationWaypointCommand."""

    class_type: type = NavigationWaypointCommand
    asset_name: str = "robot"
    waypoint_advance_threshold: float = 0.3
