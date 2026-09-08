
"""
HelixNav PD navigation debugger.

Purpose
-------
Replace keyboard/RL control with a deterministic SE(2) PD controller.

The controller consumes the first NavigationWaypointCommand lookahead and
outputs normalized [vx, vy, yaw_rate] commands. Every command component is
hard-clamped to [-1, +1].

The script:
  1. Runs multiple Isaac Lab environments (default: 2).
  2. Uses only the PD controller to drive the robot.
  3. Logs robot trajectory, goal distance, path distance, waypoint index,
     lookahead normalization, command values, and both PBRS potentials.
  4. Independently computes nearest-point-on-A*-path geometry so the current
     waypoint-based along-potential can be compared against a geometric
     reference.
  5. Saves one trajectory plot and six diagnostic plots per environment.
  6. Prints automatic PASS/WARN/FAIL conclusions at the end.

Important
---------
The action sent to env.step() is the normalized navigation command:
    [vx, vy, yaw_rate] in [-1, +1].

The physical maximums are deliberately 1.0 in this debug configuration:
    max |vx|      = 1.0
    max |vy|      = 1.0
    max |yaw_rate|= 1.0

This is a command-path / navigation-state debugger, not an RL-performance
test. It is useful for proving that the navigation command term behaves
sensibly before introducing PPO.
"""

from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from isaaclab.app import AppLauncher


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="HelixNav PD navigation debug.")
parser.add_argument(
    "--num_envs",
    type=int,
    default=2,
    help="Number of parallel environments.",
)
parser.add_argument(
    "--duration",
    type=float,
    default=30.0,
    help="Maximum simulation time in seconds.",
)
parser.add_argument("--kp_xy", type=float, default=2.0)
parser.add_argument("--kd_xy", type=float, default=0.5)
parser.add_argument("--kp_yaw", type=float, default=2.0)
parser.add_argument("--kd_yaw", type=float, default=0.3)
parser.add_argument(
    "--print_every",
    type=int,
    default=25,
    help="Print compact status every N navigation steps.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# ---------------------------------------------------------------------------
# Imports requiring Isaac runtime
# ---------------------------------------------------------------------------

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import euler_xyz_from_quat

from helix_nav.tasks.manager_based.navigation.config.go2.env_configs.debug.helixnav_debug_base_rl_env_cfg import (
    HelixNavDebugBaseRLEnvCfg,
)


# ---------------------------------------------------------------------------
# Debug constants
# ---------------------------------------------------------------------------

# Normalized command limits.
MAX_VX = 1.0
MAX_VY = 1.0
MAX_WZ = 1.0

# First NavigationWaypointCommand lookahead is 0.5 m.
FIRST_LOOKAHEAD_DISTANCE = 0.5

# Diagnostic only. This does not replace the environment's goal criterion.
GOAL_TOLERANCE = 0.35


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-pi, pi]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Extract yaw from Isaac Lab quaternion tensor."""
    return euler_xyz_from_quat(quat)[2]


def world_to_body_xy(
    vec_w: torch.Tensor,
    yaw: torch.Tensor,
) -> torch.Tensor:
    """Rotate world-frame XY vectors into the robot body frame."""
    c = torch.cos(yaw)
    s = torch.sin(yaw)

    x = c * vec_w[..., 0] + s * vec_w[..., 1]
    y = -s * vec_w[..., 0] + c * vec_w[..., 1]

    return torch.stack((x, y), dim=-1)


# ---------------------------------------------------------------------------
# Deterministic PD controller
# ---------------------------------------------------------------------------

def pd_controller(
    cmd,
    robot,
    kp_xy: float,
    kd_xy: float,
    kp_yaw: float,
    kd_yaw: float,
) -> torch.Tensor:
    """
    Compute normalized [vx, vy, yaw_rate] for every environment.

    Position control:
        u_xy = Kp * lookahead_position - Kd * body_velocity

    Heading control:
        u_wz = Kp * heading_error - Kd * yaw_rate

    All three outputs are hard-clamped to [-1, +1].
    """
    device = robot.data.root_pos_w.device
    num_envs = robot.data.root_pos_w.shape[0]

    action = torch.zeros(num_envs, 3, device=device)

    # ------------------------------------------------------------------
    # First lookahead
    #
    # Stored as:
    #   [unit_dx, unit_dy, norm_dist]
    #
    # norm_dist = actual_distance / 0.5 m
    # ------------------------------------------------------------------
    la = cmd.lookaheads[:, 0, :].clone()

    actual_target_distance = (
        la[:, 2] * FIRST_LOOKAHEAD_DISTANCE
    )

    target_local = (
        la[:, :2]
        * actual_target_distance.unsqueeze(-1)
    )

    # ------------------------------------------------------------------
    # Robot velocity in body frame
    # ------------------------------------------------------------------
    yaw = yaw_from_quat(robot.data.root_quat_w)

    vel_local = world_to_body_xy(
        robot.data.root_lin_vel_w[:, :2],
        yaw,
    )

    # ------------------------------------------------------------------
    # XY PD
    # ------------------------------------------------------------------
    u_xy = (
        kp_xy * target_local
        - kd_xy * vel_local
    )

    action[:, 0] = torch.clamp(
        u_xy[:, 0] / MAX_VX,
        -1.0,
        1.0,
    )

    action[:, 1] = torch.clamp(
        u_xy[:, 1] / MAX_VY,
        -1.0,
        1.0,
    )

    # ------------------------------------------------------------------
    # Yaw-rate PD
    #
    # Lookahead direction is already in robot frame, so atan2(dy, dx)
    # directly gives the desired relative heading.
    # ------------------------------------------------------------------
    desired_heading_local = torch.atan2(
        la[:, 1],
        la[:, 0],
    )

    heading_error = wrap_angle(
        desired_heading_local
    )

    u_wz = (
        kp_yaw * heading_error
        - kd_yaw * robot.data.root_ang_vel_w[:, 2]
    )

    action[:, 2] = torch.clamp(
        u_wz / MAX_WZ,
        -1.0,
        1.0,
    )

    return action


# ---------------------------------------------------------------------------
# Independent geometric reference
# ---------------------------------------------------------------------------

def nearest_segment_projection(
    cmd,
    robot_pos_w: torch.Tensor,
):
    """
    Independently project each robot onto its nearest valid A* path segment.

    This does NOT modify NavigationWaypointCommand.

    Returns:
        projected_remaining:
            cumulative A* path distance from the projection point to goal.
        distance_to_path:
            perpendicular distance from robot to nearest path segment.
        projection_s:
            cumulative A* distance from path start to projection point.
    """
    path = cmd.path_world
    max_len = path.shape[1]

    starts = path[:, :-1]
    ends = path[:, 1:]

    vec = ends - starts

    seg_len_sq = (
        (vec * vec)
        .sum(dim=-1)
        .clamp(min=1e-8)
    )

    rel = (
        robot_pos_w.unsqueeze(1)
        - starts
    )

    t = (
        (rel * vec).sum(dim=-1)
        / seg_len_sq
    ).clamp(0.0, 1.0)

    projection = (
        starts
        + t.unsqueeze(-1) * vec
    )

    distance = torch.norm(
        robot_pos_w.unsqueeze(1)
        - projection,
        dim=-1,
    )

    segment_indices = torch.arange(
        max_len - 1,
        device=path.device,
    ).unsqueeze(0)

    valid = (
        segment_indices
        < (cmd.path_lengths - 1).unsqueeze(1)
    )

    distance = torch.where(
        valid,
        distance,
        torch.full_like(
            distance,
            float("inf"),
        ),
    )

    best_distance, best_segment = (
        distance.min(dim=1)
    )

    segment_length = torch.norm(
        vec,
        dim=-1,
    )

    cum_at_segment = (
        cmd.cum_dist
        .gather(
            1,
            best_segment.unsqueeze(1),
        )
        .squeeze(1)
    )

    best_t = (
        t.gather(
            1,
            best_segment.unsqueeze(1),
        )
        .squeeze(1)
    )

    best_segment_length = (
        segment_length
        .gather(
            1,
            best_segment.unsqueeze(1),
        )
        .squeeze(1)
    )

    projection_s = (
        cum_at_segment
        + best_t * best_segment_length
    )

    last_idx = (
        cmd.path_lengths - 1
    ).clamp(min=0)

    total_path_length = (
        cmd.cum_dist
        .gather(
            1,
            last_idx.unsqueeze(1),
        )
        .squeeze(1)
    )

    projected_remaining = (
        total_path_length
        - projection_s
    ).clamp(min=0.0)

    return (
        projected_remaining,
        best_distance,
        projection_s,
    )


# ---------------------------------------------------------------------------
# Episode records
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    path_x: np.ndarray | None
    path_y: np.ndarray | None
    traj_x: list
    traj_y: list


def snapshot_paths(cmd, num_envs):
    """Copy the current A* path of every environment."""
    paths = []

    for i in range(num_envs):
        length = int(
            cmd.path_lengths[i].item()
        )

        if length > 0:
            path = (
                cmd.path_world[
                    i,
                    :length,
                    :2,
                ]
                .detach()
                .cpu()
                .numpy()
                .copy()
            )
            paths.append(path)
        else:
            paths.append(None)

    return paths


def new_episode_records(paths):
    """Create one active episode record per environment."""
    records = []

    for path in paths:
        if path is None:
            records.append(
                EpisodeRecord(
                    None,
                    None,
                    [],
                    [],
                )
            )
        else:
            records.append(
                EpisodeRecord(
                    path[:, 0].copy(),
                    path[:, 1].copy(),
                    [],
                    [],
                )
            )

    return records


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def init_history():
    return defaultdict(list)


def append_history(history, values):
    for key, value in values.items():
        history[key].append(
            float(value)
        )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    output_dir,
    records_by_env,
    history,
):
    """
    Save separate figures.

    Per environment:
      1. A* path vs PD trajectory
      2. Remaining distances: dE, dPath, min(dE, dPath)
      3. Haro-style potential Phi = -min(dE, dPath)
      4. One-step potential progress
      5. Normalized PD commands
      4. Path-tracking errors
      5. Waypoint index
      6. Lookahead normalization
    """
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    for env_id, records in records_by_env.items():

        # ---------------------------------------------------------------
        # 1. Trajectory
        # ---------------------------------------------------------------
        fig, ax = plt.subplots(
            figsize=(9, 8)
        )

        for ep_idx, record in enumerate(records):

            if record.path_x is not None:
                ax.plot(
                    record.path_x,
                    record.path_y,
                    linewidth=1.5,
                    label=f"A* path ep {ep_idx}",
                )

            if record.traj_x:
                ax.plot(
                    record.traj_x,
                    record.traj_y,
                    linewidth=2.0,
                    label=f"PD trajectory ep {ep_idx}",
                )

        ax.set_title(
            f"HelixNav env {env_id}: A* path vs PD trajectory"
        )
        ax.set_xlabel("World X [m]")
        ax.set_ylabel("World Y [m]")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                output_dir,
                f"env_{env_id}_trajectory.png",
            ),
            dpi=180,
        )
        plt.close(fig)

        h = history[env_id]
        t = np.asarray(
            h["time"]
        )

        # ---------------------------------------------------------------
        # 2. Remaining-distance diagnostics
        # ---------------------------------------------------------------
        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.plot(
            t,
            h["euclidean"],
            label="Euclidean remaining",
        )

        ax.plot(
            t,
            h["along"],
            label="A* path remaining dPath",
        )

        ax.plot(
            t,
            np.minimum(
                np.asarray(h["euclidean"]),
                np.asarray(h["along"]),
            ),
            label="min(dE, dPath)",
            linewidth=2.0,
        )

        ax.set_title(
            f"Env {env_id}: remaining-distance diagnostics"
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Distance [m]")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                output_dir,
                f"env_{env_id}_remaining_distances.png",
            ),
            dpi=180,
        )
        plt.close(fig)

        # ---------------------------------------------------------------
        # 3. Haro-style potential
        #
        # Phi(s) = -min(d_euclidean, d_path)
        # ---------------------------------------------------------------
        d_euclidean = np.asarray(h["euclidean"])
        d_path = np.asarray(h["along"])
        d_min = np.minimum(d_euclidean, d_path)
        phi = -d_min

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t, phi, label="Phi = -min(dE, dPath)", linewidth=2.0)
        ax.axhline(0.0, linestyle="--", linewidth=1.0, label="goal / zero-distance")
        ax.set_title(f"Env {env_id}: Haro-style navigation potential")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Potential Phi")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"env_{env_id}_potential.png"), dpi=180)
        plt.close(fig)

        # ---------------------------------------------------------------
        # 4. One-step progress diagnostics
        # ---------------------------------------------------------------
        delta_euc = d_euclidean[:-1] - d_euclidean[1:]
        delta_path = d_path[:-1] - d_path[1:]
        delta_min = d_min[:-1] - d_min[1:]
        t_delta = t[1:]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t_delta, delta_euc, label="delta dE")
        ax.plot(t_delta, delta_path, label="delta dPath")
        ax.plot(t_delta, delta_min, label="delta min(dE,dPath)", linewidth=2.0)
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_title(f"Env {env_id}: one-step navigation progress")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Progress per navigation step [m]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"env_{env_id}_potential_progress.png"), dpi=180)
        plt.close(fig)

        # ---------------------------------------------------------------
        # 5. Commands
        # ---------------------------------------------------------------
        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.plot(
            t,
            h["cmd_vx"],
            label="vx command",
        )

        ax.plot(
            t,
            h["cmd_vy"],
            label="vy command",
        )

        ax.plot(
            t,
            h["cmd_wz"],
            label="yaw-rate command",
        )

        ax.set_title(
            f"Env {env_id}: normalized PD commands"
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Command [-1, 1]")
        ax.set_ylim(
            -1.1,
            1.1,
        )
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                output_dir,
                f"env_{env_id}_commands.png",
            ),
            dpi=180,
        )
        plt.close(fig)

        # ---------------------------------------------------------------
        # 4. Tracking error
        # ---------------------------------------------------------------
        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.plot(
            t,
            h["distance_to_path"],
            label="Distance to A* path",
        )

        ax.plot(
            t,
            h["robot_to_wp"],
            label="Distance to current waypoint",
        )

        ax.set_title(
            f"Env {env_id}: path-tracking error"
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Distance [m]")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                output_dir,
                f"env_{env_id}_tracking_error.png",
            ),
            dpi=180,
        )
        plt.close(fig)

        # ---------------------------------------------------------------
        # 5. Waypoint index
        # ---------------------------------------------------------------
        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.plot(
            t,
            h["wp_idx"],
            label="current waypoint index",
        )

        ax.set_title(
            f"Env {env_id}: waypoint progression"
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Waypoint index")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                output_dir,
                f"env_{env_id}_waypoint_index.png",
            ),
            dpi=180,
        )
        plt.close(fig)

        # ---------------------------------------------------------------
        # 6. Lookahead normalization
        # ---------------------------------------------------------------
        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        for k in range(8):
            ax.plot(
                t,
                h[f"la_norm_{k}"],
                label=f"LA{k}",
            )

        ax.set_title(
            f"Env {env_id}: lookahead distance normalization"
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(
            "Euclidean / requested distance"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(
            ncol=2,
            fontsize=8,
        )

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                output_dir,
                f"env_{env_id}_lookahead_norm.png",
            ),
            dpi=180,
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# Automatic conclusions
# ---------------------------------------------------------------------------

def print_summary(
    env,
    history,
    total_steps,
    elapsed_time,
):
    """
    Print hard structural checks plus navigation/PBRS diagnostics.

    All values are reconstructed from the current history schema:
        time, cmd_vx, cmd_vy, cmd_wz, wp_idx,
        goal_distance, distance_to_path, robot_to_wp,
        euclidean, along, phi, projection_s, reset_before,
        and lookahead normalization entries.

    Reset transitions are excluded from one-step progress and waypoint
    regression checks because the path/episode state legitimately jumps
    when an environment resets.
    """
    print("\n" + "=" * 78)
    print("HELIXNAV PD NAVIGATION DEBUG — FINAL CONCLUSIONS")
    print("=" * 78)

    print(f"Actual elapsed simulation time : {elapsed_time:.2f} s")
    print(f"Navigation steps              : {total_steps}")
    print(f"Parallel environments         : {env.num_envs}")

    overall_structural_pass = True

    for env_id in range(env.num_envs):
        h = history[env_id]

        if not h["time"]:
            print(f"\nENV {env_id}: no samples recorded.")
            overall_structural_pass = False
            continue

        # ------------------------------------------------------------------
        # Core telemetry
        # ------------------------------------------------------------------
        time = np.asarray(h["time"], dtype=float)
        goal_distance = np.asarray(h["goal_distance"], dtype=float)
        distance_to_path = np.asarray(h["distance_to_path"], dtype=float)
        euclidean = np.asarray(h["euclidean"], dtype=float)
        path_remaining = np.asarray(h["along"], dtype=float)
        waypoint_index = np.asarray(h["wp_idx"], dtype=float)
        reset_before = np.asarray(h["reset_before"], dtype=bool)

        d_min = np.minimum(euclidean, path_remaining)
        phi = -d_min

        # ------------------------------------------------------------------
        # Numerical telemetry check
        # ------------------------------------------------------------------
        numeric_series = [
            time,
            goal_distance,
            distance_to_path,
            euclidean,
            path_remaining,
            waypoint_index,
            np.asarray(h["cmd_vx"], dtype=float),
            np.asarray(h["cmd_vy"], dtype=float),
            np.asarray(h["cmd_wz"], dtype=float),
            phi,
        ]

        finite = all(np.isfinite(x).all() for x in numeric_series)

        # ------------------------------------------------------------------
        # Command magnitude
        # ------------------------------------------------------------------
        command_max = max(
            np.max(np.abs(np.asarray(h["cmd_vx"], dtype=float))),
            np.max(np.abs(np.asarray(h["cmd_vy"], dtype=float))),
            np.max(np.abs(np.asarray(h["cmd_wz"], dtype=float))),
        )

        # ------------------------------------------------------------------
        # Reset-aware waypoint regression
        # ------------------------------------------------------------------
        waypoint_regressions = int(
            np.sum(
                (~reset_before[1:])
                & (waypoint_index[1:] < waypoint_index[:-1] - 1e-6)
            )
        )

        # ------------------------------------------------------------------
        # Haro-style potential:
        #
        #   Phi(s) = -min(dE, dPath)
        #
        # One-step progress is:
        #
        #   delta_min = min(dE_prev,dPath_prev)
        #             - min(dE_next,dPath_next)
        #
        # This is intentionally NOT min(delta_dE, delta_dPath).
        # ------------------------------------------------------------------
        if len(euclidean) > 1:
            delta_euc = euclidean[:-1] - euclidean[1:]
            delta_path = path_remaining[:-1] - path_remaining[1:]
            delta_min = d_min[:-1] - d_min[1:]

            valid_delta = ~reset_before[1:]

            euclidean_progress_steps = int(
                np.sum((delta_euc > 0.0) & valid_delta)
            )
            path_progress_steps = int(
                np.sum((delta_path > 0.0) & valid_delta)
            )
            min_progress_steps = int(
                np.sum((delta_min > 0.0) & valid_delta)
            )
            valid_count = max(int(np.sum(valid_delta)), 1)
        else:
            euclidean_progress_steps = 0
            path_progress_steps = 0
            min_progress_steps = 0
            valid_count = 1

        potential_progress_fraction = (
            min_progress_steps / valid_count
        )

        # ------------------------------------------------------------------
        # Print metrics
        # ------------------------------------------------------------------
        print(f"\n--- ENV {env_id} ---")
        print(f"max |vx, vy, wz| command     : {command_max:.4f}")
        print(f"waypoint regressions         : {waypoint_regressions}")
        print(f"minimum goal distance        : {goal_distance.min():.4f} m")
        print(f"minimum A* path distance     : {distance_to_path.min():.4f} m")
        print(
            f"Euclidean progress steps     : "
            f"{euclidean_progress_steps}/{max(len(euclidean) - 1, 1)}"
        )
        print(
            f"dPath progress steps         : "
            f"{path_progress_steps}/{valid_count}"
        )
        print(
            f"delta[min] progress steps    : "
            f"{min_progress_steps}/{valid_count}"
        )
        print(
            f"positive-potential fraction  : "
            f"{potential_progress_fraction:.3f}"
        )
        print(f"minimum min(dE, dPath)       : {d_min.min():.4f} m")
        print(f"minimum Haro potential Phi   : {phi.min():.4f}")

        # ------------------------------------------------------------------
        # Hard check: command limits
        # ------------------------------------------------------------------
        if command_max > 1.00001:
            print("❌ FAIL: PD command exceeded [-1, 1].")
            overall_structural_pass = False
        else:
            print("✅ PASS: all PD commands stayed inside [-1, 1].")

        # ------------------------------------------------------------------
        # Hard check: waypoint monotonicity
        # ------------------------------------------------------------------
        if waypoint_regressions > 0:
            print("❌ FAIL: waypoint index regressed within an episode.")
            overall_structural_pass = False
        else:
            print("✅ PASS: waypoint index remained monotonic within episodes.")

        # ------------------------------------------------------------------
        # Hard check: numerical stability
        # ------------------------------------------------------------------
        if not finite:
            print("❌ FAIL: NaN/Inf found in telemetry.")
            overall_structural_pass = False
        else:
            print("✅ PASS: telemetry remained finite.")

        # ------------------------------------------------------------------
        # Goal result
        # ------------------------------------------------------------------
        if goal_distance.min() <= GOAL_TOLERANCE:
            print(
                f"✅ PASS: robot entered the diagnostic goal tolerance "
                f"({GOAL_TOLERANCE:.2f} m)."
            )
        else:
            print(
                f"⚠️ WARN: robot did not enter the diagnostic goal tolerance "
                f"({GOAL_TOLERANCE:.2f} m)."
            )

        # ------------------------------------------------------------------
        # Path-following result
        # ------------------------------------------------------------------
        if distance_to_path.min() < 0.20:
            print("✅ PASS: robot came within 0.20 m of the A* path.")
        else:
            print("⚠️ WARN: robot never came within 0.20 m of the A* path.")

    print("\n" + "-" * 78)

    if overall_structural_pass:
        print("OVERALL STRUCTURAL RESULT: ✅ PASS")
        print(
            "The PD command interface, numerical state, and reset-aware "
            "waypoint tracking passed."
        )
    else:
        print("OVERALL STRUCTURAL RESULT: ❌ FAIL")
        print("At least one hard structural check failed.")

    print("\nWHAT THIS TEST PROVES:")
    print(
        "  • NavigationWaypointCommand can run with multiple "
        "parallel environments."
    )
    print(
        "  • A deterministic controller can drive the same "
        "SE(2) command interface without keyboard input."
    )
    print(
        "  • The command output respects the requested "
        "[-1, +1] limits."
    )
    print(
        "  • Lookaheads, waypoint state and PBRS state remain "
        "finite during the run."
    )
    print(
        "  • The Haro-style diagnostic potential is computed as "
        "Phi = -min(dE, dPath)."
    )

    print("\nWHAT THIS TEST DOES NOT PROVE:")
    print("  • PPO learning performance.")
    print(
        "  • That the learned navigation policy will produce "
        "the same behavior as this PD controller."
    )
    print(
        "  • Collision-free navigation unless the trajectory plots "
        "and collision instrumentation are inspected."
    )

    print("\nMOST IMPORTANT GRAPHS:")
    print("  env_*_potential.png          -> Phi = -min(dE, dPath)")
    print(
        "  env_*_potential_progress.png -> "
        "delta dE, delta dPath, delta[min]"
    )
    print(
        "\nFor the Haro-style formulation, delta[min] is the state-potential "
        "distance delta; it is not min(delta dE, delta dPath)."
    )
    print("=" * 78)


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run(env):
    output_dir = os.path.join(
        os.path.dirname(
            os.path.realpath(__file__)
        ),
        "output_pd_debug",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    env.reset()
    env.setup_manager_visualizers()

    cmd = env.command_manager.get_term(
        "navigation_goal"
    )

    robot = env.scene["robot"]

    print("\n" + "=" * 78)
    print("HELIXNAV PD NAVIGATION DEBUG START")
    print("=" * 78)

    print(
        f"Requested environments : "
        f"{args.num_envs}"
    )

    print(
        f"Actual environments    : "
        f"{env.num_envs}"
    )

    print(
        f"Duration               : "
        f"{args.duration:.1f} s"
    )

    sim_dt = float(
        getattr(
            env.cfg.sim,
            "dt",
            0.005,
        )
    )

    decimation = int(
        getattr(
            env.cfg,
            "decimation",
            1,
        )
    )

    nav_dt = sim_dt * decimation

    print(
        f"Estimated navigation dt: "
        f"{nav_dt:.4f} s"
    )

    print(
        f"XY PD                 : "
        f"Kp={args.kp_xy}, Kd={args.kd_xy}"
    )

    print(
        f"Yaw PD                : "
        f"Kp={args.kp_yaw}, Kd={args.kd_yaw}"
    )

    print(
        "Command limits        : "
        "vx, vy, yaw_rate ∈ [-1, +1]"
    )

    print(
        "Controller target     : "
        "first 0.5 m lookahead"
    )

    print("=" * 78)

    histories = {
        i: init_history()
        for i in range(env.num_envs)
    }

    initial_paths = snapshot_paths(
        cmd,
        env.num_envs,
    )

    active_records = new_episode_records(
        initial_paths
    )

    completed_records = {
        i: []
        for i in range(env.num_envs)
    }

    total_steps = 0

    max_steps = max(
        1,
        int(
            args.duration
            / nav_dt
        ),
    )

    while (
        simulation_app.is_running()
        and total_steps < max_steps
    ):
        with torch.inference_mode():

            # -----------------------------------------------------------
            # PD action
            # -----------------------------------------------------------
            action = pd_controller(
                cmd=cmd,
                robot=robot,
                kp_xy=args.kp_xy,
                kd_xy=args.kd_xy,
                kp_yaw=args.kp_yaw,
                kd_yaw=args.kd_yaw,
            )

            # This is the actual final command sent to the environment.
            action.clamp_(
                -1.0,
                1.0,
            )

            # -----------------------------------------------------------
            # Step environment
            # -----------------------------------------------------------
            obs, rewards, terminated, truncated, extras = env.step(
                action
            )

            total_steps += 1

            robot_pos = (
                robot.data.root_pos_w[:, :2]
            )

            goal_pos = cmd.goal_pos_w

            goal_distance = torch.norm(
                robot_pos - goal_pos,
                dim=-1,
            )

            # Independent geometric reference.
            (
                projected_remaining,
                distance_to_path,
                projection_s,
            ) = nearest_segment_projection(
                cmd,
                robot_pos,
            )

            reset_mask = (
                terminated
                | truncated
            )

            # -----------------------------------------------------------
            # Record every environment
            # -----------------------------------------------------------
            for i in range(
                env.num_envs
            ):
                h = histories[i]

                current_wp_idx = int(
                    cmd.current_waypoint_idx[
                        i
                    ].item()
                )

                path_length = int(
                    cmd.path_lengths[
                        i
                    ].item()
                )

                last_idx = max(
                    path_length - 1,
                    0,
                )

                current_wp = cmd.path_world[
                    i,
                    current_wp_idx,
                ]

                robot_to_wp = torch.norm(
                    robot_pos[i]
                    - current_wp
                )

                append_history(
                    h,
                    {
                        "time": total_steps * nav_dt,
                        "x": robot_pos[
                            i,
                            0,
                        ].item(),
                        "y": robot_pos[
                            i,
                            1,
                        ].item(),
                        "goal_distance": goal_distance[
                            i
                        ].item(),
                        "distance_to_path": distance_to_path[
                            i
                        ].item(),
                        "robot_to_wp": robot_to_wp.item(),
                        "wp_idx": current_wp_idx,
                        "euclidean": cmd.path_remaining_euclidean[
                            i
                        ].item(),
                        "along": cmd.path_remaining_along[
                            i
                        ].item(),
                        "projection_s": projection_s[
                            i
                        ].item(),
                        "reset_before": bool(reset_mask[i].item()),
                        "cmd_vx": action[
                            i,
                            0,
                        ].item(),
                        "cmd_vy": action[
                            i,
                            1,
                        ].item(),
                        "cmd_wz": action[
                            i,
                            2,
                        ].item(),
                        "speed": torch.norm(
                            robot.data.root_lin_vel_w[
                                i,
                                :2,
                            ]
                        ).item(),
                    },
                )

                for k in range(8):
                    h[
                        f"la_norm_{k}"
                    ].append(
                        float(
                            cmd.lookaheads[
                                i,
                                k,
                                2,
                            ].item()
                        )
                    )

                active_records[
                    i
                ].traj_x.append(
                    robot_pos[
                        i,
                        0,
                    ].item()
                )

                active_records[
                    i
                ].traj_y.append(
                    robot_pos[
                        i,
                        1,
                    ].item()
                )

            # -----------------------------------------------------------
            # Compact console output
            # -----------------------------------------------------------
            if (
                total_steps
                % args.print_every
                == 0
            ):
                i = 0

                print(
                    f"[step {total_steps:5d}/{max_steps}] "
                    f"env0 "
                    f"pos=("
                    f"{robot_pos[i,0].item():+.2f},"
                    f"{robot_pos[i,1].item():+.2f}) "
                    f"wp={cmd.current_waypoint_idx[i].item():2d} "
                    f"goal={goal_distance[i].item():.2f}m "
                    f"dist_path={distance_to_path[i].item():.2f}m "
                    f"cmd=("
                    f"{action[i,0].item():+.2f},"
                    f"{action[i,1].item():+.2f},"
                    f"{action[i,2].item():+.2f})"
                )

                if env.num_envs > 1:
                    i = 1

                    print(
                        f"                  env1 "
                        f"pos=("
                        f"{robot_pos[i,0].item():+.2f},"
                        f"{robot_pos[i,1].item():+.2f}) "
                        f"wp={cmd.current_waypoint_idx[i].item():2d} "
                        f"goal={goal_distance[i].item():.2f}m "
                        f"dist_path={distance_to_path[i].item():.2f}m "
                        f"cmd=("
                        f"{action[i,0].item():+.2f},"
                        f"{action[i,1].item():+.2f},"
                        f"{action[i,2].item():+.2f})"
                    )

            # -----------------------------------------------------------
            # Episode boundaries
            # -----------------------------------------------------------
            if reset_mask.any():
                reset_ids = (
                    reset_mask
                    .nonzero(
                        as_tuple=True
                    )[0]
                    .tolist()
                )

                print(
                    f"[RESET] envs: "
                    f"{reset_ids}"
                )

                # env.step() has already performed the environment reset
                # before returning, so cmd now contains the new path.
                new_paths = snapshot_paths(
                    cmd,
                    env.num_envs,
                )

                for i in reset_ids:

                    completed_records[
                        i
                    ].append(
                        active_records[
                            i
                        ]
                    )

                    new_path = new_paths[i]

                    if new_path is None:
                        active_records[
                            i
                        ] = EpisodeRecord(
                            None,
                            None,
                            [],
                            [],
                        )
                    else:
                        active_records[
                            i
                        ] = EpisodeRecord(
                            new_path[:, 0].copy(),
                            new_path[:, 1].copy(),
                            [],
                            [],
                        )

            # -----------------------------------------------------------
            # Hard numerical checks
            # -----------------------------------------------------------
            if not torch.isfinite(
                action
            ).all():
                raise RuntimeError(
                    "PD controller produced NaN/Inf."
                )

            if not torch.isfinite(
                cmd.lookaheads
            ).all():
                raise RuntimeError(
                    "Navigation lookaheads contain NaN/Inf."
                )

            if not torch.isfinite(
                cmd.path_remaining_euclidean
            ).all():
                raise RuntimeError(
                    "Euclidean potential contains NaN/Inf."
                )

            if not torch.isfinite(
                cmd.path_remaining_along
            ).all():
                raise RuntimeError(
                    "Continuous along-path remaining distance contains NaN/Inf."
                )

    # Finish currently active episodes.
    for i in range(
        env.num_envs
    ):
        completed_records[
            i
        ].append(
            active_records[i]
        )

    plot_results(
        output_dir=output_dir,
        records_by_env=completed_records,
        history=histories,
    )

    print_summary(
        env=env,
        history=histories,
        total_steps=total_steps,
        elapsed_time=total_steps * nav_dt,
    )

    print(
        f"\nPlots saved to: "
        f"{output_dir}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cfg = HelixNavDebugBaseRLEnvCfg()

    # Force the requested parallelism for this debugger.
    cfg.scene.num_envs = args.num_envs

    env = ManagerBasedRLEnv(
        cfg
    )

    try:
        run(env)
    finally:
        env.close()


if __name__ == "__main__":
    main()
