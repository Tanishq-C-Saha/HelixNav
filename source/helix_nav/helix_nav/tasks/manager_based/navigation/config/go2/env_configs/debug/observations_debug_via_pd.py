"""HelixNav debug script: keyboard-controlled Go2 with focused
navigation / PBRS debugging.

Debug modes:
    DEBUG_EXTRAS       -> general observations, rewards, reset information
    DEBUG_PBRS         -> path_remaining_euclidean, path_remaining_along and PBRS diagnostics
    DEBUG_LOOKAHEADS   -> lookahead vector diagnostics

The main purpose of this script is to inspect:
    - current waypoint tracking
    - cumulative path distances
    - robot-to-waypoint distance
    - path_remaining_euclidean
    - path_remaining_along
    - delta path_remaining_along
    - robot velocity
    - lookahead vectors
    - snap flags
    - individual reward terms

"""

import argparse
import os

import torch

from isaaclab.app import AppLauncher


# ──────────────────────────────────────────────
# CLI args
# ──────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="HelixNav keyboard debug environment."
)

parser.add_argument(
    "--save-camera-images",
    action="store_true",
    default=False,
    help="Save RGB + depth images every step (fills disk fast, use sparingly).",
)

parser.add_argument(
    "--save-occupancy-map",
    action="store_true",
    default=False,
    help="Save the global occupancy grid image on first reset.",
)

AppLauncher.add_app_launcher_args(parser=parser)

args, _ = parser.parse_known_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# ──────────────────────────────────────────────
# Imports that require Omniverse runtime
# ──────────────────────────────────────────────

from isaaclab.utils import configclass
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.sensors import MultiMeshRayCaster, ContactSensor
from isaaclab.assets import RigidObject, Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import yaw_quat, euler_xyz_from_quat

from helix_nav.tasks.manager_based.navigation.config.go2.env_configs.debug.helixnav_debug_base_rl_env_cfg import (
    HelixNavDebugBaseRLEnvCfg,
)

from helix_nav.tasks.manager_based.navigation.map_generators import (
    RandomMapGenerator,
    visualize_map_spec,
)

from helix_nav.tasks.manager_based.navigation.config.go2.env_configs.debug.debug_viz_utils import (
    save_images_grid,
    save_occupancy_map,
)


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

STATIC_OBSTACLE_COUNT = 13
FORCE_THRESHOLD = 1.0


# ──────────────────────────────────────────────
# Debug flags
# ──────────────────────────────────────────────

# General debug information:
# observations, full reward tensors, terminated/truncated, etc.
DEBUG_EXTRAS = False

# Main navigation/PBRS diagnostics.
DEBUG_PBRS = True

# Lookahead diagnostics.
DEBUG_LOOKAHEADS = True

# How often to print navigation diagnostics.
#
# 1  -> every step
# 10 -> every 10 steps
# 100 -> every 100 steps
DEBUG_EVERY_N_STEPS = 1

# Which parallel environment to inspect.
DEBUG_ENV_ID = 0


# ──────────────────────────────────────────────
# Keyboard config
# ──────────────────────────────────────────────

@configclass
class MyKeyboardCfg(Se2KeyboardCfg):

    v_x_sensitivity: float = 1.0
    v_y_sensitivity: float = 0.5
    omega_z_sensitivity: float = 1.0


# ──────────────────────────────────────────────
# CommandTerm verification
# ──────────────────────────────────────────────

def verify_command_term(env: ManagerBasedRLEnv):
    """Run CommandTerm sanity checks after env.reset()."""

    print("\n" + "=" * 60)
    print("COMMAND TERM VERIFICATION")
    print("=" * 60)

    n = 0

    cmd = env.command_manager.get_term("navigation_goal")

    # ──────────────────────────────────────────
    # Shape checks
    # ──────────────────────────────────────────

    assert cmd.lookaheads.shape == (
        env.num_envs,
        8,
        3,
    ), (
        f"❌ lookaheads shape wrong: "
        f"got {cmd.lookaheads.shape}, "
        f"expected ({env.num_envs}, 8, 3)"
    )

    print(f"✅ lookaheads shape: {cmd.lookaheads.shape}")

    assert cmd.snap_flags.shape == (
        env.num_envs,
        8,
    ), (
        f"❌ snap_flags shape wrong: "
        f"got {cmd.snap_flags.shape}, "
        f"expected ({env.num_envs}, 8)"
    )

    print(f"✅ snap_flags shape: {cmd.snap_flags.shape}")

    # ──────────────────────────────────────────
    # Path remaining sanity
    # ──────────────────────────────────────────

    assert cmd.path_remaining_euclidean.shape == (
        env.num_envs,
    ), (
        f"❌ path_remaining_euclidean shape wrong: "
        f"got {cmd.path_remaining_euclidean.shape}, "
        f"expected ({env.num_envs},)"
    )

    assert cmd.path_remaining_along.shape == (
        env.num_envs,
    ), (
        f"❌ path_remaining_along shape wrong: "
        f"got {cmd.path_remaining_along.shape}, "
        f"expected ({env.num_envs},)"
    )

    assert cmd.path_remaining_euclidean[n] > 0, (
        f"❌ path_remaining_euclidean is zero for env {n} "
        f"despite valid A* path"
    )

    assert cmd.path_remaining_along[n] > 0, (
        f"❌ path_remaining_along is zero for env {n} "
        f"despite valid A* path"
    )

    print(
        f"✅ path_remaining_euclidean[env {n}] = "
        f"{cmd.path_remaining_euclidean[n].item():.3f} m"
    )

    print(
        f"✅ path_remaining_along[env {n}] = "
        f"{cmd.path_remaining_along[n].item():.3f} m"
    )
    # ──────────────────────────────────────────
    # Find successful A* paths
    # ──────────────────────────────────────────

    successful_envs = (
        (env._path_lengths > 0)
        .nonzero(as_tuple=True)[0]
    )

    print(
        f"\nEnvs with valid A* path: "
        f"{len(successful_envs)}/{env.num_envs}"
    )

    if len(successful_envs) == 0:

        print(
            "⚠️ WARNING: no envs have a valid A* path. "
            "Cannot verify lookahead values."
        )

        print(
            "   This could mean: A* is failing for all envs, "
            "or obstacle density is too high."
        )

        return

    n = successful_envs[0].item()

    # ──────────────────────────────────────────
    # Lookahead sanity
    # ──────────────────────────────────────────

    assert cmd.lookaheads[n].abs().sum() > 0, (
        f"❌ lookaheads all zero for env {n} after reset — "
        f"_resample_command not firing?"
    )

    print(
        f"✅ lookaheads[env {n}] nonzero after reset"
    )

    # ──────────────────────────────────────────
    # Path remaining sanity
    # ──────────────────────────────────────────

    path_remaining = cmd.path_remaining_along[n]

    if not torch.isfinite(path_remaining):
        raise RuntimeError(
            f"❌ path_remaining_along is non-finite for env {n}: "
            f"{path_remaining.item()}"
        )
    if path_remaining < 0:
        raise RuntimeError(
            f"❌ path_remaining_along is negative for env {n}: "
            f"{path_remaining.item()}"
        )
    if path_remaining == 0:
        print(f"⚠️ path_remaining_along[env {n}] = 0.000 m (robot may be at/near goal)")
    else:
        print(f"✅ path_remaining_along[env {n}] = {path_remaining.item():.3f} m")

    # ──────────────────────────────────────────
    # Full initial state
    # ──────────────────────────────────────────

    print(f"\nEnv {n} full state after reset:")

    print(
        f"  path_length      = "
        f"{cmd.path_lengths[n].item()} waypoints"
    )

    print(
        f"  path_remaining   = "
        f"{cmd.path_remaining_along[n].item():.3f} m"
    )

    print(
        f"  current_wp_idx   = "
        f"{cmd.current_waypoint_idx[n].item()}"
    )

    print(
        f"  goal_pos_w       = "
        f"{cmd.goal_pos_w[n].tolist()}"
    )

    print(
        f"  lookahead[0]     = "
        f"{cmd.lookaheads[n, 0].tolist()} "
        f"(dx, dy, norm_dist @ 0.5m)"
    )

    print(
        f"  lookahead[7]     = "
        f"{cmd.lookaheads[n, 7].tolist()} "
        f"(dx, dy, norm_dist @ 8.0m)"
    )

    print(
        f"  snap_flags       = "
        f"{cmd.snap_flags[n].tolist()}"
    )

    # ──────────────────────────────────────────
    # Step once with zero action
    # ──────────────────────────────────────────

    lookahead_before = cmd.lookaheads[n].clone()
    path_remaining_before = cmd.path_remaining_along[n].item()

    env.step(
        torch.zeros(
            env.num_envs,
            3,
            device=env.device,
        )
    )

    lookahead_after = cmd.lookaheads[n]
    path_remaining_after = cmd.path_remaining_along[n].item()

    print("\nAfter 1 step (zero action):")

    print(
        f"  path_remaining: "
        f"{path_remaining_before:.3f} "
        f"-> "
        f"{path_remaining_after:.3f}"
    )

    print(
        f"  lookahead[0] changed: "
        f"{not torch.allclose(lookahead_before[0], lookahead_after[0])}"
    )

    # ──────────────────────────────────────────
    # NaN checks
    # ──────────────────────────────────────────

    assert not torch.isnan(cmd.lookaheads).any(), (
        "❌ NaN in lookaheads after step"
    )

    assert not torch.isnan(cmd.path_remaining_along).any(), (
        "❌ NaN in path_remaining after step"
    )

    print("✅ No NaN after step")

    print("\n" + "=" * 60)
    print("✅ COMMAND TERM VERIFIED — proceeding to main loop")
    print("=" * 60 + "\n")


# ──────────────────────────────────────────────
# Navigation / PBRS debug
# ──────────────────────────────────────────────

def debug_navigation_state(
    env: ManagerBasedRLEnv,
    cmd,
    env_id: int,
):
    """Print detailed navigation and PBRS state for one environment."""

    i = env_id

    # ──────────────────────────────────────────
    # Robot state
    # ──────────────────────────────────────────

    robot_pos = cmd.robot.data.root_pos_w[i, :2]

    robot_vel = cmd.robot.data.root_lin_vel_w[i, :2]

    robot_speed = torch.norm(robot_vel)

    # ──────────────────────────────────────────
    # Current waypoint
    # ──────────────────────────────────────────

    wp_idx = cmd.current_waypoint_idx[i]

    current_wp = cmd.path_world[
        i,
        wp_idx,
    ]

    # ──────────────────────────────────────────
    # Cumulative path distances
    # ──────────────────────────────────────────

    cum_at_wp = cmd.cum_dist[
        i,
        wp_idx,
    ]

    last_idx = max(
        cmd.path_lengths[i].item() - 1,
        0,
    )

    cum_at_goal = cmd.cum_dist[
        i,
        last_idx,
    ]

    path_dist = cum_at_goal - cum_at_wp

    # ──────────────────────────────────────────
    # Robot → current waypoint
    # ──────────────────────────────────────────

    robot_to_wp = torch.norm(
        robot_pos - current_wp
    )

    # ──────────────────────────────────────────
    # Reconstruct path_remaining
    # ──────────────────────────────────────────

    reconstructed_remaining = (
        path_dist + robot_to_wp
    )

    # ──────────────────────────────────────────
    # PBRS progress
    # ──────────────────────────────────────────

    previous_remaining = (
        cmd.prev_path_remaining_along[i]
    )

    current_remaining = (
        cmd.path_remaining_along[i]
    )

    delta_path = (
        previous_remaining - current_remaining
    )

    # ──────────────────────────────────────────
    # Print
    # ──────────────────────────────────────────

    print("\n" + "-" * 75)
    print(
        f"NAVIGATION / PBRS DEBUG — "
        f"env {i}"
    )
    print("-" * 75)

    # Robot
    print("\n--- ROBOT ---")

    print(
        f"robot_pos_w          : "
        f"{robot_pos.tolist()}"
    )

    print(
        f"robot_velocity_w     : "
        f"{robot_vel.tolist()}"
    )

    print(
        f"robot_speed          : "
        f"{robot_speed.item():.6f} m/s"
    )

    print(
        f"goal_pos_w           : "
        f"{cmd.goal_pos_w[i].tolist()}"
    )
    # ============================================================
    # PRINT ACTUAL A* PATH WAYPOINTS
    # ============================================================

    print("\n========== ACTUAL PATH ==========")

    i = env_id

    n = int(cmd.path_lengths[i].item())

    for k in range(min(n, 15)):
        wp = cmd.path_world[i, k]

        print(
            f"WP[{k:2d}] "
            f"pos=({wp[0].item():6.3f}, {wp[1].item():6.3f}) "
            f"cum={cmd.cum_dist[i,k].item():6.3f}"
        )

    # Path tracking
    print("\n--- PATH TRACKING ---")

    print(
        f"path_length          : "
        f"{cmd.path_lengths[i].item()} waypoints"
    )

    print(
        f"current_wp_idx       : "
        f"{wp_idx.item()}"
    )

    print(
        f"current_wp_w         : "
        f"{current_wp.tolist()}"
    )

    # Path distances
    print("\n--- PATH DISTANCES ---")

    print(
        f"cum_at_wp            : "
        f"{cum_at_wp.item():.6f} m"
    )

    print(
        f"cum_at_goal          : "
        f"{cum_at_goal.item():.6f} m"
    )

    print(
        f"path_dist (wp→goal)  : "
        f"{path_dist.item():.6f} m"
    )

    print(
        f"robot_to_wp          : "
        f"{robot_to_wp.item():.6f} m"
    )

    # PBRS
    print("\n--- PBRS STATE ---")

    print(
        f"prev_path_remaining  : "
        f"{previous_remaining.item():.6f} m"
    )

    print(
        f"path_remaining       : "
        f"{current_remaining.item():.6f} m"
    )

    print(
        f"delta_path           : "
        f"{delta_path.item():+.6f} m"
    )

    if delta_path > 0:
        print(
            "  ↑ Robot made progress "
            "according to path_remaining."
        )

    elif delta_path < 0:
        print(
            "  ↓ path_remaining increased."
        )

    else:
        print(
            "  = path_remaining unchanged."
        )

    # Consistency
    print("\n--- PATH_REMAINING CONSISTENCY ---")

    print(
        f"path_dist + robot_to_wp : "
        f"{reconstructed_remaining.item():.6f} m"
    )

    print(
        f"stored path_remaining   : "
        f"{current_remaining.item():.6f} m"
    )

    difference = (
        reconstructed_remaining
        - current_remaining
    )

    print(
        f"difference              : "
        f"{difference.item():+.8f} m"
    )

    # Waypoint transition warning
    if previous_remaining != current_remaining:

        print("\n--- CHANGE DIAGNOSTIC ---")

        print(
            f"waypoint index          : "
            f"{wp_idx.item()}"
        )

        print(
            f"robot speed             : "
            f"{robot_speed.item():.6f} m/s"
        )

        print(
            f"robot_to_wp             : "
            f"{robot_to_wp.item():.6f} m"
        )

        if robot_speed < 0.01:

            print(
                "⚠️ Robot is almost stationary "
                "while path_remaining changed."
            )

            print(
                "   Check whether current_waypoint_idx "
                "changed because of the 0.3 m threshold."
            )

    print("-" * 75)


# ──────────────────────────────────────────────
# Lookahead debug
# ──────────────────────────────────────────────

def debug_lookaheads(cmd, env_id: int):
    """Print all lookahead vectors for one environment."""

    i = env_id

    print("\n" + "-" * 75)
    print(
        f"LOOKAHEAD DEBUG — env {i}"
    )
    print("-" * 75)

    for k in range(
        cmd.lookaheads.shape[1]
    ):

        dx, dy, norm_dist = (
            cmd.lookaheads[i, k]
        )

        snap = cmd.snap_flags[
            i,
            k,
        ]

        print(
            f"LA[{k}] "
            f"dir=({dx.item():+.4f}, "
            f"{dy.item():+.4f}) "
            f"norm_dist={norm_dist.item():.4f} "
            f"snap={int(snap.item())}"
        )

    print("-" * 75)


# ──────────────────────────────────────────────
# Reward debug
# ──────────────────────────────────────────────

def debug_rewards(
    reward_manager,
    env_id: int,
):
    """Print individual reward terms for one environment."""

    print("\n--- REWARD TERMS ---")

    for term_idx, name in enumerate(
        reward_manager._term_names
    ):

        reward_value = (
            reward_manager
            ._step_reward[
                term_idx,
                env_id,
            ]
        )

        print(
            f"{name:30s}: "
            f"{reward_value.item():+.8f}"
        )


# ──────────────────────────────────────────────
# Observation audit
# ──────────────────────────────────────────────

DEBUG_OBSERVATIONS = True
OBS_STATS_EVERY_N_STEPS = 1
OBS_PRINT_VALUES = True
OBS_STOP_ON_NONFINITE = False


def _audit_observation_tensor(name, value, env_id=0):
    """Audit one observation tensor for finiteness, shape, and useful ranges."""
    if not isinstance(value, torch.Tensor):
        print(f"⚠️ OBS {name}: expected torch.Tensor, got {type(value).__name__}")
        return False

    finite = torch.isfinite(value)
    ok = bool(finite.all().item())
    nan_count = int(torch.isnan(value).sum().item())
    inf_count = int(torch.isinf(value).sum().item())

    if value.numel() > 0 and ok:
        vmin = value.min().item()
        vmax = value.max().item()
        mean = value.float().mean().item()
        std = value.float().std(unbiased=False).item()
        print(
            f"  {'✅' if ok else '🚨'} {name:20s} "
            f"shape={tuple(value.shape)!s:18s} "
            f"min={vmin:+.5g} max={vmax:+.5g} "
            f"mean={mean:+.5g} std={std:.5g}"
        )
    else:
        print(
            f"  🚨 {name:20s} shape={tuple(value.shape)} "
            f"NaN={nan_count} Inf={inf_count}"
        )

    if not ok:
        print(
            f"\n🚨 NON-FINITE OBSERVATION: {name}\n"
            f"   NaN count: {nan_count}\n"
            f"   Inf count: {inf_count}\n"
            f"   This must be fixed before PPO training."
        )
        if OBS_STOP_ON_NONFINITE:
            raise RuntimeError(f"Non-finite observation detected: {name}")
        return False

    # Semantic/range checks for the current HelixNav observation design.
    if name.endswith("depth_images"):
        if value.min() < -1e-6 or value.max() > 1.0 + 1e-6:
            print(
                f"  ⚠️ {name}: normalized depth is outside [0, 1] "
                f"({value.min().item():.5g}, {value.max().item():.5g})"
            )

    elif name.endswith("snap_flags"):
        if not bool(((value == 0) | (value == 1)).all().item()):
            print(f"  ⚠️ {name}: contains values other than 0/1")

    elif name.endswith("lookahead_vectors"):
        if value.ndim == 2 and value.shape[-1] == 24:
            la = value.reshape(value.shape[0], 8, 3)
            dirs = la[..., :2]
            norm_dist = la[..., 2]
            dir_norm = torch.linalg.vector_norm(dirs, dim=-1)
            if dir_norm.max() > 1.0 + 1e-5:
                print("  ⚠️ lookahead direction norm > 1")
            if norm_dist.min() < -1e-6:
                print("  ⚠️ lookahead norm_dist is negative")
            if env_id < value.shape[0] and OBS_PRINT_VALUES:
                print(f"     env {env_id} lookahead[0] = {la[env_id, 0].tolist()}")

    elif name.endswith("relative_goal"):
        if value.ndim == 2 and value.shape[-1] == 3:
            direction = value[:, :2]
            norm_dist = value[:, 2]
            direction_norm = torch.linalg.vector_norm(direction, dim=-1)
            if direction_norm.max() > 1.0 + 1e-5:
                print("  ⚠️ relative_goal direction norm > 1")
            if norm_dist.min() < -1e-6:
                print("  ⚠️ relative_goal normalized distance is negative")
            if env_id < value.shape[0] and OBS_PRINT_VALUES:
                print(f"     env {env_id} relative_goal = {value[env_id].tolist()}")

    elif name.endswith("prev_actions"):
        if value.min() < -1.0 - 1e-5 or value.max() > 1.0 + 1e-5:
            print("  ⚠️ prev_actions outside expected [-1, 1] range")

    return True


def audit_observations(obs, env_id=0, step=None):
    """Recursively audit every tensor returned in the policy observations."""
    print("\n" + "=" * 75)
    print(f"OBSERVATION AUDIT — step {step if step is not None else '?'}")
    print("=" * 75)

    all_finite = True

    def visit(obj, prefix="obs"):
        nonlocal all_finite
        if isinstance(obj, torch.Tensor):
            all_finite &= _audit_observation_tensor(prefix, obj, env_id)
        elif isinstance(obj, dict):
            for key, child in obj.items():
                visit(child, f"{prefix}[{key!r}]")
        elif isinstance(obj, (list, tuple)):
            for idx, child in enumerate(obj):
                visit(child, f"{prefix}[{idx}]")
        else:
            print(f"  ⚠️ {prefix}: non-tensor value {type(obj).__name__}")

    visit(obs)

    # Expected current policy observation structure.
    policy = obs.get("policy") if isinstance(obs, dict) else None
    if isinstance(policy, dict):
        expected = {
            "prev_actions": 3,
            "lookahead_vectors": 24,
            "snap_flags": 8,
            "base_velocity": 3,
            "relative_goal": 3,
        }
        for key, width in expected.items():
            value = policy.get(key)
            if isinstance(value, torch.Tensor):
                if value.ndim < 2 or value.shape[-1] != width:
                    print(
                        f"  🚨 {key}: expected last dimension {width}, "
                        f"got {tuple(value.shape)}"
                    )
            else:
                print(f"  🚨 Missing/non-tensor policy observation: {key}")

        depth = policy.get("depth_images")
        if isinstance(depth, torch.Tensor):
            expected_shape = (env_id * 0 + depth.shape[0], 1, 54, 96)
            if tuple(depth.shape[1:]) != expected_shape[1:]:
                print(
                    f"  ⚠️ depth_images: expected per-env shape (1, 54, 96), "
                    f"got {tuple(depth.shape[1:])}"
                )

        vector_width = sum(expected.values())
        print(f"\nVector observation width: {vector_width}D")
        print("Expected: 3 + 24 + 8 + 3 + 3 = 41D")
        print("Depth branch: (1, 54, 96)")

    if all_finite:
        print("\n✅ OBSERVATION AUDIT PASSED — no NaN/Inf detected")
    else:
        print("\n🚨 OBSERVATION AUDIT FAILED — NaN/Inf detected")

    return all_finite


# ──────────────────────────────────────────────
# Main simulation
# ──────────────────────────────────────────────

def run_simulation(
    env: ManagerBasedRLEnv,
    keyboard_controller: Se2Keyboard,
):
    """Run the keyboard-controlled simulation loop."""

    output_dir = os.path.join(
        os.path.dirname(
            os.path.realpath(__file__)
        ),
        "output",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    nav_command = torch.zeros(
        env.num_envs,
        3,
        device=env.device,
    )

    collision_sensor: ContactSensor = (
        env.scene["collision_sensor"]
    )

    robot: Articulation = (
        env.scene["robot"]
    )

    reward_manager = env.reward_manager

    # Handle to our navigation command term
    cmd = env.command_manager.get_term(
        "navigation_goal"
    )

    step_count = 0

    while simulation_app.is_running():

        with torch.inference_mode():

            # ──────────────────────────────────
            # Keyboard
            # ──────────────────────────────────

            vx, vy, yaw_rate = (
                keyboard_controller.advance()
            )

            nav_command[:, 0] = vx
            nav_command[:, 1] = vy
            nav_command[:, 2] = yaw_rate

            # Collision forces
            net_forces_w = (
                collision_sensor
                .data
                .net_forces_w
                .clone()
            )

            # ──────────────────────────────────
            # Simulation step
            # ──────────────────────────────────

            obs, rewards, terminated, truncated, extras = (
                env.step(nav_command)
            )

            step_count += 1

            if (
                DEBUG_OBSERVATIONS
                and step_count % OBS_STATS_EVERY_N_STEPS == 0
            ):
                audit_observations(
                    obs,
                    env_id=DEBUG_ENV_ID,
                    step=step_count,
                )

            # ──────────────────────────────────
            # Focused PBRS debugging
            # ──────────────────────────────────

            if (
                DEBUG_PBRS
                and step_count % DEBUG_EVERY_N_STEPS == 0
            ):

                debug_navigation_state(
                    env=env,
                    cmd=cmd,
                    env_id=DEBUG_ENV_ID,
                )

                debug_rewards(
                    reward_manager=reward_manager,
                    env_id=DEBUG_ENV_ID,
                )

            # ──────────────────────────────────
            # Lookahead debugging
            # ──────────────────────────────────

            if (
                DEBUG_LOOKAHEADS
                and step_count % DEBUG_EVERY_N_STEPS == 0
            ):

                debug_lookaheads(
                    cmd=cmd,
                    env_id=DEBUG_ENV_ID,
                )

            # ──────────────────────────────────
            # General debugging
            # ──────────────────────────────────

            if DEBUG_EXTRAS:

                print(
                    f"\n[DEBUG]: "
                    f"Obs rel_goal = "
                    f"{obs['policy']['relative_goal']}"
                )

                print(
                    f"[DEBUG]: "
                    f"path_remaining_along = "
                    f"{cmd.path_remaining_along}"
                )

                print(
                    f"[DEBUG]: "
                    f"current_waypoint_idx = "
                    f"{cmd.current_waypoint_idx}"
                )

                print(
                    f"[DEBUG]: "
                    f"snap_flags = "
                    f"{cmd.snap_flags}"
                )

                print(
                    f"[DEBUG]: "
                    f"reward_manager terms = "
                    f"{reward_manager._term_names}"
                )

                print(
                    f"[DEBUG]: "
                    f"reward_manager step rewards = "
                    f"{reward_manager._step_reward}"
                )

                print(
                    f"[DEBUG]: "
                    f"rewards = {rewards}"
                )

                print(
                    f"[DEBUG]: "
                    f"terminated = {terminated}"
                )

                print(
                    f"[DEBUG]: "
                    f"truncated = {truncated}"
                )

            # ──────────────────────────────────
            # NaN / Inf checks
            # ──────────────────────────────────

            if torch.isnan(rewards).any():

                print(
                    f"\n🚨 NaN DETECTED IN REWARDS "
                    f"at step {step_count}!"
                )

                print(
                    f"reward breakdown: "
                    f"{reward_manager._step_reward}"
                )

            if torch.isinf(rewards).any():

                print(
                    f"\n🚨 INF DETECTED IN REWARDS "
                    f"at step {step_count}!"
                )

            if torch.isnan(
                cmd.lookaheads
            ).any():

                print(
                    f"\n🚨 NaN DETECTED IN LOOKAHEADS "
                    f"at step {step_count}!"
                )

            if torch.isnan(
                cmd.path_remaining_along
            ).any():

                print(
                    f"\n🚨 NaN DETECTED IN "
                    f"PATH_REMAINING "
                    f"at step {step_count}!"
                )

            # ──────────────────────────────────
            # Episode reset debugging
            # ──────────────────────────────────

            if (
                terminated.any()
                or truncated.any()
            ):

                reset_envs = (
                    terminated | truncated
                ).nonzero(
                    as_tuple=True
                )[0]

                if DEBUG_EXTRAS or DEBUG_PBRS:

                    print(
                        "\n[DEBUG] envs resetting: "
                        f"{reset_envs.tolist()}"
                    )


            # ──────────────────────────────────
            # Optional camera images
            # ──────────────────────────────────

            if args.save_camera_images:

                rgb_images = (
                    env.scene[
                        "rgb_camera"
                    ]
                    .data
                    .output["rgb"]
                )

                depth_images = (
                    env.scene[
                        "depth_camera"
                    ]
                    .data
                    .output[
                        "distance_to_camera"
                    ]
                )

                for i in range(
                    env.num_envs
                ):

                    save_images_grid(
                        images=[
                            rgb_images[i],
                            depth_images[i],
                        ],
                        subtitles=[
                            "RGB",
                            "Depth",
                        ],
                        cmap="turbo",
                        title=(
                            "MultiMeshRayCasterCamera "
                            "on Unitree Go2 (obs)"
                        ),
                        filename=os.path.join(
                            output_dir,
                            "camera_frames",
                            f"env_{i}",
                            f"{env._sim_step_counter:04d}.jpg",
                        ),
                    )


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():

    env_cfg = (
        HelixNavDebugBaseRLEnvCfg()
    )

    env = ManagerBasedRLEnv(
        env_cfg
    )

    env.reset()

    env.setup_manager_visualizers()

    # ──────────────────────────────────────────
    # Verify CommandTerm
    # ──────────────────────────────────────────

    verify_command_term(env)

    # ──────────────────────────────────────────
    # Keyboard
    # ──────────────────────────────────────────

    keyboard_controller = Se2Keyboard(
        cfg=MyKeyboardCfg()
    )

    keyboard_controller.reset()

    print(
        "[INFO]: Setup complete."
    )

    # Second reset before actual run
    env.reset()

    run_simulation(
        env=env,
        keyboard_controller=keyboard_controller,
    )

    env.close()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":

    main()

    simulation_app.close()