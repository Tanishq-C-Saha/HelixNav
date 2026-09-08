"""
Function definitions for custom functions for Observations.
"""

from dataclasses import MISSING
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import MultiMeshRayCasterCamera
from isaaclab.assets import Articulation

import torch

from isaaclab.utils import math as math_utils

from .events import init_nav_state

# getting depth image from the env


def get_depth_images(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = MISSING,
    data_type: str = "distance_to_camera",
    normalize: bool = True,
) -> torch.Tensor:
    """Raw depth observation for the navigation policy."""

    # extract sensor from the scene
    sensor: MultiMeshRayCasterCamera = env.scene[sensor_cfg.name]

    depth_images = sensor.data.output[data_type].clone()
    max_distance = sensor.cfg.max_distance

    if normalize:
        depth_images = torch.clamp(
            torch.where(torch.isinf(depth_images), max_distance, depth_images),
            min=0.0,
            max=max_distance,
        )

        # normalize to give values form 0 to 1
        depth_images = depth_images / max_distance

    return depth_images


# calculate relative gaol vector
def get_lookahead_vectors(
    env,
    command_name="navigation_goal",
) -> torch.Tensor:
    """(N, 24) : 8 lookaheads × [unit_dx, unit_dy, norm_dist]."""
    cmd = env.command_manager.get_term(command_name)
    return cmd.lookaheads.reshape(env.num_envs, -1)


def get_snap_flags(
    env,
    command_name="navigation_goal",
) -> torch.Tensor:
    """(N, 8) : binary lookahead snap flags."""
    cmd = env.command_manager.get_term(command_name)
    return cmd.snap_flags


def get_base_velocity(
    env,
    asset_cfg=SceneEntityCfg("robot"),
) -> torch.Tensor:
    """(N, 3) : [vx, vy, yaw_rate] in robot body frame."""
    robot = env.scene[asset_cfg.name]

    lin = robot.data.root_lin_vel_b[:, :2]
    yaw_rate = robot.data.root_ang_vel_b[:, 2:3]

    return torch.cat(
        [lin, yaw_rate],
        dim=-1,
    )


def get_relative_goal_vector(
    env,
    command_name,
    asset_cfg,
    arena_size=12.0,
) -> torch.Tensor:
    """(N, 3) : [unit_dx, unit_dy, normalized_distance]
    in the robot body frame.
    """

    cmd = env.command_manager.get_term(command_name)
    robot = env.scene[asset_cfg.name]

    # XY only: goal and robot positions in world frame.
    goal_w = cmd.goal_pos_w[:, :2]
    robot_w = robot.data.root_pos_w[:, :2]

    # World-frame robot -> goal vector.
    diff_w = torch.zeros(
        env.num_envs,
        3,
        device=env.device,
    )
    diff_w[:, :2] = goal_w - robot_w

    # Rotate world vector into robot/body frame.
    robot_yaw_quat = math_utils.yaw_quat(robot.data.root_quat_w)

    diff_local = math_utils.quat_apply_inverse(
        robot_yaw_quat,
        diff_w,
    )[:, :2]

    # Direction + normalized distance.
    dist = torch.norm(
        diff_local,
        dim=-1,
        keepdim=True,
    ).clamp(min=1e-6)

    unit_dir = diff_local / dist
    norm_dist = dist / arena_size

    return torch.cat(
        [unit_dir, norm_dist],
        dim=-1,
    )
