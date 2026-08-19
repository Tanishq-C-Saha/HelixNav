"""
Function definitions for custom functions for Observations.
"""

# TODO(cp8): migrate nav state to CommandTerm

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
    normalize: bool = True
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
        depth_images = depth_images/max_distance

    return depth_images



# calculate relative gaol vector
def get_relative_goal_vector(
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        arena_size: float = MISSING,
):  
    """Relative goal vector."""

    # intialize env state
    if not hasattr(env, "_nav_state_initialized"):
        init_nav_state(env)

    robot: Articulation = env.scene[asset_cfg.name]

    goal_w = env._goal_positions + env.scene.env_origins
    robot_w = robot.data.root_pos_w.clone()

    diff_w = goal_w - robot_w
    diff_w[:, 2] = 0.0  # project to ground plane

    robot_yaw_quat = math_utils.yaw_quat(robot.data.root_quat_w)

    # goal vector in robot frame
    diff_local = math_utils.quat_apply_inverse(robot_yaw_quat, goal_w - robot_w)[:, :2]  # (N,2)

    dist = torch.norm(diff_local, dim=-1, keepdim=True).clamp(min=1e-6)  #(N,1)

    unit_dir = diff_local/dist  # (N,2)

    norm_dist = dist/arena_size

    return torch.cat([unit_dir, norm_dist], dim=-1)
    




    