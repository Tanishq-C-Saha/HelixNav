

from dataclasses import MISSING
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor

import torch


# termination due to collisions
def terminations_by_collisions(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg = MISSING,
        force_threshold: float = 1
):
    """Check the collision sensor | contact sensor for collisions."""

    sensor: ContactSensor = env.scene[sensor_cfg.name]

    net_forces_w = sensor.data.net_forces_w

    forces_norm = torch.norm(net_forces_w, dim=-1)

    collisions = torch.any(forces_norm > force_threshold, dim=-1)

    return collisions


# termination on goal reached
def goal_reached(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        min_distance_threshold: float = 0.5,
):
    """Fire termination when goal reached.(dist_to_goal <= threshold)"""

    robot: Articulation = env.scene[asset_cfg.name]

    goal_w = env._goal_positions + env.scene.env_origins
    goal_w[:, 2] = 0

    robot_w = robot.data.root_pos_w.clone()
    robot_w[:, 2] = 0

    dist = torch.norm(goal_w - robot_w, dim=-1)

    return dist <= min_distance_threshold
