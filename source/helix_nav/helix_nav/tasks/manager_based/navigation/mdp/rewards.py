
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation

import torch



# gola reached sparse reward 
def goal_reached_reward(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        min_goal_threshold: float = 0.3  #! should be same as terminations goal reached dist
):
    """Provide sparse reward on goal reahed."""

    robot: Articulation = env.scene[asset_cfg.name]

    goal_w = env._goal_positions[:, :2] + env.scene.env_origins[:, :2]
    
    robot_w = robot.data.root_pos_w[:, :2]

    dist = torch.norm(goal_w - robot_w, dim=-1)  # (N,)

    return (dist <= min_goal_threshold).float()
