
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.rewards import action_rate_l2

import torch




def goal_reached_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "navigation_goal",
    min_goal_threshold: float = 0.3,
) -> torch.Tensor:
    """Sparse reward: 1.0 when the robot is within the goal threshold."""

    cmd = env.command_manager.get_term(command_name)

    robot_w = env.scene["robot"].data.root_pos_w[:, :2]
    goal_w = cmd.goal_pos_w[:, :2]

    dist = torch.norm(robot_w - goal_w, dim=-1)

    return (dist <= min_goal_threshold).float()


def pbrs_progress_reward(
    env: ManagerBasedRLEnv,
    gamma: float = 0.99,
    command_name: str = "navigation_goal",
) -> torch.Tensor:
    """Hölder-Haro PBRS using Φ(s) = -min(d_euclidean, d_path)."""

    cmd = env.command_manager.get_term(command_name)

    # Current potential:
    # Φ(s_t) = -min(d_euclidean_t, d_path_t)
    current_min = torch.minimum(
        cmd.path_remaining_euclidean,
        cmd.path_remaining_along,
    )

    # Previous potential:
    # Φ(s_{t-1}) = -min(d_euclidean_{t-1}, d_path_{t-1})
    previous_min = torch.minimum(
        cmd.prev_path_remaining_euclidean,
        cmd.prev_path_remaining_along,
    )

    phi_current = -current_min
    phi_previous = -previous_min


    return phi_current - gamma * phi_previous