# mdp/curriculums.py

"""Curriculum terms for HelixNav CP7."""

from __future__ import annotations
from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv


def advance_map_difficulty(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_name: str = "robot",
    command_name: str = "navigation_goal",
    window_size: int = 200,
    min_episodes: int = 50,
    max_difficulty: int = 5,
    thresholds: dict | None = None,
    max_episodes_per_level: int = 100_000,
) -> dict:
    """Advance map difficulty using rolling goal-reach success rate.

    Called at the start of _reset_idx, so env_ids correspond to episodes
    that just ended. The command state still contains the previous episode's
    goal position at this point.
    """


    if not hasattr(env, "_curriculum_success_buf"):
        env._curriculum_success_buf = torch.zeros(
            window_size,
            device=env.device,
        )
        env._curriculum_write_idx = 0
        env._curriculum_total_episodes = 0

    # initalize current difficulty if not present
    if not hasattr(env, "_current_difficulty"):
        env._current_difficulty = 1

    if thresholds is None:
        thresholds = {
            "1": 0.70,
            "2": 0.65,
            "3": 0.60,
            "4": 0.55,
        }

    # CurriculumTermCfg.params passes through Hydra's config serialization
    # (class_to_dict), which requires all dict keys to be strings — normalize
    # back to int keys here, once, for the lookups below.
    thresholds = {int(k): v for k, v in thresholds.items()}

    # skip recording on the very first reset (goal_pos_w not yet set)
    if env._curriculum_total_episodes == 0 and not hasattr(env, "_nav_state_initialized"):
        return {
            "difficulty": env._current_difficulty,
            "rolling_success_rate": 0.0,
            "total_episodes": 0,
        }

    # ------------------------------------------------------------
    # Record outcomes of episodes that just ended
    # ------------------------------------------------------------
    cmd = env.command_manager.get_term(command_name)

    robot_pos_w = env.scene[asset_name].data.root_pos_w[env_ids, :2]
    goal_pos_w = cmd.goal_pos_w[env_ids, :2]

    dist = torch.norm(robot_pos_w - goal_pos_w, dim=-1)
    reached = (dist <= 0.3).float()

    for val in reached:
        idx = env._curriculum_write_idx % window_size
        env._curriculum_success_buf[idx] = val
        env._curriculum_write_idx += 1
        env._curriculum_total_episodes += 1

    # ------------------------------------------------------------
    # Check advancement
    # ------------------------------------------------------------
    current = env._current_difficulty

    rolling_sr = 0.0

    if env._curriculum_total_episodes >= min_episodes:
        rolling_sr = env._curriculum_success_buf.mean().item()

        success_advance = (
            current in thresholds
            and rolling_sr >= thresholds[current]
        )

        stuck_advance = (
            env._curriculum_total_episodes >= max_episodes_per_level
        )

        should_advance = (
            success_advance or stuck_advance
        )

        if should_advance and current < max_difficulty:
            if success_advance:
                reason = "success"
            else:
                reason = "stuck"

            env._current_difficulty += 1

            env._curriculum_success_buf.zero_()
            env._curriculum_write_idx = 0
            env._curriculum_total_episodes = 0

            print(
                f"[CURRICULUM] ▲ difficulty "
                f"{current} → {env._current_difficulty} "
                f"(reason={reason}, SR={rolling_sr:.2%})"
            )

    return {
        "difficulty": env._current_difficulty,
        "rolling_success_rate": rolling_sr,
        "total_episodes": env._curriculum_total_episodes,
    }