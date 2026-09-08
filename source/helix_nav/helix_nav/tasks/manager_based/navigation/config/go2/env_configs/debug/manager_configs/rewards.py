from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from helix_nav.tasks.manager_based.navigation import mdp


@configclass
class RewardCfg:
    """Rewards for HelixNav debug env."""

    goal_reached = RewTerm(
        func=mdp.goal_reached_reward,
        weight=50.0,
        params={
            "command_name": "navigation_goal",
            "min_goal_threshold": 0.3,
        },
    )

    progress = RewTerm(
        func=mdp.pbrs_progress_reward,
        weight=1.0,
        params={
            "gamma": 0.99,
            "command_name": "navigation_goal",
        },
    )

    smoothness = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

    collisions = RewTerm(
        func=mdp.is_terminated_term,
        weight=-50.0,
        params={
            "term_keys": "collisions",
        },
    )