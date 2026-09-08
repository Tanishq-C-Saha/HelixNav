from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from helix_nav.tasks.manager_based.navigation import mdp


@configclass
class RewardCfg:
    """Rewards for HelixNav debug env."""

    goal_reached = RewTerm(
        func=mdp.goal_reached_reward,
        params={
            "min_goal_threshold": 0.3  # ! should be equal to min goal threshold of terminations
        },
        weight=50
    )