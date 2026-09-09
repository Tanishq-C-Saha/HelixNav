# manager_configs/curriculums.py

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils.configclass import configclass

import helix_nav.tasks.manager_based.navigation.mdp as mdp


@configclass
class CurriculumsCfg:
    """Curriculum configuration."""

    map_difficulty = CurrTerm(
        func=mdp.advance_map_difficulty,
        params={
            "command_name": "navigation_goal",
            "window_size": 200,
            "min_episodes": 50,
            "max_difficulty": 3,
            "thresholds": {1: 0.70, 2: 0.65},
            "max_episodes_per_level": 100_000,
        },
    )