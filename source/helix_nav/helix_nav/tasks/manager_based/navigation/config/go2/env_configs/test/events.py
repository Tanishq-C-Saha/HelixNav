from isaaclab.utils import configclass

from isaaclab.managers import EventTermCfg as EventTerm

from helix_nav.tasks.manager_based.navigation import mdp


# NOTE: keys are strings ("1"/"2"/"3"), not ints. This dict lives in an
# EventTermCfg's params, which is walked by Isaac Lab's configclass.to_dict()
# (e.g. via hydra_task_config) — that walk assumes every dict key is a str and
# crashes on an int key. UniversalRandomMapGenerator normalizes these back to
# int internally, so string keys here are required, not just a style choice.
_REFERENCE_DIFFICULTY_CONFIGS = {

    "1": {
        "count": (10, 20),
        "min_spacing": (1.0, 2.0),
        "min_goal_dist": 12.0,
        "max_goal_dist": 18.0,
    },

    "2": {
        "count": (20, 35),
        "min_spacing": 0.7,
        "min_goal_dist": 18.0,
        "max_goal_dist": 27.0,
    },

    "3": {
        "count": (35, 50),
        "min_spacing": 0.4,
        "min_goal_dist": 27.0,
        "max_goal_dist": 36.0,
    },

}

# events
@configclass
class EventsCfg:

    reset_map_and_spawn = EventTerm(
        func=mdp.reset_map_and_spawn,
        mode="reset",
        params={
            "visualize_map": False,
            "use_large_map_generator": True,
            "arena_size": 40.0,
            "resolution": 0.2,
            "difficulty_configs": _REFERENCE_DIFFICULTY_CONFIGS,

        }
    )
