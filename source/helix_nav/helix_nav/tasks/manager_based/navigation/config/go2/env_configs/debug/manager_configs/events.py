from isaaclab.utils import configclass

from isaaclab.managers import EventTermCfg as EventTerm

from helix_nav.tasks.manager_based.navigation import mdp

# events
@configclass
class EventsCfg:

    reset_map_and_spawn = EventTerm(
        func=mdp.reset_map_and_spawn,
        mode="reset",
        params={
            "visualize_map": True
        }
    )
