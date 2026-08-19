
from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg

from helix_nav.tasks.manager_based.navigation import mdp


@configclass
class TerminationsCfg:
    """Termination Cfg for the HelixNav RL Env."""

    # collisions
    collisions = DoneTerm(
        func=mdp.terminations_by_collisions,
        params={
            "sensor_cfg": SceneEntityCfg("collision_sensor")
        },
        time_out=False,
    )

    # goal reached
    goal_reached = DoneTerm(
        func=mdp.goal_reached,
        params={
           "min_distance_threshold": 0.3,
        },
        time_out=False
    )

    # time out
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True
    )
