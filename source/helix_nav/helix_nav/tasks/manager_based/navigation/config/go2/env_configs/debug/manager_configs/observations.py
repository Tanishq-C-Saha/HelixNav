from isaaclab.utils import configclass

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from helix_nav.tasks.manager_based.navigation import mdp



# observations
@configclass
class ObservationsCfg:
    """Observations for the HelixNav"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations."""
        # Obs 0 : previous actions
        prev_actions = ObsTerm(
            func=mdp.last_action
        )

        # lookahead_waypoints = ObsTerm(
        #     func=mdp.
        # )

        # current velocity 
        current_lin_vel = ObsTerm(
            func=mdp.base_lin_vel
        )

        current_ang_vel = ObsTerm(
            func=mdp.base_ang_vel
        )

        # previous actions 
        prev_actions = ObsTerm(
            mdp.last_action
        )
        # TODO(cp8): migrate nav state to CommandTerm
        # relative goal vector with normalized distance
        rel_goal_w = ObsTerm(
            func=mdp.get_relative_goal_vector,
            params={
                "arena_size": 12.0,
            }
        )

        # SRU-GRU branch obs
        depth_images = ObsTerm(
            func=mdp.get_depth_images,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "normalize": True,
            }
        )

        def __post_init__(self):
            # so that we can receive dict form of obs and easy to debug
            self.concatenate_terms = False  

    policy = PolicyCfg()
