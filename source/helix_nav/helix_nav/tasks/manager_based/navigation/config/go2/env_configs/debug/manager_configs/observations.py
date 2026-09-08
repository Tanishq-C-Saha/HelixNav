from isaaclab.utils import configclass

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from helix_nav.tasks.manager_based.navigation import mdp


# observations
@configclass
class ObservationsCfg:
    """Observations for the HelixNav
    policy
        ├── prev_actions        → (N, 3)
        ├── lookahead_vectors   → (N, 24)
        ├── snap_flags          → (N, 8)
        ├── base_velocity       → (N, 3)
        ├── relative_goal       → (N, 3)
        └── depth_images        → (N, 1, 54, 96)

        Scalar: 3+24+8+3+3=41
        Depth: 54x96
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations."""
        # Obs 0 : previous actions
        prev_actions = ObsTerm(
            func=mdp.last_action
        )

        lookahead_vectors = ObsTerm(
            func=mdp.get_lookahead_vectors,
            params={
                "command_name": "navigation_goal",
            },
        )

        snap_flags = ObsTerm(
            func=mdp.get_snap_flags,
            params={
                "command_name": "navigation_goal",
            },
        )

        base_velocity = ObsTerm(
            func=mdp.get_base_velocity,
        )

        # relative goal vector with normalized distance
        relative_goal = ObsTerm(
            func=mdp.get_relative_goal_vector,
            params={
                "command_name": "navigation_goal",
                "asset_cfg": SceneEntityCfg("robot"),
                "arena_size": 12.0,
            },
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
            # Keep observations as a dictionary so the depth branch and
            # vector branch can be handled separately later.
            self.concatenate_terms = False  

    policy = PolicyCfg()
