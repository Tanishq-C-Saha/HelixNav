from isaaclab.utils import configclass

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from helix_nav.tasks.manager_based.navigation import mdp


# observations
@configclass
class ObservationsCfg:
    """Observations for the HelixNav policy.

    concatenate_terms=True → a single flat (N, 5225) tensor. The slice order
    is this field declaration order — do not reorder without updating the
    matching slice offsets in skrl_policy.py.

        depth_images:       ObsTerm   # (N, 5184)  flattened from (54, 96, 1) by get_flatten_depth_images
        lookahead_vectors:  ObsTerm   # (N, 24)
        snap_flags:         ObsTerm   # (N, 8)
        base_velocity:      ObsTerm   # (N, 3)
        prev_actions:       ObsTerm   # (N, 3)
        relative_goal:      ObsTerm   # (N, 3)

        Scalar total: 24+8+3+3+3 = 41
        Depth: 54*96 = 5184
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations."""

        # obs0: SRU-GRU branch obs
        # flattened to (N, 5184) by get_flatten_depth_images so it can be
        # concatenated with the 1-D scalar terms below (concatenate_terms=True
        # requires every term in the group to have the same number of dimensions)
        depth_images = ObsTerm(
            func=mdp.get_flatten_depth_images,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "normalize": True,
            },
        )

        # obs1 : lookahead_vectors
        lookahead_vectors = ObsTerm(
            func=mdp.get_lookahead_vectors,
            params={
                "command_name": "navigation_goal",
            },
        )

        # obs2 : snap_flags
        snap_flags = ObsTerm(
            func=mdp.get_snap_flags,
            params={
                "command_name": "navigation_goal",
            },
        )

        # obs3 : base_velocity
        base_velocity = ObsTerm(
            func=mdp.get_base_velocity,
        )

        # obs4 : prev_actions    
        prev_actions = ObsTerm(
            func=mdp.last_action
        )


        # obs5: relative goal vector with normalized distance
        relative_goal = ObsTerm(
            func=mdp.get_relative_goal_vector,
            params={
                "command_name": "navigation_goal",
                "asset_cfg": SceneEntityCfg("robot"),
                "arena_size": 12.0,
            },
        )



        def __post_init__(self):
            # flat (N, 5225) tensor, sliced by field order in skrl_policy.py
            self.concatenate_terms = True

    policy = PolicyCfg()
