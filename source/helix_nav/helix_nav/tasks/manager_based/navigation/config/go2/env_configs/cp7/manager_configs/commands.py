from isaaclab.utils import configclass
from helix_nav.tasks.manager_based.navigation import mdp


@configclass
class CommandsCfg:
    """Command terms configuration."""

    navigation_goal = mdp.NavigationWaypointCommandCfg(
        asset_name="robot",

        # effectively infinite - never resample mid-episode
        resampling_time_range=(1.0e6, 1.0e6,),
        debug_vis=False,
    )
