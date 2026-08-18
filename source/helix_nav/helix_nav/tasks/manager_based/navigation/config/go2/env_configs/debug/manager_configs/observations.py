from isaaclab.utils import configclass

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

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

        def __post_init__(self):
            # so that we can receive dict form of obs and easy to debug
            self.concatenate_terms = False  

    policy = PolicyCfg()
