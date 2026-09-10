from pathlib import Path

from isaaclab.utils import configclass

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise


from helix_nav.tasks.manager_based.navigation import mdp


# path resolving for low level policy
_GO2_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent
_LOCO_PT = _GO2_CONFIG_DIR / "policies" / "locomotion" / "policy.pt"

unitree_go2_flat_env = UnitreeGo2FlatEnvCfg()
unitree_go2_flat_env.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )


@configclass
class ActionsCfg:
    """Pretrained Loco policy as action."""

    velocity_commands = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=str(_LOCO_PT),
        low_level_decimation=4,  # 50hz
        low_level_observations=unitree_go2_flat_env.observations.policy,
        low_level_actions=unitree_go2_flat_env.actions.joint_pos,
        debug_vis=True
    )
