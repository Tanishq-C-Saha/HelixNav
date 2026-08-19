"""base nav"""


from isaaclab.utils import configclass

from isaaclab.scene import InteractiveSceneCfg


from isaaclab.envs import ManagerBasedEnvCfg

from helix_nav.tasks.manager_based.navigation.config.go2.env_configs.debug.manager_configs import (
    HelixNavDebugBaseScene,
    ObservationsCfg,
    ActionsCfg,
    EventsCfg
)

@configclass
class HelixNavDebugBaseEnvCfg(ManagerBasedEnvCfg):
    """Definition of custom env."""
    scene: InteractiveSceneCfg = HelixNavDebugBaseScene()

    observations = ObservationsCfg()

    actions = ActionsCfg()

    events = EventsCfg()

    def __post_init__(self):
        self.sim.dt = 1.0/200.0  # simulation @ 200Hz
        self.decimation = 20   # HL env running @ 10Hz
        self.scene.env_spacing = 12.2
        self.scene.num_envs = 2  # jsut debug purpose
        self.sim.render_interval = 4  # simulation rendering @50Hz



