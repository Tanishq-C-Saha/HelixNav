

"""CP7 RL environment : PBRS reward, Scalar + Depth branch obs, curriculum randomisation."""

from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import ManagerBasedRLEnvCfg

from helix_nav.tasks.manager_based.navigation.config.go2.env_configs.cp7.manager_configs import (
    ObservationsCfg,
    ActionsCfg,
    RewardCfg,
    TerminationsCfg,
    CommandsCfg,
    CurriculumsCfg,
)

from helix_nav.tasks.manager_based.navigation.config.go2.env_configs.test.test_scenes.scene_1 import (
    HelixNavTestRLScene
)

from .events import EventsCfg


@configclass
class HelixNavCP7TestRLEnvCfg(ManagerBasedRLEnvCfg):
    """CP7 navigation environment with PBRS reward and curriculum."""

    scene: InteractiveSceneCfg = HelixNavTestRLScene()

    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventsCfg()
    rewards = RewardCfg()
    terminations = TerminationsCfg()
    commands = CommandsCfg()
    curriculum = CurriculumsCfg()

    def __post_init__(self):
        self.sim.dt = 1.0 / 200.0        # physics @ 200 Hz
        self.decimation = 20              # HL policy @ 10 Hz
        self.scene.env_spacing = 50
        self.scene.num_envs = 2048        # CP7 training scale
        self.sim.render_interval = 4      # render @ 50 Hz
        self.seed = 42
        self.episode_length_s = 60



@configclass
class HelixNavCP7TestRLEnvPlayCfg(HelixNavCP7TestRLEnvCfg):
    """CP7 navigation environment with PBRS reward and curriculum."""

    def __post_init__(self):
        self.sim.dt = 1.0 / 200.0        # physics @ 200 Hz
        self.decimation = 20              # HL policy @ 10 Hz
        self.scene.env_spacing = 50
        self.scene.num_envs = 1       # CP7 Play scale
        self.sim.render_interval = 4      # render @ 50 Hz
        self.seed = 42
        self.episode_length_s = 60
