import argparse
import torch

from isaaclab.app import AppLauncher

# create parser
parser = argparse.ArgumentParser()

# append AppLauncher cli args to the parser
AppLauncher.add_app_launcher_args(parser=parser)

# parse the cli args
cli_args = parser.parse_args()

# intialize the Omniverse app
app_launcher = AppLauncher(cli_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnvCfg, ManagerBasedEnv
from isaaclab.envs import mdp
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR




JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)

@configclass
class MyScene(InteractiveSceneCfg):
    """Custom Scene configuration for the Scene consiting:
    1. ground 
    2. robot
    3. cardboard boxes
    4. lights

    *Note* : Order is Important.
    """

    # ground
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg()
    )

    # robot
    Jetbot = JETBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Jetbot")    

    # light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            intensity=7000.0,
            color=(0.75, 0.75, 0.75),
        )
    )

@configclass
class ActionCfg():

    joint_action = mdp.JointVelocityActionCfg(
        asset_name="Jetbot",
        debug_vis=True,
        joint_names=[".*"],
    )

@configclass
class ObservationCfg:
    """Observation Cfg"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Provides with Policy Observation."""

        jetbot_base_line_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("Jetbot")}
        )

        jetbot_base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("Jetbot")}
        )

        def __post_init__(self) -> None:
            self.concatenate_terms = False
            self.enable_corruption = False

    policy = PolicyCfg()



@configclass
class MyCustomEnvCfg(ManagerBasedEnvCfg):
    """Implementation of manager based env."""

    scene = MyScene(num_envs=4, env_spacing=5.0)

    actions = ActionCfg()

    observations = ObservationCfg()



    def __post_init__(self):
        """Post Intialization"""
        self.sim.dt = 1.0/200.0  # simualtion @ 200Hz
        self.decimation = 4  # 200/4 = 50Hz : Env @ 50Hz
        self.viewer.eye = (2.0, 0, 2.0)
        self.viewer.lookat = (0, 0, 0)

        # no. of physics steps per render step
        # rendering @ 50Hz
        self.sim.render_interval = self.decimation



# main 
def main():
    """Main func implementation."""

    env_cfg = MyCustomEnvCfg()
    env = ManagerBasedEnv(env_cfg)

    # setup the env 
    env.reset()
    env.setup_manager_visualizers()

    # simulation ready 
    print("[INFO]: Setup Complete...")

    #simulate physics 
    count = 0
    while simulation_app.is_running():
        """reser the env after 300 steps => 0.02 * 300 """
        if count % 400 == 0:
            print("-"*80)
            print("[INFO]: Environment Resetting...") 
            print("-"*80)
            env.reset()
            count = 0

        if count % 100 <= 75:
            action = torch.tensor([[10.0, 10.0]]).to(device=env.device)
        else:
            action = torch.tensor([[-5.0, 5.0]]).to(device=env.device)

        obs, _ = env.step(action=action)

        print(f"[INFO]: Observation got : {obs}")
        count+=1

    # close the env
    env.close()


if __name__ == "__main__":

    main()
    simulation_app.close()



    