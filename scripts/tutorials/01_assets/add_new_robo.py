import argparse
import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()

AppLauncher.add_app_launcher_args(parser)

cli_args = parser.parse_args()

app_launcher = AppLauncher(cli_args)
simulation_app = app_launcher.app


"""Everything rest follows."""

from isaaclab.utils import configclass
from isaaclab.sim import SimulationContext, SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg

# Jetbot config 
JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)


# defining custom scene
@configclass 
class MyScene(InteractiveSceneCfg):
    """Defining custom scene that spawns : 
    1. ground
    2. light
    3. robot
    4. table : usd file
    """

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg()
    )

    # robot
    Jetbot = JETBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Jetbot")

    # creating XForm using dummy asset hack
    static_objects = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/StaticObjects",
        spawn=sim_utils.CuboidCfg(
            size=(0.0001, 0.0001, 0.0001),
            visible=False
        )
    )

    # cardboard 
    cardboard = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/StaticObjects/Cardboard_Box",
        spawn=sim_utils.UsdFileCfg(
            usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis/Industrial/Containers/Cardboard/Cardbox_B3.usd",
            scale=(0.01, 0.01, 0.01),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(2.0,0.0,0.0),
        )
    )


    # light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            intensity=7000.0,
            color=(0.75, 0.75, 0.75),
        )
    )


def run_simulation( sim : SimulationContext, scene : InteractiveScene):
    """Run the simulation with custom logic for time steps"""
    sim_dt = sim.cfg.dt
    counter = 0

    while simulation_app.is_running():

        # reset envs
        if counter % 500 == 0:
            # reset counter
            counter = 0

            # reset default state 

            # get default state
            robot_default_root_state = scene["Jetbot"].data.default_root_state.clone()
            robot_default_root_state[:, :3]+=scene.env_origins  # manipulate pos for multiple envs

            # write default root data to refreshed env
            scene["Jetbot"].write_root_pose_to_sim(robot_default_root_state[:, :7])
            scene["Jetbot"].write_root_velocity_to_sim(robot_default_root_state[:, 7:])

            # read default joint data 
            robot_default_joint_pos, robot_default_joint_vel = (
                scene["Jetbot"].data.default_joint_pos.clone(),
                scene["Jetbot"].data.default_joint_vel.clone()
            )

            # write default joint state 
            scene["Jetbot"].write_joint_state_to_sim(robot_default_joint_pos, robot_default_joint_vel)

            scene.reset()
            print("[INFO]: Resetting envs...")

        # for 75 counter steps
        if counter % 100 < 75:
            action = torch.tensor([[10.0, 10.0]])

        # for remaining 25 counter steps
        else:
            action = torch.tensor([[5.0, -5.0]])

        # set target
        scene["Jetbot"].set_joint_velocity_target(action)

        # write data to sim
        scene.write_data_to_sim()

        # run physics fro 1 time step
        sim.step()

        # update the scene buffers with new values produced from new phsyics
        scene.update(sim_dt)
        counter += 1


# main
def main():
    """Main funct implementation."""

    sim_cfg = SimulationCfg(dt=1.0/200.0)  # simulation running @ 200Hz
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye=(3.0, 0, 3.0),
        target=(0.0, 0.0, 0.0))

    scene_cfg = MyScene(num_envs=4, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    run_simulation(sim=sim, scene=scene)


if __name__=="__main__":
    main()

    simulation_app.close()
