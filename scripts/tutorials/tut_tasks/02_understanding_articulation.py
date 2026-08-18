"""
Two cart-poles spawned via one Articulation handle (regex path, same trick
as the cones),randomly jittered on reset, driven by random joint efforts
continuously, forever.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()

AppLauncher.add_app_launcher_args(parser=parser)

cli_args = parser.parse_args()


# starting the omniverse app with cli_args context
app_launcher = AppLauncher(cli_args)
simulation_app = app_launcher.app


"""Rest everything follows."""


from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.assets import AssetBaseCfg, AssetBase
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG
import isaaclab.sim as sim_utils

import torch



# design scene
def design_scene(num_origins: int = 2, origin_spacing: int = 4):
    """Creation of custom scene by spawning :
    1. ground
    2. lights
    3. 2 origins
    4. 2 cartpoles
    """

    # ground
    ground_cfg =sim_utils.GroundPlaneCfg()
    ground_cfg.func(
        prim_path="/World/ground",
        cfg=ground_cfg
    )


    # light
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func(
        prim_path="/World/light",
        cfg=light_cfg
    )

    # Origins
    origins = list()
    for i in range(num_origins):
        origin = (0.0+i*origin_spacing, 0.0, 0.0)  # origin 4m apart
        sim_utils.create_prim(
            prim_type="Xform",
            prim_path=f"/World/origin_{i}",
            position=origin
        )

        origins.append(origin)

        """
        Wether to use position or translation?

        Practical rule going forward:
        1. Spawning something directly under /World (or any identity-transform
           parent)
           → translation and position are interchangeable, use whichever reads
            clearer.
        2. Spawning something under a parent that has its own nontrivial transform
           (rotation especially)
           → use position if you're thinking in world-frame terms
            ("I want this at world (x,y,z) regardless of parent orientation"),
            use translation if you're deliberately thinking in the parent's
            local frame ("offset this relative to wherever the parent
            is/however it's rotated").
        """

    # Articulation
    cartpole_cfg = CARTPOLE_CFG.copy()
    cartpole_cfg.prim_path = "/World/origin_*/cartpole"
    cartpole = Articulation(cfg=cartpole_cfg)

    # preprocess data
    origins = torch.tensor(origins).to(device="cuda:0")

    return {
        "cartpole": cartpole,
        "origins": origins,
    }




# run simulation 
def run_simulation(sim:SimulationContext, robot:Articulation, origins:torch.tensor):
    """Running the simulation."""

    sim_step_count = 0  # 0.005s
    random_action = torch.zeros(size=[2,1],dtype=torch.float, device="cuda:0")
    while simulation_app.is_running():

        with torch.inference_mode():

            # reset
            if sim_step_count % 1000 == 0:
                sim_step_count = 0
                sim.reset()

                robot_default_root_state = robot.data.default_root_state.clone()

                # setting default origin state
                robot_default_root_state[:, :3] += origins

                robot.write_root_state_to_sim(root_state=robot_default_root_state)
                robot.reset()
                # debug 
                print(f"[DEBUG]: robot_default_root_state = {robot_default_root_state}")

            # cartpole controller @ 50Hz
            if sim_step_count % 4 == 0:
                # random action
                print(f"[DEBUG]: velocity of robot : {robot.data.joint_vel}")
                action = torch.rand_like(random_action, dtype=torch.float, device="cuda:0")
                robot.set_joint_velocity_target(target=action, joint_ids=[0])
                robot.write_data_to_sim()

            sim.step()
            robot.update(dt=sim.cfg.dt)
            sim_step_count += 1

    sim.close()


# main
def main():
    """Main func implementation:
    1. simulation cfg
    2. simulation context
    3. design scene
    4. run simulation
    5. close simulation app
    """

    # simulation running @200Hz
    sim_cfg = SimulationCfg(dt=1.0/200.0, render_interval=4)  
    sim = SimulationContext(cfg=sim_cfg)

    items = design_scene()
    cartpole = items["cartpole"]
    origins = items["origins"]

    sim.reset()  # intialize the PhsyX buffers and USD stage

    # simulation setup ready
    print("[INFO]: Setup complete...")

    # debug info 
    print(f"[DEBUG]: cartpole = {cartpole}\n"
          f"[DEBUG]: no. of cartpoles instances = {cartpole.num_instances}")

    print(f"[DEBUG]: origins = {origins}")

    run_simulation(sim=sim, robot=cartpole, origins=origins)



if __name__ == "__main__":
    main()

    simulation_app.close()
