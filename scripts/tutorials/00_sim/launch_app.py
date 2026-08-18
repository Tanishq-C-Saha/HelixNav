import argparse
import time

from isaaclab.app import AppLauncher


# create arg parser
parser = argparse.ArgumentParser()

# add AppLauncher args to the parser
AppLauncher.add_app_launcher_args(parser=parser)

parser.add_argument(
    "--size",
    type=float,
    help="Size of cuboid to spawn on the stage!",
    required=True,
)

# read the cli args
cli_args = parser.parse_args()

app_launcher = AppLauncher(cli_args)
simulation_app = app_launcher.app

from isaaclab.sim import SimulationCfg, SimulationContext
import isaaclab.sim as sim_utils


# design the scene with : Ground plane, Light, Cuboid
def design_scene():
    """Design of teh Isaac sim scene : 
    1. Ground plane 
    2. Dome Light 
    3. Cuboid
    """

    # spawn ground plane
    ground_plane_cfg = sim_utils.GroundPlaneCfg()
    ground_plane_cfg.func(prim_path="/World/ground", cfg=ground_plane_cfg)

    # spawn distant light
    light_cfg = sim_utils.DistantLightCfg(
        intensity=3000,
        color=(0.75,0.75,0.75),
        )
    light_cfg.func(prim_path="/World/lightDistant",cfg=light_cfg)

    # spawn cuboid
    cuboid_cfg = sim_utils.CuboidCfg(
        size=[cli_args.size]*3,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0))
    )
    cuboid_cfg.func(prim_path="/World/cuboid", cfg=cuboid_cfg, translation=(0, 0, cli_args.size/2.0))


# main
def main():
    """Main funct implementation."""
    print("[INFO]: Starting setup...")
    start_time = time.perf_counter()  # s

    # create simulation config
    sim_cfg = SimulationCfg(dt=1.0/200.0)  # simulation @ 200Hz
    sim = SimulationContext(cfg=sim_cfg)

    # set the main camera
    sim.set_camera_view(
        eye=(2.0, 0.0, 2.5),
        target=(-0.5, 0.0, 0.0)
    )

    # setup the scene
    design_scene()

    # play simulator
    sim.reset()

    # now the simulator is ready
    stop_time = time.perf_counter()  # s
    total_time = (stop_time - start_time)*1000  # ms
    print("[INFO]: Setup Complete...")
    print(f"[INFO]: {total_time}ms took for startup!")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()

    # close the simulation app
    simulation_app.close()



