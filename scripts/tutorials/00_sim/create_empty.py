import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser()

# append app launcher cli args
AppLauncher.add_app_launcher_args(parser=parser)

# parse the args 
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.sim import SimulationCfg, SimulationContext


def main():
    """Main Function"""

    # Intialize the simulation context
    sim_cfg = SimulationCfg(dt=1.0/200.0)
    sim = SimulationContext(sim_cfg)

    # set the main camera view
    sim.set_camera_view(eye=(2.5, 2.5, 2.5), target=(0.0, 0.0, 0.0))

    # play the simulator
    sim.reset()

    # Now we are ready!
    print(f"[INFO]: Setup complete...")

    #simulate physics 
    while simulation_app.is_running():
        # perform step
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
