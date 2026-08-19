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

from isaaclab.envs import ManagerBasedEnv
from source.helix_nav.helix_nav.tasks.manager_based.navigation.config.go2.env_configs.debug.helixnav_debug_base_env_cfg import HelixNavDebugBaseEnvCfg


# main 
def main():
    """Main func implementation."""

    env_cfg = HelixNavDebugBaseEnvCfg()
    env = ManagerBasedEnv(env_cfg)

    # setup the env
    env.reset()
    env.setup_manager_visualizers()

    low_level_om = env.action_manager._terms["velocity_commands"]._low_level_obs_manager


    # simulation ready 
    print("[INFO]: Setup Complete...")

    #simulate physics 
    count = 0
    while simulation_app.is_running():
        """reser the env after 300 steps => 0.02 * 300 """
        if count % 100 == 0:
            env._current_difficulty += 1 
            env._current_difficulty = min(3, env._current_difficulty)
            
            count = 0
            print(f"\n\n[DEBUG] Resetting env {'*' * 100}\n\n")
            env.reset()

        if count % 1000 <= 600:
            nav_command = torch.tensor([[1.0, 0.0, 0.0]], device=env.device)
        else:
            nav_command = torch.tensor([[0.1, 0.0, 0.0]], device=env.device)  # [vx, vy, yaw_rate] 

        # print(low_level_om.active_terms)
        # print(low_level_om.group_obs_dim)
        obs, extras = env.step(nav_command)
        # print(f"[DEBUG]: Obs : {obs}")

        # print(f"[DEBUG]: Extras : {extras}\n\n")

        print(f"[DEBUG]: count = {count}")
        count += 1

    # close the env
    env.close()


if __name__ == "__main__":

    main()
    simulation_app.close()



    