"""HelixNav debug script : keyboard-controlled Go2 with optional visualization output."""

import argparse
import os

import torch

from isaaclab.app import AppLauncher

# CLI args 
parser = argparse.ArgumentParser(description="HelixNav keyboard debug environment.")

parser.add_argument(
    "--save-camera-images",
    action="store_true",
    default=False,
    help="Save RGB + depth images every step (fills disk fast, use sparingly).",
)
parser.add_argument(
    "--save-occupancy-map",
    action="store_true",
    default=False,
    help="Save the global occupancy grid image on first reset.",
)

AppLauncher.add_app_launcher_args(parser=parser)
args, _ = parser.parse_known_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Imports that require Omniverse runtime 
from isaaclab.utils import configclass
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.sensors import MultiMeshRayCaster, ContactSensor
from isaaclab.assets import RigidObject, Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils.math import yaw_quat, euler_xyz_from_quat

from helix_nav.tasks.manager_based.navigation.config.go2.env_configs.debug.helixnav_debug_base_env_cfg import (
    HelixNavDebugBaseEnvCfg,
)
from helix_nav.tasks.manager_based.navigation.map_generators import (
    RandomMapGenerator,
    visualize_map_spec,
)

from helix_nav.tasks.manager_based.navigation.config.go2.env_configs.debug.debug_viz_utils import save_images_grid, save_occupancy_map

# Constants 
STATIC_OBSTACLE_COUNT = 13
FORCE_THRESHOLD = 1.0

@configclass
class MyKeyboardCfg(Se2KeyboardCfg):
    v_x_sensitivity: float = 1.0
    v_y_sensitivity: float = 0.5
    omega_z_sensitivity: float = 1.0


# run_simulation
def run_simulation(env: ManagerBasedEnv, keyboard_controller: Se2Keyboard):
    """Run the keyboard-controlled simulation loop."""

    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    nav_command = torch.zeros(env.num_envs, 3, device=env.device)

    collision_sensor: ContactSensor = env.scene["collision_sensor"]
    robot: Articulation = env.scene["robot"]

    while simulation_app.is_running():
        with torch.inference_mode():

            # ── Step ──
            vx, vy, yaw_rate = keyboard_controller.advance()
            nav_command[:, 0] = vx
            nav_command[:, 1] = vy
            nav_command[:, 2] = yaw_rate

            # debug for contact sensor : collision_sensor
            net_forces_w = collision_sensor.data.net_forces_w.clone()

            # print(f"[DEBUG]: Net_forces_w (N, B, 3) = {net_forces_w}")
            # print(f"[DEBUG]: body_names = {collision_sensor.body_names}")
            # print(f"[DEBUG]: num_bodies = {collision_sensor.data.net_forces_w.shape}")

            # force_norms = torch.norm(net_forces_w, dim=-1)   # (N,B)
            # collisions = torch.any(force_norms > FORCE_THRESHOLD, dim=-1)  # (N,)
            # print(f"[DEBUG]: Collisions = {collisions}")

            obs, extras = env.step(nav_command)

            # debugging obs
            print(f"\n[DENUG]: Obs = {obs}\n\n")

            # debugging goals 
            print(f"[DEBUG]: goal_position_w = {env._goal_positions + env.scene.env_origins}")
            print(f"[DEBUG]: robot_pose_w = {robot.data.root_pose_w}")
            print(f"[DEBUG]: robot_yaw_world = {euler_xyz_from_quat(robot.data.root_quat_w)}")


            # Optional: save camera images 
            if args.save_camera_images:
                rgb_images = env.scene["rgb_camera"].data.output["rgb"]

                for i in range(env.num_envs):
                    save_images_grid(
                        images=[rgb_images[i], depth_images[i]],
                        subtitles=["RGB", "Depth"],
                        cmap="turbo",
                        title="MultiMeshRayCasterCamera on Unitree Go2 (obs)",
                        filename=os.path.join(
                            output_dir, "camera_frames", f"env_{i}", f"{env._sim_step_counter:04d}.jpg",
                        ),
                    )
                

# main
def main():
    env_cfg = HelixNavDebugBaseEnvCfg()
    env = ManagerBasedEnv(env_cfg)

    env.reset()
    env.setup_manager_visualizers()

    keyboard_controller = Se2Keyboard(cfg=MyKeyboardCfg)
    keyboard_controller.reset()

    print("[INFO]: Setup complete.")
    env.reset()
    run_simulation(env=env, keyboard_controller=keyboard_controller)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()