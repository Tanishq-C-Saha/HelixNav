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
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.sensors import MultiMeshRayCaster
from isaaclab.assets import RigidObject, AssetBase

from isaaclab.envs import ManagerBasedEnv
from helix_nav.tasks.manager_based.navigation.config.go2.env_configs.debug.helixnav_debug_base_env import (
    HelixNavDebugBaseEnvCfg,
)
from helix_nav.tasks.manager_based.navigation.map_generators import (
    RandomMapGenerator,
    visualize_map_spec
)

from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.sim import get_current_stage
import omni

import os
import matplotlib.pyplot as plt
import numpy as np

# used for Static Obstacle count
STATIC_OBSTACLE_COUNT = 13
OBSTACLE_POOL: list[tuple[float, float, float]] = []


# keyboard device cfg
@configclass
class MyKeyboardCfg(Se2KeyboardCfg):
    """Keyboard configs"""

    v_x_sensitivity: float = 1
    v_y_sensitivity: float = 0.5
    omega_z_sensitivity: float = 1.0


# helper func
def save_images_grid(
    images: list[torch.Tensor],
    cmap: str | None = "viridis",
    nrow: int = 1,
    subtitles: list[str] | None = None,
    title: str | None = None,
    filename: str | None = None,
):
    """Save images in a clean, professional grid."""

    n_images = len(images)
    ncol = int(np.ceil(n_images / nrow))

    # Figure size
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(4.0 * ncol, 3.2 * nrow),
        squeeze=False,
    )

    axes = axes.flatten()

    for idx, (img, ax) in enumerate(zip(images, axes)):

        # Tensor -> NumPy
        img = img.detach().cpu().numpy()

        # Remove unnecessary channel dimension
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]

        # Show image
        ax.imshow(
            img,
            cmap=cmap,
            interpolation="nearest",
            aspect="equal",
        )

        # Remove ticks, but KEEP the frame
        ax.set_xticks([])
        ax.set_yticks([])

        # Professional image boundary
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color("0.25")

        # Subtitle
        if subtitles and idx < len(subtitles):
            ax.set_title(
                subtitles[idx],
                fontsize=11,
                fontweight="medium",
                pad=8,
            )

    # Remove unused axes
    for ax in axes[n_images:]:
        fig.delaxes(ax)

    # Main title
    if title:
        fig.suptitle(
            title,
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

    # Spacing
    fig.tight_layout(rect=(0, 0, 1, 0.94 if title else 1))

    # Save
    if filename:
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)

        fig.savefig(
            filename,
            dpi=200,
            bbox_inches="tight",
            pad_inches=0.15,
            facecolor="white",
        )

    plt.close(fig)


# func to store global occupancy map
def store_global_occupancy_map(
    hit_points_w: torch.Tensor,
    min_threshold: float,
    max_threshold: float,
):
    """
    Sensor
        │
        ├── 3721 rays = 61 x 61
        │
        └── each ray → XYZ hit point
                    │
                    ▼
              Extract Z values
                    │
                    ▼
              Hit thresholding
                    │
                    ▼
              occupancy = 0 / 1
                    │
                    ▼
            reshape: 3721 → 61 x 61
                    │
                    ▼
             detach → CPU → NumPy
                    │
                    ▼
             Matplotlib visualization
                    │
                    ▼
                  save
    """

    # Extract ray hit points
    points = hit_points_w[0]
    points_x = points[:, 0]  # world_x
    points_y = points[:, 1]  # world_y
    points_z = points[:, 2]

    # Determine occupied rays
    hit_mask = (points_z > min_threshold) & (points_z < max_threshold)

    occupancy = torch.zeros_like(points_z)
    occupancy[hit_mask] = 1

    # Convert to NumPy
    occupancy_grid = occupancy.detach().cpu().numpy().reshape(61, 61)

    # indices = occupancy_grid.nonzero()  # gives you (row, col) pairs
    # print(f"[DEBUG]: Obstacle at world (3.0, 1.0) appears at grid indices: {indices}")

    # Professional visualization
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(
        occupancy_grid,
        origin="lower",
        extent=[-6, 6, -6, 6],
        interpolation="nearest",
        cmap="Greys",
        vmin=0,
        vmax=1,
    )

    # Major grid: every 1 m
    ax.set_xticks(np.arange(-6, 6.1, 1.0))
    ax.set_yticks(np.arange(-6, 6.1, 1.0))

    # Minor grid: every 0.2 m
    ax.set_xticks(np.arange(-6, 6.01, 0.2), minor=True)
    ax.set_yticks(np.arange(-6, 6.01, 0.2), minor=True)

    # Major grid
    ax.grid(
        which="major",
        linewidth=0.8,
        alpha=0.35,
    )

    # Minor grid = individual ray/grid cells
    ax.grid(
        which="minor",
        linewidth=0.25,
        alpha=0.25,
    )

    # Sensor
    ax.scatter(
        0,
        0,
        marker="x",
        s=100,
        linewidths=2,
        label="Sensor",
    )

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title("Occupancy Grid — MultiMeshRayCaster")

    ax.set_aspect("equal")
    ax.legend(loc="upper right")

    fig.tight_layout()

    fig.savefig(
        "occupancy.jpg",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# helper function to get child prim names 
def get_static_obstacles(env: ManagerBasedEnv):
    """
    Return all obstacle child prims under the static_obstacles parent.
    """

    stage = get_current_stage()

    parent_path = f"/World/envs/env_0/static_obstacles"
    parent_prim = stage.GetPrimAtPath(parent_path)

    if not parent_prim.IsValid():
        raise RuntimeError(
            f"Could not find prim at {parent_path}"
        )

    obstacles = []

    for child in parent_prim.GetChildren():
        print(f"[DEBUG]: child: {child}")
        print(f"[DEBUG]: child_name : {child.GetName()}")
        if child.GetName().startswith("s_obs_"):
            obstacles.append(child)

    return obstacles

# run simulation
def run_simulation(env: ManagerBasedEnv, keyboard_controller: Se2Keyboard):
    """Running the simulation."""

    low_level_om = env.action_manager._terms["velocity_commands"]._low_level_obs_manager

    nav_command = torch.zeros(env.num_envs, 3, device=env.device)

    rgb_camera = env.scene["rgb_camera"]
    depth_camera = env.scene["depth_camera"]
    occupancy_scanner: MultiMeshRayCaster = env.scene["occupancy_scanner"]


    static_obstacles: list[RigidObject] = []
    for i in range(STATIC_OBSTACLE_COUNT):
        obs = env.scene[f"s_obs_{i}"]
        static_obstacles.append(obs)

    # # test obs for grid emperical mapping
    # test_obs: RigidObject = env.scene["test_obs"]

    # Create output directory to save images
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    first_run = False  # flag to gate the reset

    while simulation_app.is_running():
        with torch.inference_mode():
            """Actions from the keyboard_copntroller"""

            if first_run == False:

                # debug purpose
                obstacles = get_static_obstacles(env=env)
                print(f"[DEBUG_script]: osbtacles = {obstacles}")

                for obs in obstacles:
                    print(obs.GetName(), obs.GetPath())


                # getting specs of obstacles 
                # printing size and position of static obstacles
                for obs in static_obstacles:
                    print(
                        f"\n\n[DEBUG]: prim_path = {obs.cfg.prim_path}\n"
                        f"[DEBUG]: pos = {obs.data.root_com_pos_w}\n"
                    )

                    if hasattr(obs.cfg.spawn, "size"):
                        print(f"[DEBUG]: size = {obs.cfg.spawn.size}\n\n")
                        OBSTACLE_POOL.append(obs.cfg.spawn.size)
                    elif hasattr(obs.cfg.spawn, "radius"):
                        print(
                            f"[DEBUG]: radius = {obs.cfg.spawn.radius}\n"
                            f"[DEBUG : height = {obs.cfg.spawn.height}\n\n"
                        )
                        size = (
                            obs.cfg.spawn.radius,
                            obs.cfg.spawn.radius,
                            obs.cfg.spawn.height,
                        )
                        OBSTACLE_POOL.append(size)

                print(f"[DEBUG]: Obstacle pool = {OBSTACLE_POOL}\n")


                # random map generator
                map_generator = RandomMapGenerator(obstacle_pool=OBSTACLE_POOL)
                map_spec = map_generator.generate_with_retry(
                    difficulty=3, seed=42, max_attempts=20
                )
                for obs_spec in map_spec.obstacles:
                    """Writing pos to obstacles."""
                    obs = static_obstacles[obs_spec.pool_index]
                    obs_pose = torch.zeros_like(obs.data.root_com_pose_w)
                    obs_pose[:, 0] = obs_spec.position[0]
                    obs_pose[:, 1] = obs_spec.position[1]
                    obs_pose[:, 2] = obs_spec.position[2]
                    obs.write_root_com_pose_to_sim(obs_pose)


                # generating occupancy map 

                occupancy_scanner_pos_w = occupancy_scanner.data.pos_w.clone()
                ray_hit_points_w = occupancy_scanner.data.ray_hits_w.clone()

                print("-------------------------------")
                print(f"{occupancy_scanner}")
                print(
                    f"[DEBUG]: occupancy_scanner world pos = {occupancy_scanner_pos_w}\n"
                    f"[DEBUG]: ocuupancy grid shape = {ray_hit_points_w.shape}\n\n"
                )

                store_global_occupancy_map(
                    hit_points_w=ray_hit_points_w, min_threshold=0.01, max_threshold=6
                )

                visualize_map_spec(map_spec=map_spec,
                                save_path=os.path.join(output_dir, "generated_maps", "map_seed_42.jpg"),
                                dpi=200,)

                first_run = True  # first run done

            vx, vy, yaw_rate = keyboard_controller.advance()
            nav_command[:, 0] = vx
            nav_command[:, 1] = vy
            nav_command[:, 2] = yaw_rate
            print(low_level_om.active_terms)
            print(low_level_om.group_obs_dim)
            obs, extras = env.step(nav_command)

            print(f"[DEBUG]: Obs : {obs}")

            print(f"[DEBUG]: Extras : {extras}\n\n")

            rgb_image = rgb_camera.data.output["rgb"]
            depth_image = depth_camera.data.output["distance_to_camera"]

            print("-------------------------------")
            print(f"{rgb_camera}")
            print(
                "Received shape of RGB image: ",
                rgb_camera.data.output["rgb"].shape,
                "\n\n",
            )

            print("-------------------------------")
            print(f"{depth_camera}")
            print(
                "Received shape of depth: ",
                depth_camera.data.output["distance_to_camera"].shape,
                "\n\n",
            )

            save_images_grid(
                images=[rgb_image[0], depth_image[0]],
                subtitles=["rgb image ", "depth image"],
                cmap="turbo",
                title="MultiMeshRayCasterCamera on Unitree Go2.",
                filename=os.path.join(
                    output_dir,
                    "Keyboard_controller" "MultiMeshRayCasterCamera",
                    "distance_to_camera",
                    f"{env._sim_step_counter:04d}.jpg",
                ),
            )


# main
def main():
    """Main func implementation."""

    env_cfg = HelixNavDebugBaseEnvCfg()
    env = ManagerBasedEnv(env_cfg)

    # setup the env
    env.reset()
    env.setup_manager_visualizers()

    keyboard_controller = Se2Keyboard(cfg=MyKeyboardCfg)
    keyboard_controller.reset()

    # simulation ready
    print("[INFO]: Setup Complete...")

    # simulate physics
    run_simulation(env=env, keyboard_controller=keyboard_controller)

    # close the env
    env.close()


if __name__ == "__main__":

    main()
    simulation_app.close()
