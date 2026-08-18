"""Spawning rgb and depth cam in the randomized map"""

"""Spawning randomized obstacles every reset."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()

AppLauncher.add_app_launcher_args(parser=parser)

cli_args = parser.parse_args()

# starting omniverse app with context
app_launcher = AppLauncher(cli_args)
simulation_app = app_launcher.app


"""Rest everything follows"""

from isaaclab.utils import configclass
from dataclasses import MISSING

from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, ArticulationCfg, AssetBase
from isaaclab.assets import RigidObject
from isaaclab.sensors import MultiMeshRayCasterCameraCfg, CameraCfg, patterns

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg

import isaaclab.sim as sim_utils

import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# randomized obstacles locations
obs_positions = torch.empty([5, 3], device="cuda:0").uniform_(-5, 5)
obs_positions[:, -1] = 0.25
obs_positions = obs_positions.tolist()


# definition of interactive scene
@configclass
class CustomScene(InteractiveSceneCfg):
    """Custom scene definition :
    1. ground
    2. robot
    3. radomized static obstacles
    4. randomized dynamic obstacles
    5. light

    """

    # ground
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())

    # centre marker cuboid
    # centroid = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/centroid",
    #     spawn=sim_utils.CuboidCfg(
    #         # long pillar to visualize the centre , especially at the tiem of env spacing
    #         size=(0.1, 0.1, 2.5),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             disable_gravity=True,
    #             kinematic_enabled=True,  # so that it remains stationary
    #         ),
    #         visual_material=sim_utils.PreviewSurfaceCfg(
    #             diffuse_color=(0.7, 0.0, 0.5),
    #         ),
    #     ),
    # )

    # robot
    # robot: ArticulationCfg = MISSING

    # creating wall boundary
    wall_north = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/wall_north",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 12.0, 2.5),
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,  # so that it remains stationary
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.5, 0.5),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(6.0, 0.0, 1.25),
        ),
    )

    wall_south = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/wall_south",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 12.0, 2.5),
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,  # so that it remains stationary
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.5, 0.5),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-6.0, 0.0, 1.25),
        ),
    )

    wall_east = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/wall_east",
        spawn=sim_utils.CuboidCfg(
            size=(12.1, 0.1, 2.5),
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,  # so that it remains stationary
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.5, 0.5),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -6.0, 1.25),
        ),
    )

    wall_west = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/wall_west",
        spawn=sim_utils.CuboidCfg(
            size=(
                12.1,
                0.1,
                2.5,
            ),  # 12.1 to make perfect sqaure else cut squares present
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,  # so that it remains stationary
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.5, 0.5),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 6.0, 1.25),
        ),
    )

    # movable sensor base
    # Sensor base
    sensor_base = RigidObjectCfg(
        prim_path="/World/sensor_base",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.0, 0.3)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-2.0, -2.0, 0.7),
        ),
    )

    # sensor

    # rgb camera
    rgb_camera = CameraCfg(
        prim_path="/World/sensor_base/rgb_camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=45.55,  # 45.55mm ->  HFoV : 87 degrees
            vertical_aperture=26.6068,  # 26.6068mm -> VFoV : 58 degrees
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.1, 0, 0.01), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"
        ),
    )

    # MultiMeshRaycasterCamera
    camera = MultiMeshRayCasterCameraCfg(
        prim_path="/World/sensor_base",
        update_period=0.02,
        debug_vis=False,
        ray_alignment="base",
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,  # 24mmwidth=80, height=80
            horizontal_aperture=45.55,  # 45.55mm ->  HFoV : 87 degrees
            vertical_aperture=26.6068,  # 26.6068mm -> VFoV : 58 degrees
            height=51,  # 51 px
            width=90,  # 90 px
        ),
        # Only this cube is visible to the ray caster
        mesh_prim_paths=[
            "/World/ground",
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/static_obstacles/obs_0",
                track_mesh_transforms=True,
            ),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/static_obstacles/obs_1",
                track_mesh_transforms=True,
            ),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/static_obstacles/obs_2",
                track_mesh_transforms=True,
            ),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/static_obstacles/obs_3",
                track_mesh_transforms=True,
            ),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/static_obstacles/obs_4",
                track_mesh_transforms=True,
            ),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/dynamic_obstacles/obs_0",
                track_mesh_transforms=True,
            ),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/dynamic_obstacles/obs_1",
                track_mesh_transforms=True,
            ),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/wall_east",
                track_mesh_transforms=True,
            ),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/wall_west",
                track_mesh_transforms=True,
            ),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/wall_north",
                track_mesh_transforms=True,
            ),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/wall_south",
                track_mesh_transforms=True,
            ),
        ],
        max_distance=5,
        data_types=["distance_to_camera"],
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=(0.1, 0, 0.01),
            # rot=(0.7071068, 0, 0.7071068, 0),  # 90 degres in y
            rot=(0.5, -0.5, 0.5, -0.5),  # from animal camera setup
        ),
    )

    # static obstacles

    # creating dummy parent prim
    static_obstacles = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/static_obstacles",
        spawn=sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01), visible=False),
    )

    # static obstacle 1
    s_obs_0 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/static_obstacles/obs_0",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.6, 0.5),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,  # so that it remains stationary
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.8, 0.3),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(obs_positions[0]),
            # rot=()
        ),
    )

    s_obs_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/static_obstacles/obs_1",
        spawn=sim_utils.CuboidCfg(
            size=(0.7, 0.3, 0.5),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,  # so that it remains stationary
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.8, 0.3),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(obs_positions[1]),
            # rot=()
        ),
    )

    s_obs_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/static_obstacles/obs_2",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.5, 0.5),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,  # so that it remains stationary
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.8, 0.3),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(obs_positions[2]),
            # rot=()
        ),
    )

    s_obs_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/static_obstacles/obs_3",
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 0.8, 0.5),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,  # so that it remains stationary
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.8, 0.3),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(obs_positions[3]),
            # rot=()
        ),
    )

    s_obs_4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/static_obstacles/obs_4",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 0.5),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,  # so that it remains stationary
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.8, 0.3),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(obs_positions[4]),
            # rot=()
        ),
    )

    # dynamic obstacle movements

    # creating a dummy parent for randomized obstacles
    dynamic_obstacles = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/dynamic_obstacles",
        spawn=sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01), visible=False),
    )

    d_obs_0 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/dynamic_obstacles/obs_0",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.7, 0.5),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25)),
    )

    d_obs_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/dynamic_obstacles/obs_1",
        spawn=sim_utils.CuboidCfg(
            size=(0.9, 0.3, 0.9),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.3, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-3.0, 0.0, 0.45)),
    )

    # light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


# heper func
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


# run simulation
def run_simulation(
    sim: SimulationContext, scene: InteractiveScene, keyboard_controller: Se2Keyboard
):
    """Running the simulation."""
    s_obs: list[RigidObject] = [
        scene[f"s_obs_{i}"] for i in range(5)  #  InteractiveScene doesnt allow wildcard
    ]
    # centroid: AssetBase = scene["centroid"]
    count = 0

    d_obs: list[RigidObject] = [
        scene["d_obs_0"],
        scene["d_obs_1"],
    ]

    camera = scene["camera"]
    rgb_camera = scene["rgb_camera"]

    # Create output directory to save images
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    while simulation_app.is_running():
        with torch.inference_mode():
            """reseting logic"""

            if count % 10000 == 0:
                """reset the env."""
                print(f"[DEBUG]: Resetting Env..." f"\n[DEBUG]: static_obs = {s_obs}")

                # print(f"[DEBUG]: centroid pos : {centroid.data.root_com_pos_w}")

                # env origins
                env_origins = scene.env_origins.clone()

                new_positions = torch.zeros(
                    [5, scene.num_envs, 3], device=sim.device
                ).uniform_(-5, 5)
                new_positions[..., 2] = 0.25

                # adding env origins to env spacing
                new_positions[..., :] += env_origins

                print(f"[DEBUG]: New positions genrated.shape = {new_positions.shape}")
                print(f"[DEBUG]: New positions genrated = {new_positions}")

                # resetting static obstacles positions
                for i, obs in enumerate(s_obs):
                    root_pose = obs.data.root_com_pose_w.clone()
                    root_pose[:, :3] = new_positions[i, ...]
                    obs.write_root_com_pose_to_sim(root_pose=root_pose)

                dummy_vel_commands = torch.zeros(
                    [2, scene.num_envs, 2], device=scene.device
                ).uniform_(-1, 1)

                # resetting root pose of dynamic obstacles
                for i, obs in enumerate(d_obs):
                    default_dummy_pos = obs.data.root_com_pose_w.clone()
                    default_dummy_pos[:, :3] = env_origins
                    obs.write_root_com_pose_to_sim(root_pose=default_dummy_pos)

                count = 0

            # Apply keyboard velocity every 4 simulation steps
            if count % 4 == 0:

                root_velocity = scene["sensor_base"].data.root_com_vel_w.clone()

                vx, vy, wz = keyboard_controller.advance()

                root_velocity[:, 0] = vx
                root_velocity[:, 1] = vy
                root_velocity[:, 5] = wz

                scene["sensor_base"].write_root_com_velocity_to_sim(
                    root_velocity=root_velocity
                )

            if count % 10 == 0:

                depth_image = scene["camera"].data.output["distance_to_camera"]
                rgb_image = scene["rgb_camera"].data.output["rgb"]

                print("-------------------------------")
                print(f"{rgb_camera}")
                print(
                    "Received shape of RGB image: ",
                    rgb_camera.data.output["rgb"].shape,
                )

                print("-------------------------------")
                print(f"{camera}")
                print(
                    "Received shape of depth: ",
                    camera.data.output["distance_to_camera"].shape,
                )
                # print("Received shape of normals: ", scene["raycast_camera"].data.output["normals"].shape)

                save_images_grid(
                    images=[rgb_image[0], depth_image[0]],
                    subtitles=["rgb image ", "distance to camera"],
                    cmap="turbo",
                    title="MultiMeshRayCasterCamera depth image ",
                    filename=os.path.join(
                        output_dir,
                        "Randomized_Map" "MultiMeshRayCasterCamera",
                        "distance_to_camera",
                        f"{count:04d}.jpg",
                    ),
                )

            # writting random velocity commands @ 10hz
            if count % 20 == 0:
                dummy_root_velocities = []
                for i, obs in enumerate(d_obs):
                    dummy_root_velocity = obs.data.root_vel_w.clone()
                    dummy_root_velocity[:, :2] = dummy_vel_commands[i, ...]

                    dummy_root_velocities.append(dummy_root_velocity)

                    print(
                        f"[DEBUG]: D_obs{i} : Dummy vel command genrated = {dummy_vel_commands}"
                    )
                    print(
                        f"[DEBUG]: D_obs{i} : Dummy root velocity = {dummy_root_velocity}"
                    )

            # writing velocity every step else physics dissipates it
            for i, obs in enumerate(d_obs):
                obs.write_root_com_velocity_to_sim(
                    root_velocity=dummy_root_velocities[i]
                )

            # simulation step
            sim.step()
            scene.update(sim.cfg.dt)

            count += 1


# main
def main():
    """Implementation of main function."""

    sim_cfg = SimulationCfg(
        dt=1.0 / 200.0,  # simulation running @ 200 Hz
        render_interval=4,  # graphic rendering @ 50 Hz
    )

    sim = SimulationContext(cfg=sim_cfg)

    scene_cfg = CustomScene(num_envs=1, env_spacing=12.2)
    scene = InteractiveScene(cfg=scene_cfg)

    # Keyboard controller
    keyboard_controller_cfg = Se2KeyboardCfg(
        v_x_sensitivity=1.0,
        v_y_sensitivity=1.0,
        omega_z_sensitivity=2.0,
    )

    keyboard_controller = Se2Keyboard(cfg=keyboard_controller_cfg)

    # design the scene using interactive scene

    # setup the simulation
    sim.reset()

    print("[DEBUG]: Setup Complete...")

    # run simulation
    run_simulation(sim=sim, scene=scene, keyboard_controller=keyboard_controller)


if __name__ == "__main__":
    main()

    simulation_app.close()

"""
mkdir -p multimesh_0_1740

cp output/Randomized_MapMultiMeshRayCasterCamera/distance_to_camera/{0000..1740..10}.jpg multimesh_0_1740/


ffmpeg -framerate 20 \
  -pattern_type glob \
  -i 'multimesh_0_1740/*.jpg' \
  -vf "scale=iw-mod(iw\,2):ih-mod(ih\,2)" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  multimesh_0_1740.mp4 
   
     """
