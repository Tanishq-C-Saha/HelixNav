"""Spawn a controllable cube on which ray caster camera sits."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser=parser)
cli_args = parser.parse_args()

# Start Omniverse with context
app_launcher = AppLauncher(cli_args)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import RayCasterCamera, RayCasterCameraCfg, patterns
from isaaclab.devices import Se2KeyboardCfg, Se2Keyboard

import isaaclab.sim as sim_utils
import os
import torch
import matplotlib.pyplot as plt
import numpy as np


# Design scene
def design_scene():
    """Design custom scene with:
    1. ground
    2. one obstacle
    3. sensor base
    4. light
    5. ray caster camera
    """

    # Ground
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func(cfg=ground_cfg, prim_path="/World/ground")

    # One obstacle cube
    cube_cfg = RigidObjectCfg(
        prim_path="/World/cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 0.25)),
    )

    cube = RigidObject(cfg=cube_cfg)

    # Sensor base
    sensor_base_cfg = RigidObjectCfg(
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
            pos=(0.0, 0.0, 0.7),
        ),
    )

    sensor_base = RigidObject(cfg=sensor_base_cfg)

    # Ray caster camera
    camera_cfg = RayCasterCameraCfg(
        prim_path="/World/sensor_base",
        update_period=0.02,
        debug_vis=True,
        ray_alignment="base",
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,  # 24mmwidth=80, height=80
            horizontal_aperture=45.55,  # 45.55mm ->  HFoV : 87 degrees
            vertical_aperture=26.6068,  # 26.6068mm -> VFoV : 58 degrees
            height=51,  # 51 px
            width=90,  # 90 px
        ),
        # Only this cube is visible to the ray caster
        mesh_prim_paths=["/World/cube"],
        max_distance=10,
        data_types=["distance_to_camera"],
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.1, 0, 0.01),
            rot=(0.7071068, 0, 0.7071068, 0),  # 90 degres in y
        ),
    )

    camera = RayCasterCamera(cfg=camera_cfg)

    # Light
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))

    light_cfg.func(prim_path="/World/light", cfg=light_cfg)

    return {
        "cube": cube,
        "sensor_base": sensor_base,
        "camera": camera,
    }


# helper func : saving images

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


# Run simulation
def run_simulation(
    sim: SimulationContext,
    sensor_base: RigidObject,
    camera: RayCasterCamera,
    keyboard_controller: Se2Keyboard,
):
    """Run the simulation."""

    # Create output directory to save images
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    count = 0

    while simulation_app.is_running():

        with torch.inference_mode():

            # Apply keyboard velocity every 4 simulation steps
            if count % 4 == 0:

                root_velocity = sensor_base.data.root_com_vel_w.clone()

                vx, vy, wz = keyboard_controller.advance()

                root_velocity[:, 0] = vx
                root_velocity[:, 1] = vy
                root_velocity[:, 5] = wz

                sensor_base.write_root_com_velocity_to_sim(root_velocity=root_velocity)

            if count % 10 == 0:

                depth_image = camera.data.output["distance_to_camera"]

                print("-------------------------------")
                print(f"{camera}")
                print(
                    "Received shape of depth: ",
                    camera.data.output["distance_to_camera"].shape,
                )
                # print("Received shape of normals: ", scene["raycast_camera"].data.output["normals"].shape)

                save_images_grid(
                    images=[depth_image[0]],
                    subtitles=["distance to camera"],
                    cmap="turbo",
                    title="RayCasterCamera depth image",
                    filename=os.path.join(
                        output_dir, "distance_to_camera", f"{count:04d}.jpg"
                    ),
                )

            # Step simulation
            sim.step()

            # Update sensor base
            sensor_base.update(sim.cfg.dt)

            # Update ray caster camera
            camera.update(sim.cfg.dt)

            count += 1


# Main
def main():

    sim_cfg = SimulationCfg(dt=1.0 / 200.0)

    sim = SimulationContext(cfg=sim_cfg)

    # Design scene
    scene = design_scene()

    # Keyboard controller
    keyboard_controller_cfg = Se2KeyboardCfg(
        v_x_sensitivity=1.0,
        v_y_sensitivity=1.0,
        omega_z_sensitivity=2.0,
    )

    keyboard_controller = Se2Keyboard(cfg=keyboard_controller_cfg)

    sensor_base = scene["sensor_base"]
    camera = scene["camera"]

    # Reset simulation to start PhysX
    sim.reset()

    print("[DEBUG]: Setup Complete...")

    run_simulation(
        sim=sim,
        sensor_base=sensor_base,
        camera=camera,
        keyboard_controller=keyboard_controller,
    )


if __name__ == "__main__":

    main()

    simulation_app.close()
