"""Spawn a controllable cube and a ray-caster sensor."""

import argparse

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser=parser)
cli_args = parser.parse_args()

app_launcher = AppLauncher(cli_args)
simulation_app = app_launcher.app


from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg, SimulationContext
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils


def design_scene():
    """Create the ground, controllable cube, ray-caster, and light."""

    # Ground
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func(
        prim_path="/World/ground",
        cfg=ground_cfg,
    )

    # Controllable cube
    cube_cfg = RigidObjectCfg(
        prim_path="/World/Cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(1.5, 1.5, 1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.8, 0.0),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.0, 0.0, 0.5),
        ),
    )

    cube = RigidObject(cfg=cube_cfg)

    # Ray-caster base
    sensor_base_cfg = RigidObjectCfg(
        prim_path="/World/Sensor_base",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.0, 0.2),
                metallic=0.4,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            rot=(0.7071068, 0, -0.7071068, 0)
        )
    )

    sensor_base = RigidObject(cfg=sensor_base_cfg)



    # Ray-caster
    sensor_cfg = RayCasterCfg(
        prim_path="/World/Sensor_base",
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.01, 0.0, 0.01),
            #rot=(0.7071068, 0, -0.7071068, 0),  # w x y z   -90 degree in y
        ),
        update_period=0.02,
        history_length=0,
        mesh_prim_paths=["/World/Cube"],
        ray_alignment="base",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1,
            size=(1.6, 1.6),
            # direction=(0.0, 1.0, 0.0),
        ),
        debug_vis=True,
        max_distance=10.0,
    )

    sensor = RayCaster(cfg=sensor_cfg)

    # Light
    light_cfg = sim_utils.DomeLightCfg(
        intensity=7000.0,
        color=(0.75, 0.75, 0.75),
    )

    light_cfg.func(
        prim_path="/World/light",
        cfg=light_cfg,
    )

    return {
        "cube": cube,
        "sensor": sensor,
        "sensor_base": sensor_base,
    }


def run_simulation(sim, cube, sensor, sensor_base, keyboard):
    """Run the simulation and control the cube with the keyboard."""

    count = 0 
    while simulation_app.is_running():
        with torch.inference_mode():

            # Read keyboard command
            vx, vy, wz = keyboard.advance()

            # Read current cube velocity
            root_velocity = sensor_base.data.root_com_vel_w

            # Apply keyboard velocity
            root_velocity[:, 0] = vx
            root_velocity[:, 1] = vy
            root_velocity[:, 3] = 0.0      # angular x
            root_velocity[:, 4] = 0.0      # angular y
            root_velocity[:, 5] = wz
            sensor_base.write_root_com_velocity_to_sim(root_velocity=root_velocity)
            # Step simulation
            sim.step()

            # Update objects and sensors
            sensor_base.update(sim.cfg.dt)
            sensor.update(sim.cfg.dt)

            if count % 50 == 0:
                # print once after a few sim steps, not every frame
                sensor_pos = sensor.data.pos_w[0]

                # w, x, y, z — the sensor's actual live world orientation
                sensor_quat = sensor.data.quat_w[0]

                local_dir = torch.tensor([0.0, 0.0, -1.0], device=sensor_quat.device)
                world_dir = math_utils.quat_apply(
                    sensor_quat.unsqueeze(0), local_dir.unsqueeze(0))[0]
                print(f"\n\n\n[DEBUG] sensor fires toward: {world_dir}\n\n")
                count = 0
                
            if count % 10 == 0:
                print(f"[DEBUG] commanded wz: {wz:.3f} | actual wz after step: {sensor_base.data.root_com_vel_w[0, 5].item():.3f}")

        count += 1




def main():
    """Set up and run the simulation."""

    # Simulation
    sim_cfg = SimulationCfg(dt=1.0 / 200.0)
    sim = SimulationContext(sim_cfg)

    # Scene
    scene = design_scene()
    cube = scene["cube"]
    sensor = scene["sensor"]
    sensor_base = scene["sensor_base"]

    # Initialize simulation
    sim.reset()
    print("[INFO]: Setup complete...")

    # Keyboard
    keyboard = Se2Keyboard(cfg=Se2KeyboardCfg(v_x_sensitivity=2.0, v_y_sensitivity=1.0))
    print(keyboard)

    # Run
    run_simulation(
        sim=sim,
        cube=cube,
        sensor=sensor,
        sensor_base=sensor_base,
        keyboard=keyboard,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
