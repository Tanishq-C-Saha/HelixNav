"""Script for testing the contact sensor."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()

# adding cli args
AppLauncher.add_app_launcher_args(parser=parser)

# reading cli args
cli_args = parser.parse_args()

# starting omniverse with context
app_launcher = AppLauncher(launcher_args=cli_args)

simulation_app = app_launcher.app


"""Rest everything follows!"""


from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import RigidObjectCfg, RigidObject
from isaaclab.sensors import ContactSensorCfg, ContactSensor
from isaaclab.devices import Se2KeyboardCfg, Se2Keyboard
import isaaclab.sim as sim_utils 

import torch


# designing custom scene
def design_scene():
    """Designing custom scene that contains:
    1. ground
    2. moving cube with contact sensor 
    3. multiple obstacles
    4. light
    """

    # ground 
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func(
        prim_path="/World/ground",
        cfg=ground_cfg
    )

    # controllable cube : RigidObject
    cube_cfg = RigidObjectCfg(
        prim_path="/World/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.6, 0.8),
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5)
        )
    )

    cube = RigidObject(cfg=cube_cfg)

    # contact_sensor 
    contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/cube",
        update_period=0.02,  # updating the sensor @ 50Hz
        debug_vis=True,
        filter_prim_paths_expr=["/World/obs_1", "/World/obs_2"],
        force_threshold=1.0
    )

    contact_sensor = ContactSensor(cfg=contact_sensor_cfg)

    # obstacles 
    obs_1_cfg = RigidObjectCfg(
        prim_path="/World/obs_1",
        spawn=sim_utils.CuboidCfg(
            size=(0.9, 0.7, 0.5),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.1, 0.3),
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.0, 0.9, 0.25)
        )
    )

    obs_1 = RigidObject(cfg=obs_1_cfg)

    # obstacles 
    obs_2_cfg = RigidObjectCfg(
        prim_path="/World/obs_2",
        spawn=sim_utils.CuboidCfg(
            size=(1.9, 0.1, 1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.1, 0.3),
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-3.0, -0.4, 0.5)
        )
    )

    obs_2 = RigidObject(cfg=obs_2_cfg)

    obs: list[RigidObject] = [obs_1, obs_2]

    return {
        "cube": cube,
        "obs": obs,
        "contact_sensor": contact_sensor,
    }


# running simulation
def run_simulation(sim:SimulationContext, scene: dict, keyboard_controller:Se2Keyboard):

    cube: RigidObject = scene["cube"]
    obs = scene["obs"]
    obs_1: RigidObject = obs[0] 
    obs_2: RigidObject = obs[1]
    contact_sensor: ContactSensor = scene["contact_sensor"]

    count = 0

    while simulation_app.is_running():
        with torch.inference_mode():

            if count % 4 == 0:
                # run control loop
                root_velocity = cube.data.root_com_vel_w.clone()

                vx, vy, wz = keyboard_controller.advance()

                root_velocity[:, 0] = vx
                root_velocity[:, 1] = vy
                root_velocity[:, 5] = wz

                cube.write_root_com_velocity_to_sim(root_velocity=root_velocity)

                force_w = contact_sensor.data.force_matrix_w

                print(f"[DEBUG]: Contact forces(N, B, M, 3) = {force_w}")


            sim.step()
            cube.update(dt=sim.cfg.dt)
            contact_sensor.update(dt=sim.cfg.dt)


# main function implementation
def main():
    """Main function implementation."""
    sim_cfg = SimulationCfg(
        dt=1.0 / 200.0,  # simulation running @ 200Hz
        render_interval=4  # redering @ 50Hz
    )  

    sim = SimulationContext(cfg=sim_cfg)

    # design the scene
    scene = design_scene()

    # setting up keyboard controller 
    keyboard_controller = Se2Keyboard(
        cfg=Se2KeyboardCfg(
            v_x_sensitivity=2.0,
            v_y_sensitivity=0.8,
            omega_z_sensitivity=1.5,
        )
    )
    
    # setup the simulation
    sim.reset()

    print(f"[DEBUG]: Setup complete...")

    # run simulation
    run_simulation(sim=sim, scene=scene, keyboard_controller=keyboard_controller)


if __name__ == "__main__":
    main()

    # close the simulation app
    simulation_app.close()
