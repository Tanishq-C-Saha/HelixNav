"""A script that spawns 4 cones at different origins,drops them 
    from random heights on a cylinder,lets them fall and settle, 
    and resets them periodically, forever, until you close the sim.
"""

import argparse 

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()

AppLauncher.add_app_launcher_args(parser=parser)

cli_args = parser.parse_args()

# start the omniverse app
app_launcher = AppLauncher(cli_args)
simulation_app = app_launcher.app


"""Everything rest follows."""


from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import RigidObjectCfg, RigidObject

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg   # confirm both exist at this import path

import isaaclab.sim as sim_utils
import torch 
import isaaclab.utils.math as math_utils 





def design_scene(num_origins:int = 4):
    """design the scene with:
        1. ground
        2. light
        3. cones : RigidBody
    """

    # ground 
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func(
        prim_path="/World/ground",
        cfg=ground_cfg
    )

    # creating 
    origins =[]
    for i in range(num_origins):
        origin = [0.0+i*4, 0.0, 0.0]
        sim_utils.create_prim(
            prim_path=f"/World/Origins_{i}",
            prim_type="Xform",
            translation=tuple(origin),
        )
        origins.append(origin)


    cone_cfg = RigidObjectCfg(
        prim_path="/World/Origins_.*/Cone",
        spawn=sim_utils.ConeCfg(
            radius=0.5,  # 0.5 m
            height=1,  # 1m
            mass_props=sim_utils.MassPropertiesCfg(mass=1),  # 1kg
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.8, 0.7)
            )
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 3.0),
        )
    )

    cone = RigidObject(cfg=cone_cfg)

    # light cfg
    light_cfg = sim_utils.DomeLightCfg(
        intensity=7000.0,
        color=(0.75, 0.75, 0.75)
    )
    light_cfg.func(
        prim_path="/World/light",
        cfg=light_cfg
    )

    # preprocessing origins to tensor
    origins = torch.tensor(origins).to(device="cuda:0")
    print(f"[DEBUG]: origins = {origins}\n"
          f"[DEBUG]: origins.shape = {origins.shape}")


    return {"cone": cone,
            "origins": origins}


# running simulation
def run_simulation(sim, cone, origins, keyboard):
    while simulation_app.is_running():
        with torch.inference_mode():
            command = keyboard.advance()   # likely returns something like [vx, vy, wz] — verify shape by printing once

            root_state = cone.data.root_state_w.clone()
            root_state[:, 0] += command[0] * sim.get_physics_dt()
            root_state[:, 1] += command[1] * sim.get_physics_dt()

            cone.write_root_link_pose_to_sim(root_pose=root_state[:, :7])
            cone.write_root_velocity_to_sim(root_velocity=torch.zeros_like(root_state[:, 7:]))

        sim.step()




# main
def main():
    """Main function implementation.
    Flow:
    1. simulation config setup
    2. simulation context setup
    3. design scene
    4. setup phsyics scene
    5. run simulation
    """

    sim_cfg = SimulationCfg(dt=1.0/200.0)
    sim = SimulationContext(sim_cfg)

    items = design_scene()
    cone = items["cone"]
    origins = items["origins"]

    sim.reset()
    print("[INFO]: Setup complete...")

    keyboard_cfg = Se2KeyboardCfg()   # check if it has usable defaults, or needs explicit sensitivity values
    keyboard = Se2Keyboard(cfg=keyboard_cfg)
    print(keyboard)   # most of these print their own key-binding help text — check it

    run_simulation(sim=sim, cone=cone, origins=origins, keyboard=keyboard)


if __name__ == "__main__":
    main()

    simulation_app.close()

