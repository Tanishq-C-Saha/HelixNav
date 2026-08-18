import argparse

from isaaclab.app import AppLauncher

# argument parser
parser = argparse.ArgumentParser()

# append AppLauncher cli agrs
AppLauncher.add_app_launcher_args(parser=parser)

# parse cli args 
cli_args = parser.parse_args()

# setup omniverse app
app_launcher = AppLauncher(cli_args)
simulation_app = app_launcher.app


"""Everything below runs after Isaac Sim has been initialized."""


from isaaclab.sim import SimulationCfg, SimulationContext
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


# design scene

def design_scene():
    """Design the scene by spwaning :
    1. ground plane
    2. light
    3. cone : simple
    4. cone : colliders adn rigid body
    5. cuboid : deformable body
    6. table : from usd file
    """

    # spawn ground plane 
    ground_plane_cfg = sim_utils.GroundPlaneCfg()
    ground_plane_cfg.func(prim_path="/World/ground", cfg=ground_plane_cfg)

    # spawn light 
    light_cfg = sim_utils.DistantLightCfg(
        intensity=7000,
        color=(0.9, 0.9, 0.9)
    )

    light_cfg.func(prim_path="/World/light", cfg=light_cfg)

    # create new Xform prim for all objects to be spawned under
    sim_utils.create_prim(prim_path="/World/Objects", prim_type="Xform")

    # spawn cone : simple
    simple_cone_cfg = sim_utils.ConeCfg(
        radius=0.5,  # 0.5m
        height=2.0,  # 2m
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.0, 0.0)),  # red color cone
    )

    simple_cone_cfg.func(prim_path="/World/Objects/Cone1",
                         cfg=simple_cone_cfg,
                         translation=(2.0, 0.0, 2.0)
                         )
    
    # creating another cone with same cfgs but different place\
    simple_cone_cfg.func(prim_path="/World/Objects/Cone2",
                         cfg=simple_cone_cfg,
                         translation=(4.0, 0.0, 2.0)
                         )


    # spawn cone : colliders and rigid body properties 
    rigid_cone_cfg = sim_utils.ConeCfg(
        radius=1.0,
        height=2.0,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 1.0, 0.0)),  # green color
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )

    rigid_cone_cfg.func(prim_path="/World/Objects/RigidCone",
                        cfg=rigid_cone_cfg,
                        translation=(-3.0, -2.0, 10.0))

    # spawn cuboid : deformable body
    deformable_cuboid_cfg = sim_utils.CuboidCfg(
        size=(1.0, 2.0, 2.0),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 0.0, 1.0)),  # blue color
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )

    deformable_cuboid_cfg.func(prim_path="/World/Objects/DeformableCuboid",
                               cfg=deformable_cuboid_cfg,
                               translation=(-3.0, -2.0, 2.0))

    # spawn usd file : table
    usd_table_cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    usd_table_cfg.func(prim_path="/World/Objects/Table",
                       cfg=usd_table_cfg,
                       translation=(0.0, 0.0, 1.05))

# main
def main():
    """Create the simulation and continuously step the physics."""

    sim_cfg = SimulationCfg(dt=1.0/200.0)  # simulation running @ 200Hz
    sim = SimulationContext(sim_cfg)

    design_scene()

    sim.reset()
    print("[INFO]: Setup complete...")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()

    simulation_app.close()
