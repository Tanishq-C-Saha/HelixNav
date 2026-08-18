import argparse

from isaaclab.app import AppLauncher

# create parser
parser = argparse.ArgumentParser()

# append app launcher cli args in the parser
AppLauncher.add_app_launcher_args(parser=parser)

# parse cli args
cli_args = parser.parse_args()

# pass cli args and setup omniverse app
app_launcher = AppLauncher(cli_args)
simulation_app = app_launcher.app


"""Everything below runs after Isaac Sim has been initialized."""


from isaaclab.utils import configclass
from isaaclab.sim import SimulationContext, SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR



@configclass
class MyScene(InteractiveSceneCfg):
    """Defining custom Interactive Scene to spawn:
    1. ground
    2. light
    3. cuboid : simple
    4. cone : rigid body
    5. usd file : table

    *Note :The adding of entities to the scene is sensitive to the order of the attributes in the configuration.
        Please make sure to add the entities in the order you want them to be added to the scene.
        The recommended order of specification is :
        terrain, physics-related assets (articulations and rigid bodies),
        sensors and non-physics-related assets (lights).
    """

    # ground cfg
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg()
    )


    # -------------------------------------------------------------------------
    # Dummy invisible cuboid used only to create the StaticObjects Xform.
    # Isaac Lab currently does not provide a simple Xform spawn configuration
    # for InteractiveScene, so this acts as an organizational parent.
    # -------------------------------------------------------------------------

    # create Xform
    static_objects = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/StaticObjects",
        spawn=sim_utils.CuboidCfg(
            size=(0.0001, 0.0001, 0.0001),
            visible=False
        ))

    # objects cfg
    # ENV_REGEX_NS : Environment Regular Expression Namespace

    # rigid cone  
    rigid_cone = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/StaticObjects/RigidCone",
        spawn=sim_utils.ConeCfg(
            radius=1.0,  # 1m
            height=2.0,  # 2m
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0),  # green color
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-2.0, -2.0, 4.0),
        )
    )

    # cuboid cone
    cuboid = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/StaticObjects/Cuboid",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.5),  # purple cuboid
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(1.0, -1.0, 0.5),
        )
    )

    # table usd
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/StaticObjects/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
            ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.05),
        )
    )

    # lights cfg
    lights = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            intensity=7000.0,
            color=(0.75, 0.75, 0.75),
        )
    )



def main():
    """Create the simulation and continuously step the physics."""

    sim_cfg = SimulationCfg(dt=1.0/200.0)  # simulation running @ 200Hz
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye=(3.5, 0, 3.2),
        target=(0.0, 0.0, 0.0)
    )

    # design scene 
    scene_cfg = MyScene(num_envs=4, env_spacing=10.0)
    scene=InteractiveScene(cfg=scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()

    simulation_app.close()
