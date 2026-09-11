from isaaclab.utils import configclass

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sensors import (
    CameraCfg,
    MultiMeshRayCasterCameraCfg,
    MultiMeshRayCasterCfg,
    ContactSensorCfg
)

import isaaclab.sim as sim_utils

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import torch
import numpy as np

OBSTACLE_GRAVEYARD_POS = (0.0, 0.0, 10)

unitree_go2_cfg = UNITREE_GO2_CFG.copy()


# ============================================================================
# Static obstacle pool generation
# ============================================================================
#
# Generates a fixed-size pool of static obstacles (s_obs_0 .. s_obs_{N-1}).
# Footprint (shape + size_x/size_y) cycles deterministically through the
# templates below so every run has the same set of obstacle "identities" —
# only the height of each obstacle is randomized, once, using a fixed seed,
# so the pool is reproducible ("constant") across runs while still giving
# height variety for the navigation policy to generalize over.
#
# get_obstacle_pool() (mdp/utils.py) reads obstacle sizes straight off the
# scene at runtime, so nothing outside this file needs to change to pick up
# a different NUM_STATIC_OBSTACLES / OBSTACLE_POOL_SEED.
# ============================================================================

NUM_STATIC_OBSTACLES = 50

OBSTACLE_POOL_SEED = 42

MIN_OBSTACLE_HEIGHT = 0.3
MAX_OBSTACLE_HEIGHT = 1.8

# (shape, size_x, size_y) — for "cylinder"/"cone", size_x == size_y == radius.
_OBSTACLE_FOOTPRINT_TEMPLATES: list[tuple[str, float, float]] = [
    ("cuboid", 0.3, 0.3),
    ("cuboid", 0.7, 0.7),
    ("cuboid", 1.2, 0.5),
    ("cuboid", 2.0, 1.3),
    ("cuboid", 0.6, 1.5),
    ("cuboid", 0.4, 0.8),
    ("cylinder", 0.2, 0.2),
    ("cylinder", 0.5, 0.5),
    ("cylinder", 0.9, 0.9),
    ("cylinder", 0.6, 0.6),
    ("cone", 0.4, 0.4),
    ("cone", 0.8, 0.8),
    ("cone", 1.2, 1.2),
]


def _build_static_obstacle_cfg(
    index: int,
    shape: str,
    size_x: float,
    size_y: float,
    height: float,
) -> RigidObjectCfg:
    """Build one static obstacle RigidObjectCfg, spawned in the graveyard."""

    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=True,
        kinematic_enabled=True,  # so that it remains stationary
    )
    mass_props = sim_utils.MassPropertiesCfg(mass=0.5)
    collision_props = sim_utils.CollisionPropertiesCfg()
    visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.3))

    if shape == "cuboid":
        spawn = sim_utils.CuboidCfg(
            size=(size_x, size_y, height),
            mass_props=mass_props,
            rigid_props=rigid_props,
            collision_props=collision_props,
            visual_material=visual_material,
        )
    elif shape == "cylinder":
        spawn = sim_utils.CylinderCfg(
            radius=size_x,
            height=height,
            mass_props=mass_props,
            rigid_props=rigid_props,
            collision_props=collision_props,
            visual_material=visual_material,
        )
    elif shape == "cone":
        spawn = sim_utils.ConeCfg(
            radius=size_x,
            height=height,
            mass_props=mass_props,
            rigid_props=rigid_props,
            collision_props=collision_props,
            visual_material=visual_material,
        )
    else:
        raise ValueError(f"unknown obstacle shape: {shape}")

    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/static_obstacles/s_obs_{index}",
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBSTACLE_GRAVEYARD_POS),
    )


def _build_static_obstacle_pool(num_obstacles: int, seed: int) -> dict[str, RigidObjectCfg]:
    """Build `num_obstacles` static obstacle configs with seeded random heights."""

    rng = np.random.default_rng(seed)

    pool: dict[str, RigidObjectCfg] = {}

    for i in range(num_obstacles):

        shape, size_x, size_y = _OBSTACLE_FOOTPRINT_TEMPLATES[i % len(_OBSTACLE_FOOTPRINT_TEMPLATES)]

        height = float(rng.uniform(MIN_OBSTACLE_HEIGHT, MAX_OBSTACLE_HEIGHT))

        pool[f"s_obs_{i}"] = _build_static_obstacle_cfg(i, shape, size_x, size_y, height)

    return pool



@configclass
class HelixNavTestRLScene(InteractiveSceneCfg):
    """
    Custom Scene config:
    1. ground
    2. robot
    3. rgb camera (only for debug purpose)
    4. MultiMeshRayCaster Camera
    5. Wall boundary
    6. static obstacles
    7. light
    """

    # ground
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
            )
        ),
    )

    # robot
    robot: ArticulationCfg = unitree_go2_cfg.replace(
        prim_path="{ENV_REGEX_NS}/robot",
    )

    # sensors

    # robot height sensor :
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    """
    Simulating Intel Realsense D345i
    16:9 image ratio
    running at 20 FPS
    horizontal_aperture=45.55,  # 45.55mm ->  HFoV : 87 degrees
    vertical_aperture=26.6068,  # 26.6068mm -> VFoV : 58 degrees
    Accurate depth range = 0.3 to 3.0 m
    """

    # # adding rgb camera
    # rgb_camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/robot/base/front_cam",
    #     update_period=0.05,  # 0.05 s -> 20Hz ~ 20FPS
    #     offset=CameraCfg.OffsetCfg(
    #         pos=(0.3, 0.0, 0.1),
    #         rot=(0.5, -0.5, 0.5, -0.5),
    #         convention="ros",
    #     ),
    #     height=480,
    #     width=640,
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0,
    #         focus_distance=400.0,
    #         horizontal_aperture=45.55,  # 45.55mm ->  HFoV : 87 degrees
    #         vertical_aperture=26.6068,  # 26.6068mm -> VFoV : 58 degrees
    #         clipping_range=(0.1, 1.0e5),
    #     ),
    # )

    # depth camera
    depth_camera = MultiMeshRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/base",
        update_period=0.05,  # depth camera running @ 20 FPS
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,  # 24mm
            horizontal_aperture=45.55,  # 45.55mm ->  HFoV : 87 degrees
            vertical_aperture=26.6068,  # 26.6068mm -> VFoV : 58 degrees
            width=96,  # 96px
            height=54,  # 54px
        ),
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=(0.3, 0.0, 0.1),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
        mesh_prim_paths=[
            "/World/ground",
            "{ENV_REGEX_NS}/wall_north",
            "{ENV_REGEX_NS}/wall_south",
            "{ENV_REGEX_NS}/wall_east",
            "{ENV_REGEX_NS}/wall_west",
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/static_obstacles/s_obs.*",
                track_mesh_transforms=True,
            ),
        ],
        ray_alignment="base",
        max_distance=5.0,  # in order to simualte Realsense depth range
        depth_clipping_behavior="none",
        data_types=["distance_to_camera"],
    )

    # collision sensors :
    collision_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/robot/.*_(hip|thigh)|Head_(upper|lower)",
        update_period=0.0,
    )

    # walls

    
    # north wall
    wall_north = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/wall_north",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 40.0, 2.5),
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
            pos=(20.0, 0.0, 1.25),
        ),
    )

    # south wall
    wall_south = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/wall_south",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 40.0, 2.5),
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
            pos=(-20.0, 0.0, 1.25),
        ),
    )

    # east wall
    wall_east = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/wall_east",
        spawn=sim_utils.CuboidCfg(
            size=(40.1, 0.1, 2.5),
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
            pos=(0.0, -20.0, 1.25),
        ),
    )

    # west wall
    wall_west = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/wall_west",
        spawn=sim_utils.CuboidCfg(
            size=(
                40.1,
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
            pos=(0.0, 20.0, 1.25),
        ),
    )

    """Rigid body obstacles. Placement/count controlled by NUM_STATIC_OBSTACLES."""

    # creating dummy parent prim
    static_obstacles = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/static_obstacles",
        spawn=sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01), visible=False),
    )


    """
    Obstacle pool: NUM_STATIC_OBSTACLES obstacles (s_obs_0 .. s_obs_{N-1}),
    generated by _build_static_obstacle_pool() above. Footprint cycles
    through _OBSTACLE_FOOTPRINT_TEMPLATES (cuboid/cylinder/cone identities);
    height is randomized once per obstacle using OBSTACLE_POOL_SEED, so the
    pool is reproducible across runs. All obstacles spawn at
    OBSTACLE_GRAVEYARD_POS and are moved into the arena at reset time.
    """

    # Injects s_obs_0 .. s_obs_{NUM_STATIC_OBSTACLES - 1} as class members.
    # Safe because a class body's `locals()` is the real namespace dict that
    # becomes cls.__dict__ — configclass infers annotations for any
    # unannotated member from its value (see other members above, e.g.
    # `ground`, `wall_north`), so this works exactly like writing each
    # s_obs_N = RigidObjectCfg(...) out by hand.
    locals().update(_build_static_obstacle_pool(NUM_STATIC_OBSTACLES, OBSTACLE_POOL_SEED))


    # light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )

    # Global occupancy grid creator sensor

    # occupancy_scanner_mount
    occupancy_scanner_mount = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/occupancy_scanner_mount",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.05, 0.05, 0.05),
            visible=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 5.0),  # ceiling height
        ),
    )

    occupancy_scanner = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/occupancy_scanner_mount",
        mesh_prim_paths=[
            # NO ground — free cells must return max_distance
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/static_obstacles/s_obs.*",
                track_mesh_transforms=True,
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/wall_.*",
                track_mesh_transforms=False,
                is_shared=True,
            ),
            # MultiMeshRayCasterCfg.RaycastTargetCfg(
            #     prim_expr="{ENV_REGEX_NS}/static_obstacles/test_obs",
            #     track_mesh_transforms=True,
            # ),
        ],
        update_period=999.0,  # effectively never auto-updates
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=(12, 12)),
        max_distance=10.0,
        offset=MultiMeshRayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
        ),
        debug_vis=True,
    )
