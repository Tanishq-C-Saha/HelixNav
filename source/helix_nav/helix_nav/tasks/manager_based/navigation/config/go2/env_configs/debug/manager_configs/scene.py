from isaaclab.utils import configclass

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sensors import (
    CameraCfg,
    MultiMeshRayCasterCameraCfg,
    MultiMeshRayCasterCfg,
)

import isaaclab.sim as sim_utils

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import torch

OBSTACLE_GRAVEYARD_POS = (0.0, 0.0, 10)

unitree_go2_cfg = UNITREE_GO2_CFG.copy()



@configclass
class HelixNavDebugBaseScene(InteractiveSceneCfg):
    """
    Custom Scene config:
    1. ground
    2. robot
    3. rgb camera
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

    # adding rgb camera
    rgb_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/base/front_cam",
        update_period=0.05,  # 0.05 s -> 20Hz ~ 20FPS
        offset=CameraCfg.OffsetCfg(
            pos=(0.3, 0.0, 0.1),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
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
    )

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

    # walls

    # north wall
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

    # south wall
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

    # east wall
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

    # west wall
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

    """Rigid body obstacles.
    Count = 5
    will change placement accoprding to need."""

    # creating dummy parent prim
    static_obstacles = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/static_obstacles",
        spawn=sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01), visible=False),
    )


    """
    Obstacle summary genrated using GPT
    ============================================================
    OBSTACLE POOL / GRAVEYARD — 13 obstacle specifications
    
    All obstacles are initially spawned at OBSTACLE_GRAVEYARD_POS
    and can later be moved into the active navigation arena.
    
    INDEX | SHAPE      | DIMENSIONS / PARAMETERS
    ------------------------------------------------------------
    obs_0 | Cube       | 0.30 × 0.30 × 0.30 m
    obs_1 | Cube       | 0.70 × 0.70 × 0.70 m
    obs_2 | Rectangle  | 1.20 × 0.50 × 0.50 m
    obs_3 | Rectangle  | 2.00 × 1.30 × 1.20 m
    obs_4 | Rectangle  | 0.60 × 1.50 × 0.80 m
    obs_5 | Rectangle  | 0.40 × 0.80 × 1.40 m
    obs_6 | Cylinder   | radius = 0.20 m, height = 0.70 m
    obs_7 | Cylinder   | radius = 0.50 m, height = 0.50 m
    obs_8 | Cylinder   | radius = 0.90 m, height = 1.80 m
    obs_9 | Cylinder   | radius = 0.60 m, height = 0.60 m
    obs_10| Cone       | radius = 0.40 m, height = 0.50 m
    obs_11| Cone       | radius = 0.80 m, height = 0.75 m
    obs_12| Cone       | radius = 1.20 m, height = 1.00 m
    
    ------------------------------------------------------------
    FOOTPRINT RANGE
    ------------------------------------------------------------
    Smallest footprint:
      Cube      = 0.30 × 0.30 m
    
    Largest footprint:
      Cylinder  = diameter 1.80 m
      Cone      = diameter 2.40 m
      Rectangle = 2.00 × 1.30 m
    
    Height range:
      Minimum = 0.30 m
      Maximum = 1.80 m
    
    Shapes:
      Cubes      → obs_0, obs_1
      Rectangles → obs_2, obs_3, obs_4, obs_5
      Cylinders  → obs_6, obs_7, obs_8, obs_9
      Cones      → obs_10, obs_11, obs_12
    
    ============================================================
    
    """


    # static obstacle 1
    s_obs_0 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/static_obstacles/s_obs_0",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.3),
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
            pos=OBSTACLE_GRAVEYARD_POS
            # rot=()
        ),
    )

    s_obs_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/static_obstacles/s_obs_1",
        spawn=sim_utils.CuboidCfg(
            size=(0.7, 0.7, 0.7),
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
        i


@configclass
class HelixNavDebugBaseScene(Interactinit_state=RigidObjectCfg.InitialStateCfg(
            pos=OBSTACLE_GRAVEYARD_POS
            # rot=()
        ),
    )

    s_obs_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/static_obstacles/s_obs_2",
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
            pos=OBSTACLE_GRAVEYARD_POS
            # rot=()
        ),
    )

    s_obs_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/static_obstacles/s_obs_3",
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 1.3, 1.2),
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
            pos=OBSTACLE_GRAVEYARD_POS
            # rot=()
        ),
    )

    s_obs_4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/static_obstacles/s_obs_4",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 1.5, 0.8),
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
            pos=OBSTACLE_GRAVEYARD_POS
            # rot=()
        ),
    )

    s_obs_5 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/static_obstacles/s_obs_5",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.8, 1.4),
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
            pos=OBSTACLE_GRAVEYARD_POS
            # rot=()
        ),
    )

    s_obs_6 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/static_obstacles/s_obs_6",
        spawn=sim_utils.CylinderCfg(
            radius=0.2,
            height=0.7,
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
            pos=OBSTACLE_GRAVEYARD_POS
            # rot=()
        ),
    )

    s_obs_7 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/static_obstacles/s_obs_7",
        spawn=sim_utils.CylinderCfg(
            radius=0.5,
            height=0.5,
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
            pos=OBSTACLE_GRAVEYARD_POS
            # rot=()
        ),
    )

    s_obs_8 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/static_obstacles/s_obs_8",
        spawn=sim_utils.CylinderCfg(
            radius=0.9,
            height=1.8,
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
            pos=OBSTACLE_GRAVEYARD_POS
            # rot=()
        ),
    )

    s_obs_9 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/static_obstacles/s_obs_9",
        spawn=sim_utils.CylinderCfg(
            radius=0.6,
            height=0.6,
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
            pos=OBSTACLE_GRAVEYARD_POS
            # rot=()
        ),
    )

    s_obs_10 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/static_obstacles/s_obs_10",
        spawn=sim_utils.ConeCfg(
            radius=0.4,
            height=0.5,
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
            pos=OBSTACLE_GRAVEYARD_POS
            # rot=()
        ),
    )

    s_obs_11 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/static_obstacles/s_obs_11",
        spawn=sim_utils.ConeCfg(
            radius=0.8,
            height=0.75,
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
            pos=OBSTACLE_GRAVEYARD_POS
            # rot=()
        ),
    )

    s_obs_12 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/static_obstacles/s_obs_12",
        spawn=sim_utils.ConeCfg(
            radius=1.2,
            height=1.0,
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
            pos=OBSTACLE_GRAVEYARD_POS
        ),
    )



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
