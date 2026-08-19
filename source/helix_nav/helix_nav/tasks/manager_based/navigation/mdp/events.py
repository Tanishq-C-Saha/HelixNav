"""
Function definitions for custom functions for events
"""

from isaaclab.envs import ManagerBasedEnv
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import MultiMeshRayCaster

from helix_nav.tasks.manager_based.navigation.map_generators import (
    RandomMapGenerator,
    MapSpec,
    inflate_grid,
    CELLS,
    astar,
    grid_to_world
)


from .utils import get_obstacle_pool
from .visualize_utils import plot_nav_state, visualize_nav_states

import numpy as np
import torch
import os

# output dir
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")  # ecxpecting
os.makedirs(output_dir, exist_ok=True)


# constants
GRID_CELLS = 61
MAX_PATH_LENGTH = 300
GO2_STANDING_HEIGHT = 0.34
MAX_OBSTACLES = 15


def reset_map_and_spawn(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    visualize_map: bool = False,
):
    """
    Reset event: generate new map, place obstacles, run A*, spawn robot.

    Order of operations (strict):
        1. Generate map per env
        2. Write obstacle positions to sim
        3. Store occupancy grid + run A*
        4. Spawn robot at start position
    """

    # lazy initialize
    if not hasattr(env, "_nav_state_initialized"):
        init_nav_state(env)

    # *** 1. Generate maps for each reset env ***
    for idx, env_id in enumerate(env_ids.tolist()):
        seed = int(env_id * env._global_seed + env._reset_counter[env_id].item())
        env._reset_counter[env_id] += 1

        map_spec: MapSpec = env._map_generator.generate_with_retry(
            difficulty=env._current_difficulty,
            seed=seed,
        )

        env._map_specs[env_id] = map_spec

        # Store obstacle positions (LOCAL coordinates, no env_origin)
        # First zero out all slots (handles variable obstacle count)
        env._obstacles_pos[env_id] = 0.0
        env._obstacles_pos[env_id, :, 2] = -10.0  # unused slots hidden underground

        for obs_spec in map_spec.obstacles:
            env._obstacles_pos[env_id, obs_spec.pool_index, 0] = obs_spec.position[0]
            env._obstacles_pos[env_id, obs_spec.pool_index, 1] = obs_spec.position[1]
            env._obstacles_pos[env_id, obs_spec.pool_index, 2] = obs_spec.position[2]

        # Store start and goal (LOCAL coordinates)
        env._start_positions[env_id, 0] = map_spec.start_position[0]
        env._start_positions[env_id, 1] = map_spec.start_position[1]
        env._start_positions[env_id, 2] = GO2_STANDING_HEIGHT

        env._goal_positions[env_id, 0] = map_spec.goal_position[0]
        env._goal_positions[env_id, 1] = map_spec.goal_position[1]
        env._goal_positions[env_id, 2] = GO2_STANDING_HEIGHT

        # Store occupancy grid will implement wioth multimesh raycaster
        env._occupancy_grids[env_id] = torch.tensor(
            map_spec.occupancy_grid, dtype=torch.float32, device=env.device
        )

        # *** Run A* on inflated grid ***
        inflated = inflate_grid(map_spec.occupancy_grid, cells=1)
        path = astar(inflated, map_spec.start_grid, map_spec.goal_grid)

        # Store path
        env._path_lengths[env_id] = 0
        env._paths_world[env_id] = 0.0

        if path is not None:
            # Convert grid path to world coords
            path_world = np.array([grid_to_world(r, c) for r, c in path])
            path_len = min(len(path_world), MAX_PATH_LENGTH)
            env._paths_world[env_id, :path_len, :] = torch.tensor(
                path_world[:path_len], dtype=torch.float32, device=env.device
            )
            env._path_lengths[env_id] = path_len

            # Compute initial path_remaining (arc length)
            diffs = path_world[1:] - path_world[:-1]
            arc_length = float(np.sqrt((diffs**2).sum(axis=1)).sum())
            env._path_remaining[env_id] = arc_length
            env._prev_path_remaining[env_id] = arc_length

            if visualize_map:
                """Used to visualize the map."""

                # occupancy map from map specs
                plot_nav_state(
                    map_spec.occupancy_grid,
                    path_world=path_world,
                    start_world=map_spec.start_position,
                    goal_world=map_spec.goal_position,
                    title=f"MapSpec | Env {env_id} | D{env._current_difficulty} | Reset {env._reset_counter[env_id].item() - 1} | seed {seed}",
                    save_path=os.path.join(
                        output_dir,
                        "occupancy_maps",
                        "map_specs",
                        f"env_{env_id}",
                        f"{env._reset_counter[env_id]-1:04d}.jpg",
                    ),
                )

    # *** 2. Write obstacles to sim ***
    _spawn_obstacles(env, env_ids)

    # *** 3. Spawn robot ***
    _spawn_robot(env, env_ids, asset_cfg)

    # MultiMesh RayCaster
    occupancy_scanner: MultiMeshRayCaster = env.scene["occupancy_scanner"]


    if visualize_map:
        """Storing Occupancy map generated from MultiMesh RayCaster"""
        visualize_nav_states(
            env=env,
            env_ids=env_ids,
            hit_points_w=occupancy_scanner.data.ray_hits_w.clone(),
            min_threshold=0.1,
            max_threshold=6,
            output_dir=output_dir      
        )

    # *** 4. Zero out previous actions ***
    env._prev_actions[env_ids] = 0.0


def _spawn_obstacles(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """Write obstacle positions to sim. Uses LOCAL coords + env_origin."""

    env_origins = env.scene.env_origins  # don't clone, just read

    for i, obs in enumerate(env._static_obstacles):
        pose = obs.data.root_com_pose_w.clone()

        # Local coords + env origin (NOT in-place on stored positions)
        pose[env_ids, 0] = env._obstacles_pos[env_ids, i, 0] + env_origins[env_ids, 0]
        pose[env_ids, 1] = env._obstacles_pos[env_ids, i, 1] + env_origins[env_ids, 1]
        pose[env_ids, 2] = env._obstacles_pos[env_ids, i, 2] + env_origins[env_ids, 2]
        pose[env_ids, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)

        obs.write_root_com_pose_to_sim(pose, env_ids=env_ids)


def _spawn_robot(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
):
    """Spawn robot at start position with random yaw."""

    asset: Articulation = env.scene[asset_cfg.name]
    env_origins = env.scene.env_origins

    # Get default root state for reset envs
    root_state = asset.data.default_root_state[env_ids].clone()

    # Position: local start + env origin
    root_state[:, 0] = env._start_positions[env_ids, 0] + env_origins[env_ids, 0]
    root_state[:, 1] = env._start_positions[env_ids, 1] + env_origins[env_ids, 1]
    root_state[:, 2] += env_origins[env_ids, 2]

    # Random yaw
    num_reset = len(env_ids)
    yaw = torch.empty(num_reset, device=env.device).uniform_(-3.14159, 3.14159)
    half_yaw = yaw / 2
    root_state[:, 3] = torch.cos(half_yaw)  # w
    root_state[:, 4] = 0.0  # x
    root_state[:, 5] = 0.0  # y
    root_state[:, 6] = torch.sin(half_yaw)  # z

    # Zero velocity
    root_state[:, 7:] = 0.0

    # Write pose and velocity
    asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    # Reset joints to default standing
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def init_nav_state(env: ManagerBasedEnv):
    """One-time initialization of persistent navigation state."""

    # Get obstacle pool from scene
    static_obstacles_dict = get_obstacle_pool(env)
    obstacle_pool = static_obstacles_dict["static_obstacles_pool"]
    env._static_obstacles = static_obstacles_dict["static_obstacles"]

    env._map_generator = RandomMapGenerator(obstacle_pool=obstacle_pool)

    env._reset_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    env._current_difficulty = 1

    if hasattr(env.cfg, "seed") and env.cfg.seed is not None:
        env._global_seed = env.cfg.seed
    else:
        env._global_seed = 100000

    # Per-env persistent tensors
    env._occupancy_grids = torch.zeros(
        env.num_envs, GRID_CELLS, GRID_CELLS, device=env.device
    )
    env._start_positions = torch.zeros(env.num_envs, 3, device=env.device)
    env._goal_positions = torch.zeros(env.num_envs, 3, device=env.device)
    env._paths_world = torch.zeros(
        env.num_envs, MAX_PATH_LENGTH, 2, device=env.device
    )
    env._path_lengths = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.long
    )
    env._path_remaining = torch.zeros(env.num_envs, device=env.device)
    env._prev_path_remaining = torch.zeros(env.num_envs, device=env.device)
    env._prev_actions = torch.zeros(env.num_envs, 3, device=env.device)
    env._obstacles_pos = torch.zeros(
        env.num_envs, len(env._static_obstacles), 3, device=env.device
    )
    env._map_specs = [None] * env.num_envs

    env._nav_state_initialized = (
        True  # note: fixed typo "intialized" → "initialized"
    )
