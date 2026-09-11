"""
Standalone random map generator for a square HelixNav quadruped navigation
testing environment of ANY size (not just 40m).

No Isaac Lab, torch, ROS, or gymnasium imports.

Supports any square arena size (20m, 40m, 60m, 100m, ...) via `arena_size`.

The generator:
    1. Randomly places obstacles.
    2. Builds a (cells x cells) occupancy grid.
    3. Samples collision-free start and goal positions.
    4. Inflates the grid for robot clearance.
    5. Validates reachability using 8-connected BFS.

A* is NOT computed here. A* planning happens later in the RL/testing
environment (see astar.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from ..bfs import bfs_reachable
    from .grid_utils_universal import (
        ArenaSpec,
        inflate_grid,
        mark_obstacle,
        mark_walls,
        world_to_grid,
    )
except ImportError:
    # Allow running this file directly (e.g. `python3 random_map_gen_universal.py`)
    # without the full helix_nav/Isaac Lab package being importable.
    import os
    import sys

    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _THIS_DIR)
    sys.path.insert(0, os.path.dirname(_THIS_DIR))

    from bfs import bfs_reachable
    from grid_utils_universal import (
        ArenaSpec,
        inflate_grid,
        mark_obstacle,
        mark_walls,
        world_to_grid,
    )
# ============================================================================
# Obstacle size pool
# ============================================================================
#
# Each pool entry is:
#     (size_x, size_y, height)
#
# Pool indices are sampled WITHOUT replacement within one map.
# ============================================================================

OBSTACLE_POOL: list[tuple[float, float, float]] = [
    (0.5, 0.6, 0.5),
    (0.7, 0.3, 0.5),
    (1.2, 0.5, 0.5),
    (2.0, 0.8, 0.5),
    (0.8, 0.8, 0.5),
    (0.4, 1.5, 0.5),
    (0.6, 0.4, 0.5),
    (1.0, 1.0, 0.5),
    (0.3, 0.3, 0.5),
    (1.5, 0.3, 0.5),
    (0.9, 0.6, 0.5),
    (0.5, 1.0, 0.5),
    (1.8, 0.4, 0.5),
    (0.4, 0.4, 0.5),
    (1.1, 0.7, 0.5),
    (0.6, 1.2, 0.5),
    (1.3, 0.3, 0.5),
    (0.7, 0.7, 0.5),
    (0.3, 0.8, 0.5),
    (1.6, 0.6, 0.5),
]


# ============================================================================
# Reference difficulty configuration
# ============================================================================
#
# Tuned for a 40m reference arena. For any other arena size, obstacle counts
# are scaled by area
# ((size / 40) ** 2) and goal distances are scaled linearly (size / 40), so
# a 20m arena feels proportionally as cluttered/far as the original 40m one,
# and a 80m arena proportionally as roomy.
# ============================================================================

_REFERENCE_ARENA_SIZE = 40.0

_REFERENCE_DIFFICULTY_CONFIGS = {
    1: {
        "count": (6, 8),
        "min_spacing": (1.0, 2.0),
        "min_goal_dist": 12.0,
        "max_goal_dist": 18.0,
    },
    2: {
        "count": (10, 14),
        "min_spacing": 0.7,
        "min_goal_dist": 18.0,
        "max_goal_dist": 27.0,
    },
    3: {
        "count": (15, 20),
        "min_spacing": 0.4,
        "min_goal_dist": 27.0,
        "max_goal_dist": 36.0,
    },
}


def scaled_difficulty_configs(arena_size: float, pool_size: int = len(OBSTACLE_POOL)) -> dict:
    """
    Build DIFFICULTY_CONFIGS for `arena_size`, scaled from the reference
    40m configuration.

    Obstacle counts scale with arena area, goal distances scale linearly
    with arena side length. min_spacing is left unscaled (it is a robot
    clearance requirement, not an arena-size property). Counts are capped
    at `pool_size` since each map can only use each obstacle identity once.
    """

    linear_scale = arena_size / _REFERENCE_ARENA_SIZE

    area_scale = linear_scale**2

    configs: dict = {}

    for level, base in _REFERENCE_DIFFICULTY_CONFIGS.items():

        count_lo, count_hi = base["count"]

        count_lo = min(pool_size, max(1, round(count_lo * area_scale)))
        count_hi = min(pool_size, max(count_lo, round(count_hi * area_scale)))

        configs[level] = {
            "count": (count_lo, count_hi),
            "min_spacing": base["min_spacing"],
            "min_goal_dist": base["min_goal_dist"] * linear_scale,
            "max_goal_dist": base["max_goal_dist"] * linear_scale,
        }

    return configs


# ============================================================================
# Clearance constants (size-independent)
# ============================================================================

OBSTACLE_WALL_MARGIN = 0.3

START_GOAL_OBS_MARGIN = 0.5

START_GOAL_WALL_MARGIN = 0.5

BFS_INFLATION_CELLS = 1

GO2_STANDING_HEIGHT = 0.34

OBSTACLE_Z_OFFSET = 0.25


# ============================================================================
# Generation attempt limits
# ============================================================================

MAX_OBSTACLE_ATTEMPTS = 200

MAX_START_GOAL_ATTEMPTS = 200


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class ObstacleSpec:
    """Single obstacle specification."""

    position: np.ndarray
    # (3,) world frame, z = obstacle_height / 2

    size: np.ndarray
    # (3,) = (size_x, size_y, height)

    radius: float
    # Circumscribing radius used for spacing checks.

    pool_index: int
    # Unique obstacle identity within one map.

    @staticmethod
    def compute_radius(size_x: float, size_y: float) -> float:
        """Compute conservative circumscribing radius (larger horizontal dim)."""

        return max(size_x, size_y) / 2


@dataclass
class MapSpec:
    """Complete map specification for one square arena of any size."""

    arena: ArenaSpec

    obstacles: list[ObstacleSpec]

    obstacle_count: int

    occupancy_grid: np.ndarray
    # (arena.cells, arena.cells) uint8, 1 = occupied, 0 = free

    start_position: np.ndarray
    # (3,) world frame

    goal_position: np.ndarray
    # (3,) world frame

    start_grid: tuple[int, int]
    # (row, col)

    goal_grid: tuple[int, int]
    # (row, col)

    reachable: bool

    difficulty: int

    seed: int


# ============================================================================
# Sampling helpers
# ============================================================================


def _sample_count(rng: np.random.Generator, config: dict) -> int:

    count = config["count"]

    if isinstance(count, tuple):
        low, high = count

        return int(rng.integers(low, high + 1))

    return int(count)


def _sample_min_spacing(rng: np.random.Generator, config: dict) -> float:

    spacing = config["min_spacing"]

    if isinstance(spacing, tuple):
        low, high = spacing

        return float(rng.uniform(low, high))

    return float(spacing)


# ============================================================================
# Universal random map generator
# ============================================================================


class UniversalRandomMapGenerator:
    """
    Generates collision-checked, reachability-validated navigation maps for
    a square arena of any size.

    Example:
        gen = UniversalRandomMapGenerator(arena_size=60.0)
        map_spec = gen.generate_with_retry(difficulty=2, seed=42)
    """

    def __init__(
        self,
        arena_size: float,
        resolution: float = 0.2,
        obstacle_pool: list[tuple[float, float, float]] = OBSTACLE_POOL,
        difficulty_configs: dict | None = None,
    ):

        self.arena = ArenaSpec(size=arena_size, resolution=resolution)

        self.obstacle_pool = obstacle_pool

        if difficulty_configs is not None:
            # Normalize keys to int. Config sources that pass through Isaac Lab's
            # configclass tree (e.g. an EventTermCfg params dict) can't use int
            # dict keys — configclass's to_dict()/class_to_dict() assumes every
            # dict key is a str (it calls key.startswith("__") unconditionally)
            # and raises AttributeError on an int key. So difficulty_configs is
            # accepted with either int or str keys ({1: ...} or {"1": ...}) and
            # normalized to int here, where it's actually indexed by difficulty.
            difficulty_configs = {int(k): v for k, v in difficulty_configs.items()}

        self.difficulty_configs = (
            difficulty_configs
            if difficulty_configs is not None
            else scaled_difficulty_configs(arena_size)
        )

        # Keep obstacle centers at least 0.5m away from the wall.
        self._obstacle_center_bound = self.arena.half - 0.5

    # ------------------------------------------------------------------------
    # Generate one map
    # ------------------------------------------------------------------------

    def generate(
        self,
        difficulty: int,
        seed: int,
    ) -> MapSpec | None:

        if difficulty not in self.difficulty_configs:
            raise ValueError(f"unknown difficulty={difficulty}")

        arena = self.arena

        config = self.difficulty_configs[difficulty]

        rng = np.random.default_rng(seed)

        obstacle_count = _sample_count(rng, config)

        min_spacing = _sample_min_spacing(rng, config)

        if obstacle_count > len(self.obstacle_pool):
            return None

        used_pool_indices: set[int] = set()

        obstacles: list[ObstacleSpec] = []

        # ====================================================================
        # Place obstacles
        # ====================================================================

        for _ in range(obstacle_count):

            placed = False

            for _attempt in range(MAX_OBSTACLE_ATTEMPTS):

                available_indices = [
                    idx
                    for idx in range(len(self.obstacle_pool))
                    if idx not in used_pool_indices
                ]

                if not available_indices:
                    break

                pool_idx = int(rng.choice(available_indices))

                size_x, size_y, height = self.obstacle_pool[pool_idx]

                radius = ObstacleSpec.compute_radius(size_x, size_y)

                x_lo = max(
                    -arena.half + OBSTACLE_WALL_MARGIN + size_x / 2,
                    -self._obstacle_center_bound,
                )

                x_hi = min(
                    arena.half - OBSTACLE_WALL_MARGIN - size_x / 2,
                    self._obstacle_center_bound,
                )

                y_lo = max(
                    -arena.half + OBSTACLE_WALL_MARGIN + size_y / 2,
                    -self._obstacle_center_bound,
                )

                y_hi = min(
                    arena.half - OBSTACLE_WALL_MARGIN - size_y / 2,
                    self._obstacle_center_bound,
                )

                if x_lo > x_hi or y_lo > y_hi:
                    # Arena too small for this obstacle footprint.
                    continue

                x = float(rng.uniform(x_lo, x_hi))

                y = float(rng.uniform(y_lo, y_hi))

                ok = True

                for other in obstacles:

                    ox = other.position[0]
                    oy = other.position[1]

                    center_dist = math.hypot(x - ox, y - oy)

                    edge_dist = center_dist - radius - other.radius

                    if edge_dist < min_spacing:
                        ok = False
                        break

                if ok:

                    used_pool_indices.add(pool_idx)

                    obstacles.append(
                        ObstacleSpec(
                            position=np.array([x, y, height / 2], dtype=float),
                            size=np.array([size_x, size_y, height], dtype=float),
                            radius=radius,
                            pool_index=pool_idx,
                        )
                    )

                    placed = True

                    break

            if not placed:
                return None

        # ====================================================================
        # Validate unique obstacle identities / positions
        # ====================================================================

        assert len({obs.pool_index for obs in obstacles}) == len(obstacles)

        for obs in obstacles:
            assert -self._obstacle_center_bound <= obs.position[0] <= self._obstacle_center_bound
            assert -self._obstacle_center_bound <= obs.position[1] <= self._obstacle_center_bound

        # ====================================================================
        # Build occupancy grid
        # ====================================================================

        grid = np.zeros((arena.cells, arena.cells), dtype=np.uint8)

        mark_walls(arena, grid)

        for obs in obstacles:
            mark_obstacle(arena, grid, obs.position[0], obs.position[1], obs.size[0], obs.size[1])

        assert grid.shape == (arena.cells, arena.cells)
        assert grid.dtype == np.uint8
        assert grid[0, :].sum() == arena.cells
        assert grid[:, 0].sum() == arena.cells

        # ====================================================================
        # Sample START / GOAL
        # ====================================================================

        start_xy = self._sample_clear_point(rng, grid, obstacles)

        if start_xy is None:
            return None

        start_row, start_col = world_to_grid(arena, *start_xy)

        assert grid[start_row, start_col] == 0

        goal_xy = self._sample_goal_point(
            rng,
            grid,
            obstacles,
            start_xy,
            config["min_goal_dist"],
            config["max_goal_dist"],
        )

        if goal_xy is None:
            return None

        goal_row, goal_col = world_to_grid(arena, *goal_xy)

        assert grid[goal_row, goal_col] == 0

        # ====================================================================
        # Inflate grid for reachability check
        # ====================================================================

        inflated = inflate_grid(grid, cells=BFS_INFLATION_CELLS)

        if inflated[start_row, start_col] != 0:
            return None

        if inflated[goal_row, goal_col] != 0:
            return None

        # ====================================================================
        # BFS reachability
        # ====================================================================

        reachable = bfs_reachable(inflated, (start_row, start_col), (goal_row, goal_col))

        if not reachable:
            return None

        assert inflated[start_row, start_col] == 0
        assert inflated[goal_row, goal_col] == 0

        # ====================================================================
        # Return map specification
        # ====================================================================

        return MapSpec(
            arena=arena,
            obstacles=obstacles,
            obstacle_count=len(obstacles),
            occupancy_grid=grid,
            start_position=np.array([start_xy[0], start_xy[1], GO2_STANDING_HEIGHT], dtype=float),
            goal_position=np.array([goal_xy[0], goal_xy[1], GO2_STANDING_HEIGHT], dtype=float),
            start_grid=(start_row, start_col),
            goal_grid=(goal_row, goal_col),
            reachable=True,
            difficulty=difficulty,
            seed=seed,
        )

    # ========================================================================
    # Start / goal sampling
    # ========================================================================

    @staticmethod
    def _clear_of_obstacles(
        x: float,
        y: float,
        obstacles: list[ObstacleSpec],
    ) -> bool:

        for obs in obstacles:

            ox = obs.position[0]
            oy = obs.position[1]

            dist_to_edge = math.hypot(x - ox, y - oy) - obs.radius

            if dist_to_edge < START_GOAL_OBS_MARGIN:
                return False

        return True

    def _sample_clear_point(
        self,
        rng: np.random.Generator,
        grid: np.ndarray,
        obstacles: list[ObstacleSpec],
    ) -> tuple[float, float] | None:

        arena = self.arena

        bound = arena.half - START_GOAL_WALL_MARGIN

        for _attempt in range(MAX_START_GOAL_ATTEMPTS):

            x = float(rng.uniform(-bound, bound))

            y = float(rng.uniform(-bound, bound))

            row, col = world_to_grid(arena, x, y)

            if grid[row, col] != 0:
                continue

            if not self._clear_of_obstacles(x, y, obstacles):
                continue

            return x, y

        return None

    def _sample_goal_point(
        self,
        rng: np.random.Generator,
        grid: np.ndarray,
        obstacles: list[ObstacleSpec],
        start_xy: tuple[float, float],
        min_goal_dist: float,
        max_goal_dist: float,
    ) -> tuple[float, float] | None:

        arena = self.arena

        bound = arena.half - START_GOAL_WALL_MARGIN

        start_x, start_y = start_xy

        for _attempt in range(MAX_START_GOAL_ATTEMPTS):

            x = float(rng.uniform(-bound, bound))

            y = float(rng.uniform(-bound, bound))

            row, col = world_to_grid(arena, x, y)

            if grid[row, col] != 0:
                continue

            if not self._clear_of_obstacles(x, y, obstacles):
                continue

            dist_to_start = math.hypot(x - start_x, y - start_y)

            if dist_to_start < min_goal_dist:
                continue

            if dist_to_start > max_goal_dist:
                continue

            return x, y

        return None

    # ========================================================================
    # Retry wrapper
    # ========================================================================

    def generate_with_retry(
        self,
        difficulty: int,
        seed: int,
        max_attempts: int = 20,
    ) -> MapSpec:

        for attempt in range(max_attempts):

            result = self.generate(difficulty, seed + attempt)

            if result is not None:
                return result

        raise RuntimeError(
            f"Failed to generate difficulty={difficulty} "
            f"{self.arena.size}m map after {max_attempts} attempts "
            f"starting from seed={seed}"
        )
