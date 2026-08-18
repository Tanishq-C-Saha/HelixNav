"""Standalone random map generator for the HelixNav quadruped navigation project.

No Isaac Lab, torch, ROS, or gymnasium imports — pure numpy + stdlib (plus
scipy for grid dilation via grid_utils.inflate_grid). A* paths are NOT
computed here; the generator only validates reachability via BFS. A*
planning happens later in the RL env at reset time (see astar.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .bfs import bfs_reachable
from .grid_utils import CELLS, HALF, inflate_grid, mark_obstacle, mark_walls, world_to_grid

# ---------------------------------------------------------------------------
# Obstacle size pool
# ---------------------------------------------------------------------------

# Each entry is a distinct obstacle "identity" (size_x, size_y, height).
# Isaac Lab spawns one USD prim per pool entry per env, so a single map may
# not reuse a pool index for two different obstacles — see
# RandomMapGenerator.generate(), which samples pool indices without
# replacement. Sized comfortably above the largest difficulty's obstacle
# count (15, for difficulty 3) so unique sampling never runs out headroom.
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

# ---------------------------------------------------------------------------
# Difficulty configuration
# ---------------------------------------------------------------------------

DIFFICULTY_CONFIGS = {
    1: {"count": 5, "min_spacing": (1.0, 2.0), "min_goal_dist": 4.0, "max_goal_dist": 6.0},
    2: {"count": (8, 10), "min_spacing": 0.7, "min_goal_dist": 7.0, "max_goal_dist": 8.0},
    3: {"count": (12, 13), "min_spacing": 0.4, "min_goal_dist": 9.0, "max_goal_dist": 11.0},
}

# ---------------------------------------------------------------------------
# Clearance constants
# ---------------------------------------------------------------------------

OBSTACLE_WALL_MARGIN = 0.3   # obstacle edge at least 0.3m from wall
START_GOAL_OBS_MARGIN = 0.5  # start/goal at least 0.5m from any obstacle edge
START_GOAL_WALL_MARGIN = 0.5  # start/goal at least 0.5m from wall
BFS_INFLATION_CELLS = 1      # inflate grid by 1 cell before BFS reachability check
GO2_STANDING_HEIGHT = 0.34   # z coordinate for start/goal positions
OBSTACLE_Z_OFFSET = 0.25     # z = height / 2 for obstacles resting on ground

# Hard cap on obstacle center placement, independent of per-size wall margin.
# (The wall-margin formula alone would allow the smallest pool obstacles —
# half-size 0.15m — up to |5.55|, which would violate the [-5.5, 5.5]
# invariant asserted after placement. This cap keeps that invariant true
# for every obstacle in OBSTACLE_POOL.)
_OBSTACLE_CENTER_BOUND = 5.5

MAX_OBSTACLE_ATTEMPTS = 100
MAX_START_GOAL_ATTEMPTS = 50


@dataclass
class ObstacleSpec:
    """Single obstacle specification."""

    position: np.ndarray  # (3,) world frame, z = obstacle_height / 2
    size: np.ndarray  # (3,) (size_x, size_y, height)
    radius: float  # circumscribing circle radius for spacing checks
    pool_index: int  # index into OBSTACLE_POOL; unique within a MapSpec

    @staticmethod
    def compute_radius(size_x: float, size_y: float) -> float:
        """Circumscribing circle radius from bounding box.

        Works for all shapes:
          - Rectangle: radius = max(size_x, size_y) / 2
          - Cylinder/Cone: size_x == size_y, radius = size_x / 2
          - Any shape: use the larger dimension as conservative bound
        """
        return max(size_x, size_y) / 2


@dataclass
class MapSpec:
    """Complete map specification for one environment."""

    obstacles: list[ObstacleSpec]
    obstacle_count: int
    occupancy_grid: np.ndarray  # (61, 61) uint8, 1=occupied, 0=free
    start_position: np.ndarray  # (3,) world frame, z=0.34 (Go2 standing height)
    goal_position: np.ndarray  # (3,) world frame, z=0.34
    start_grid: tuple[int, int]  # (row, col) grid indices of start
    goal_grid: tuple[int, int]  # (row, col) grid indices of goal
    reachable: bool  # BFS validated
    difficulty: int  # 1, 2, or 3
    seed: int  # for reproducibility


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


class RandomMapGenerator:
    """Generates collision-checked, reachability-validated navigation maps."""

    def __init__(self, obstacle_pool: list[tuple[float, float, float]] = OBSTACLE_POOL):
        self.obstacle_pool = obstacle_pool

    def generate(self, difficulty: int, seed: int) -> MapSpec | None:
        """Generate a single map. Returns None if generation fails."""
        assert difficulty in DIFFICULTY_CONFIGS, f"unknown difficulty {difficulty}"
        config = DIFFICULTY_CONFIGS[difficulty]
        rng = np.random.default_rng(seed)

        obstacle_count = _sample_count(rng, config)
        min_spacing = _sample_min_spacing(rng, config)

        if obstacle_count > len(self.obstacle_pool):
            # Can't place more obstacles than there are unique pool entries
            # (each entry is a distinct spawnable asset — no duplicates
            # within one env). Caller retries with a different seed.
            return None

        used_pool_indices: set[int] = set()
        obstacles: list[ObstacleSpec] = []
        for _ in range(obstacle_count):
            placed = False
            for _attempt in range(MAX_OBSTACLE_ATTEMPTS):
                available_indices = [
                    idx for idx in range(len(self.obstacle_pool)) if idx not in used_pool_indices
                ]
                if not available_indices:
                    break  # pool exhausted; this obstacle can't be placed uniquely

                pool_idx = int(rng.choice(available_indices))
                size_x, size_y, height = self.obstacle_pool[pool_idx]
                radius = ObstacleSpec.compute_radius(size_x, size_y)

                x_lo = max(-HALF + OBSTACLE_WALL_MARGIN + size_x / 2, -_OBSTACLE_CENTER_BOUND)
                x_hi = min(HALF - OBSTACLE_WALL_MARGIN - size_x / 2, _OBSTACLE_CENTER_BOUND)
                y_lo = max(-HALF + OBSTACLE_WALL_MARGIN + size_y / 2, -_OBSTACLE_CENTER_BOUND)
                y_hi = min(HALF - OBSTACLE_WALL_MARGIN - size_y / 2, _OBSTACLE_CENTER_BOUND)

                x = float(rng.uniform(x_lo, x_hi))
                y = float(rng.uniform(y_lo, y_hi))

                ok = True
                for other in obstacles:
                    ox, oy = other.position[0], other.position[1]
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

        assert len({obs.pool_index for obs in obstacles}) == len(obstacles), (
            "pool indices must be unique within a single map"
        )

        for obs in obstacles:
            assert -5.5 <= obs.position[0] <= 5.5
            assert -5.5 <= obs.position[1] <= 5.5

        grid = np.zeros((CELLS, CELLS), dtype=np.uint8)
        mark_walls(grid)
        for obs in obstacles:
            mark_obstacle(grid, obs.position[0], obs.position[1], obs.size[0], obs.size[1])
        assert grid.shape == (CELLS, CELLS) and grid.dtype == np.uint8
        assert grid[0, :].sum() == CELLS and grid[:, 0].sum() == CELLS

        start_xy = self._sample_clear_point(rng, grid, obstacles)
        if start_xy is None:
            return None
        start_row, start_col = world_to_grid(*start_xy)
        assert grid[start_row, start_col] == 0

        goal_xy = self._sample_goal_point(
            rng, grid, obstacles, start_xy, config["min_goal_dist"], config["max_goal_dist"]
        )
        if goal_xy is None:
            return None
        goal_row, goal_col = world_to_grid(*goal_xy)
        assert grid[goal_row, goal_col] == 0

        inflated = inflate_grid(grid, cells=BFS_INFLATION_CELLS)
        if inflated[start_row, start_col] != 0 or inflated[goal_row, goal_col] != 0:
            return None

        reachable = bfs_reachable(inflated, (start_row, start_col), (goal_row, goal_col))
        if not reachable:
            return None
        assert inflated[start_row, start_col] == 0 and inflated[goal_row, goal_col] == 0

        return MapSpec(
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

    @staticmethod
    def _clear_of_obstacles(x: float, y: float, obstacles: list[ObstacleSpec]) -> bool:
        for obs in obstacles:
            ox, oy = obs.position[0], obs.position[1]
            dist_to_edge = math.hypot(x - ox, y - oy) - obs.radius
            if dist_to_edge < START_GOAL_OBS_MARGIN:
                return False
        return True

    def _sample_clear_point(
        self, rng: np.random.Generator, grid: np.ndarray, obstacles: list[ObstacleSpec]
    ) -> tuple[float, float] | None:
        bound = HALF - START_GOAL_WALL_MARGIN
        for _attempt in range(MAX_START_GOAL_ATTEMPTS):
            x = float(rng.uniform(-bound, bound))
            y = float(rng.uniform(-bound, bound))
            row, col = world_to_grid(x, y)
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
        bound = HALF - START_GOAL_WALL_MARGIN
        start_x, start_y = start_xy
        for _attempt in range(MAX_START_GOAL_ATTEMPTS):
            x = float(rng.uniform(-bound, bound))
            y = float(rng.uniform(-bound, bound))
            row, col = world_to_grid(x, y)
            if grid[row, col] != 0:
                continue
            if not self._clear_of_obstacles(x, y, obstacles):
                continue
            dist_to_start = math.hypot(x - start_x, y - start_y)
            if dist_to_start < min_goal_dist or dist_to_start > max_goal_dist:
                continue
            return x, y
        return None

    def generate_with_retry(self, difficulty: int, seed: int,
                            max_attempts: int = 20) -> MapSpec:
        """Try seeds seed, seed+1, ... until a valid map is generated."""
        for attempt in range(max_attempts):
            result = self.generate(difficulty, seed + attempt)
            if result is not None:
                return result
        raise RuntimeError(
            f"Failed to generate difficulty={difficulty} map after "
            f"{max_attempts} attempts starting from seed={seed}"
        )
