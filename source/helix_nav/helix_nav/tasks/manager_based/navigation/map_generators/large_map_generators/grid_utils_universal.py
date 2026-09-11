"""
Grid <-> world conversion utilities for a square arena of ANY size.

This is the size-agnostic counterpart to grid_utils_40m.py. That module
stays untouched (and is still what random_map_gen_40m.py uses) — this one
lets random_map_gen_universal.py build maps for any square arena.

Grid convention (same as grid_utils_40m.py, parameterized by `ArenaSpec`):
    - Arena: size x size (meters), centered at the origin
    - World coordinates: [-size/2, +size/2] on both axes
    - Grid: cells x cells points, cells = size / resolution + 1
    - grid[row, col]
    - ROW = Y axis
    - COL = X axis
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Arena spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArenaSpec:
    """Defines the grid geometry of one square arena.

    Args:
        size: full arena side length in meters (e.g. 40.0 for a 40m arena).
        resolution: meters per grid cell.
    """

    size: float
    resolution: float = 0.2

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"arena size must be > 0, got {self.size}")

        if self.resolution <= 0:
            raise ValueError(f"resolution must be > 0, got {self.resolution}")

        intervals = self.size / self.resolution

        if abs(intervals - round(intervals)) > 1e-6:
            raise ValueError(
                f"arena size ({self.size}) must be an integer multiple of "
                f"resolution ({self.resolution})"
            )

    @property
    def half(self) -> float:
        return self.size / 2

    @property
    def cells(self) -> int:
        return int(round(self.size / self.resolution)) + 1

    @property
    def world_min(self) -> float:
        return -self.half

    @property
    def world_max(self) -> float:
        return self.half


# ---------------------------------------------------------------------------
# World <-> grid
# ---------------------------------------------------------------------------


def world_to_grid(arena: ArenaSpec, x: float, y: float) -> tuple[int, int]:
    """
    Convert world position to grid indices.

    Returns:
        (row, col) where row = Y, col = X
    """

    col = int((x + arena.half) / arena.resolution)
    row = int((y + arena.half) / arena.resolution)

    col = max(0, min(arena.cells - 1, col))
    row = max(0, min(arena.cells - 1, row))

    assert 0 <= row < arena.cells
    assert 0 <= col < arena.cells

    return row, col


def grid_to_world(arena: ArenaSpec, row: int, col: int) -> tuple[float, float]:
    """
    Convert grid indices to world position.

    Returns:
        (x, y)
    """

    assert 0 <= row < arena.cells
    assert 0 <= col < arena.cells

    x = -arena.half + col * arena.resolution
    y = -arena.half + row * arena.resolution

    return x, y


# ---------------------------------------------------------------------------
# Obstacle marking
# ---------------------------------------------------------------------------


def mark_obstacle(
    arena: ArenaSpec,
    grid: np.ndarray,
    pos_x: float,
    pos_y: float,
    size_x: float,
    size_y: float,
) -> None:
    """
    Mark all grid points/cells overlapping an axis-aligned bounding box.

    Args:
        arena: arena geometry.
        grid: occupancy grid, shape (arena.cells, arena.cells)
        pos_x: obstacle center X
        pos_y: obstacle center Y
        size_x: obstacle width along X
        size_y: obstacle width along Y
    """

    assert grid.shape == (arena.cells, arena.cells)

    x_min = pos_x - size_x / 2
    x_max = pos_x + size_x / 2

    y_min = pos_y - size_y / 2
    y_max = pos_y + size_y / 2

    col_min = int((x_min + arena.half) / arena.resolution)
    col_max = int((x_max + arena.half) / arena.resolution)

    row_min = int((y_min + arena.half) / arena.resolution)
    row_max = int((y_max + arena.half) / arena.resolution)

    col_min = max(0, min(arena.cells - 1, col_min))
    col_max = max(0, min(arena.cells - 1, col_max))

    row_min = max(0, min(arena.cells - 1, row_min))
    row_max = max(0, min(arena.cells - 1, row_max))

    grid[row_min : row_max + 1, col_min : col_max + 1] = 1


# ---------------------------------------------------------------------------
# Grid inflation
# ---------------------------------------------------------------------------


def inflate_grid(
    grid: np.ndarray,
    cells: int = 1,
) -> np.ndarray:
    """
    Inflate occupied cells by `cells` cells in every direction.

    Size-independent, so it is identical to grid_utils_40m.inflate_grid.

    Returns:
        New uint8 occupancy grid.
    """

    from scipy.ndimage import binary_dilation

    if cells <= 0:
        return grid.copy()

    structure = np.ones(
        (2 * cells + 1, 2 * cells + 1),
        dtype=bool,
    )

    inflated = binary_dilation(
        grid.astype(bool),
        structure=structure,
    )

    return inflated.astype(np.uint8)


# ---------------------------------------------------------------------------
# Arena walls
# ---------------------------------------------------------------------------


def mark_walls(arena: ArenaSpec, grid: np.ndarray) -> None:
    """
    Mark the outer boundary of the arena as occupied.

    Walls are located at x = -half, x = +half, y = -half, y = +half.
    """

    assert grid.shape == (arena.cells, arena.cells)

    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def grid_distance(
    arena: ArenaSpec,
    a: tuple[int, int],
    b: tuple[int, int],
) -> float:
    """
    Euclidean distance between two grid coordinates in meters.
    """

    dr = a[0] - b[0]
    dc = a[1] - b[1]

    return float(np.hypot(dr, dc) * arena.resolution)


def world_distance(
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    """
    Euclidean distance between two world positions in meters.
    """

    return float(
        np.hypot(
            a[0] - b[0],
            a[1] - b[1],
        )
    )


def is_inside_world(arena: ArenaSpec, x: float, y: float) -> bool:
    """
    Check whether a world position is inside the arena.
    """

    return (
        arena.world_min <= x <= arena.world_max
        and arena.world_min <= y <= arena.world_max
    )


def is_valid_grid_position(
    arena: ArenaSpec,
    row: int,
    col: int,
) -> bool:
    """
    Check whether grid coordinates are valid.
    """

    return (
        0 <= row < arena.cells
        and 0 <= col < arena.cells
    )
