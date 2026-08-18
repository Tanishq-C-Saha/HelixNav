"""Grid <-> world conversion utilities for the HelixNav planning stack.

Grid convention (verified empirically against MultiMeshRayCaster output in
Isaac Lab — DO NOT deviate from this anywhere in the codebase):
  - Arena: 12m x 12m, world coordinates [-6.0, 6.0] on both axes.
  - Grid: 61 x 61 cells (61 points, 60 intervals of 0.2m, -6.0 to +6.0 inclusive).
  - Resolution: 0.2m per cell.
  - grid[row, col] where ROW = Y axis, COL = X axis.
  - Cell i covers [(-6.0 + i*0.2), (-6.0 + (i+1)*0.2)).
  - Last cell (index 60) is the wall boundary point at +6.0.
"""

from __future__ import annotations

import numpy as np

RES = 0.2
HALF = 6.0
CELLS = 61


def world_to_grid(x: float, y: float) -> tuple[int, int]:
    """World position to grid indices. Returns (row, col)."""
    col = int((x + HALF) / RES)
    row = int((y + HALF) / RES)
    col = max(0, min(CELLS - 1, col))
    row = max(0, min(CELLS - 1, row))
    assert 0 <= row <= CELLS - 1 and 0 <= col <= CELLS - 1
    return row, col


def grid_to_world(row: int, col: int) -> tuple[float, float]:
    """Grid indices to world position. Returns (x, y)."""
    x = -HALF + col * RES
    y = -HALF + row * RES
    return x, y


def mark_obstacle(grid: np.ndarray, pos_x: float, pos_y: float,
                  size_x: float, size_y: float) -> None:
    """Mark all grid cells overlapping an axis-aligned bounding box."""
    assert grid.shape == (CELLS, CELLS)
    col_min = max(0, int((pos_x - size_x / 2 + HALF) / RES))
    col_max = min(CELLS - 1, int((pos_x + size_x / 2 + HALF) / RES))
    row_min = max(0, int((pos_y - size_y / 2 + HALF) / RES))
    row_max = min(CELLS - 1, int((pos_y + size_y / 2 + HALF) / RES))
    grid[row_min:row_max + 1, col_min:col_max + 1] = 1


def inflate_grid(grid: np.ndarray, cells: int = 1) -> np.ndarray:
    """Dilate occupied cells by N cells in all directions. Returns new array."""
    from scipy.ndimage import binary_dilation
    struct = np.ones((2 * cells + 1, 2 * cells + 1), dtype=bool)
    return binary_dilation(grid, structure=struct).astype(np.uint8)


def mark_walls(grid: np.ndarray) -> None:
    """Mark grid boundary cells as occupied (walls at ±6.0)."""
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
