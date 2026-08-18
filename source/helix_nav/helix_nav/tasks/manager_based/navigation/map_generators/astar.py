"""A* pathfinding on an 8-connected occupancy grid.

Used by the RL env at reset time (not by the map generator, which only
validates reachability via bfs.py). Connectivity intentionally matches
bfs.py exactly (plain 8-connected, no corner-cutting rule) so that any
map bfs_reachable() accepts is guaranteed to have an A* path.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

# 8-connected neighbor offsets (dr, dc) and their step costs.
_NEIGHBOR_OFFSETS = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)),
    (1, 1, math.sqrt(2)),
]


def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Euclidean-distance heuristic (admissible for 8-connected grids with
    diagonal cost sqrt(2))."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def astar(grid: np.ndarray, start_rc: tuple[int, int],
          goal_rc: tuple[int, int]) -> np.ndarray | None:
    """A* on 8-connected grid. Returns (P, 2) int array of (row, col) or None."""
    height, width = grid.shape
    start_rc = (int(start_rc[0]), int(start_rc[1]))
    goal_rc = (int(goal_rc[0]), int(goal_rc[1]))

    assert grid[start_rc[0], start_rc[1]] == 0, "start cell must be free"
    assert grid[goal_rc[0], goal_rc[1]] == 0, "goal cell must be free"

    if start_rc == goal_rc:
        return np.array([start_rc], dtype=int)

    g_score = {start_rc: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    open_heap: list[tuple[float, tuple[int, int]]] = [
        (_heuristic(start_rc, goal_rc), start_rc)
    ]
    closed = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal_rc:
            path = [current]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            path_arr = np.array(path, dtype=int)
            assert tuple(path_arr[0]) == start_rc and tuple(path_arr[-1]) == goal_rc
            assert all(grid[r, c] == 0 for r, c in path_arr), "path crosses occupied cell"
            return path_arr

        closed.add(current)
        row, col = current

        for dr, dc, step_cost in _NEIGHBOR_OFFSETS:
            nr, nc = row + dr, col + dc
            if nr < 0 or nr >= height or nc < 0 or nc >= width:
                continue
            if grid[nr, nc] != 0:
                continue
            neighbor = (nr, nc)
            if neighbor in closed:
                continue
            tentative_g = g_score[current] + step_cost
            if tentative_g < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + _heuristic(neighbor, goal_rc)
                heapq.heappush(open_heap, (f_score, neighbor))

    return None
