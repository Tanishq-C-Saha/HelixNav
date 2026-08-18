"""8-connected BFS flood fill for grid reachability checks."""

from __future__ import annotations

from collections import deque

import numpy as np

_NEIGHBOR_OFFSETS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
]


def bfs_reachable(grid: np.ndarray, start_rc: tuple[int, int],
                  goal_rc: tuple[int, int]) -> bool:
    """8-connected BFS flood fill. Returns True if goal reachable from start."""
    height, width = grid.shape
    start_rc = (int(start_rc[0]), int(start_rc[1]))
    goal_rc = (int(goal_rc[0]), int(goal_rc[1]))

    if start_rc == goal_rc:
        return True

    visited = {start_rc}
    queue = deque([start_rc])

    while queue:
        row, col = queue.popleft()
        for dr, dc in _NEIGHBOR_OFFSETS:
            nr, nc = row + dr, col + dc
            if nr < 0 or nr >= height or nc < 0 or nc >= width:
                continue
            if grid[nr, nc] != 0:
                continue
            neighbor = (nr, nc)
            if neighbor in visited:
                continue
            if neighbor == goal_rc:
                return True
            visited.add(neighbor)
            queue.append(neighbor)

    return False
