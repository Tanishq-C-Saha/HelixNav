"""Standalone visualizer for MapSpec produced by any arena size.

Import `plot_map_spec` (or `visualize_map_spec`) into your own script to draw
one MapSpec (from random_map_gen_universal.py or random_map_gen_40m.py) —
obstacles, walls, A* path, start, and goal. A* is computed here purely for
plotting (on an inflated copy of map_spec.occupancy_grid); it is never read
from or written back into the MapSpec.

Usage as a library:

    from visualize_large_map import plot_map_spec
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plot_map_spec(map_spec, ax=ax)
    plt.show()

Usage as a CLI (generates a fresh map for any square arena size):

    python3 visualize_large_map.py --arena-size 60 --difficulty 2 --seed 42
    python3 visualize_large_map.py --arena-size 40 --difficulty 3 --seed 777 --out output/large.png
    python3 visualize_large_map.py --pickle output/mapspec_d2_s42.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Rectangle

try:
    from ..astar import astar
    from .grid_utils_universal import grid_to_world, inflate_grid
    from .random_map_gen_universal import BFS_INFLATION_CELLS, MapSpec, UniversalRandomMapGenerator
except ImportError:
    # Allow running this file directly (e.g. `python3 visualize_large_map.py`)
    # without the full helix_nav/Isaac Lab package being importable.
    import sys

    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _THIS_DIR)
    sys.path.insert(0, os.path.dirname(_THIS_DIR))

    from astar import astar
    from grid_utils_universal import grid_to_world, inflate_grid
    from random_map_gen_universal import BFS_INFLATION_CELLS, MapSpec, UniversalRandomMapGenerator


def _path_length_meters(path_world: np.ndarray) -> float:
    """Total Euclidean length of a world-frame polyline, in meters."""
    if len(path_world) < 2:
        return 0.0
    deltas = np.diff(path_world, axis=0)
    return float(np.sum(np.hypot(deltas[:, 0], deltas[:, 1])))


def plot_map_spec(
    map_spec: MapSpec,
    ax: Axes | None = None,
    show_spacing_circles: bool = True,
    show_astar: bool = True,
) -> Axes:
    """Draw one MapSpec (obstacles, walls, A* path, start, goal) onto `ax`.

    Creates a new figure/axes if `ax` is None. Returns the axes drawn on.
    """
    arena = map_spec.arena

    if ax is None:
        _fig, ax = plt.subplots(figsize=(7, 7))

    for obs in map_spec.obstacles:
        x, y, _z = obs.position
        size_x, size_y, _height = obs.size
        ax.add_patch(
            Rectangle(
                (x - size_x / 2, y - size_y / 2),
                size_x,
                size_y,
                facecolor="#CCCCCC",
                edgecolor="black",
                linewidth=1.0,
                zorder=2,
            )
        )
        if show_spacing_circles:
            ax.add_patch(
                Circle(
                    (x, y),
                    obs.radius,
                    fill=False,
                    edgecolor="blue",
                    linestyle="--",
                    alpha=0.3,
                    zorder=1,
                )
            )

    ax.add_patch(
        Rectangle(
            (-arena.half, -arena.half),
            2 * arena.half,
            2 * arena.half,
            fill=False,
            edgecolor="black",
            linewidth=3.0,
            zorder=3,
        )
    )

    path_len = 0.0
    if show_astar:
        inflated = inflate_grid(map_spec.occupancy_grid, cells=BFS_INFLATION_CELLS)
        path_grid = astar(inflated, map_spec.start_grid, map_spec.goal_grid)
        if path_grid is not None:
            path_world = np.array(
                [grid_to_world(arena, r, c) for r, c in path_grid], dtype=float
            )
            ax.plot(path_world[:, 0], path_world[:, 1], "b-", linewidth=1.5, zorder=4)
            ax.plot(path_world[:, 0], path_world[:, 1], "b.", markersize=4, zorder=4)
            path_len = _path_length_meters(path_world)

    ax.scatter(*map_spec.start_position[:2], c="green", s=150, marker="o", zorder=5, label="Start")
    ax.scatter(*map_spec.goal_position[:2], c="red", s=150, marker="x", zorder=5, label="Goal")

    view_bound = arena.half + 1.0
    ax.set_xlim(-view_bound, view_bound)
    ax.set_ylim(-view_bound, view_bound)
    tick_step = max(1.0, round(arena.size / 20.0))
    ax.set_xticks(np.arange(-arena.half, arena.half + 0.01, tick_step))
    ax.set_yticks(np.arange(-arena.half, arena.half + 0.01, tick_step))
    ax.grid(True, alpha=0.15)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        f"{arena.size:g}m | D{map_spec.difficulty} | Seed {map_spec.seed} | "
        f"{map_spec.obstacle_count} obs | Path: {path_len:.1f}m",
        fontsize=10,
    )

    return ax


def visualize_map_spec(
    map_spec: MapSpec,
    save_path: str | None = None,
    show: bool = False,
    dpi: int = 150,
) -> None:
    """Convenience wrapper: build a fresh figure, plot map_spec, save/show it."""
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_map_spec(map_spec, ax=ax)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a MapSpec for any square arena size.")
    parser.add_argument("--pickle", type=str, default=None, help="Path to a pickled MapSpec.")
    parser.add_argument(
        "--arena-size", type=float, default=40.0, help="Generate fresh: square arena side length in meters."
    )
    parser.add_argument("--difficulty", type=int, default=None, help="Generate fresh: difficulty 1-3.")
    parser.add_argument("--seed", type=int, default=None, help="Generate fresh: seed.")
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "large_map.png"),
        help="Output PNG path.",
    )
    parser.add_argument("--show", action="store_true", help="Also open an interactive window.")
    args = parser.parse_args()

    if args.pickle is not None:
        with open(args.pickle, "rb") as f:
            map_spec = pickle.load(f)
    elif args.difficulty is not None and args.seed is not None:
        generator = UniversalRandomMapGenerator(arena_size=args.arena_size)
        map_spec = generator.generate_with_retry(args.difficulty, args.seed)
    else:
        parser.error("Provide either --pickle, or both --difficulty and --seed.")
        return

    visualize_map_spec(map_spec, save_path=args.out, show=args.show)
    print(f"OK — saved to {args.out}")


if __name__ == "__main__":
    _main()
