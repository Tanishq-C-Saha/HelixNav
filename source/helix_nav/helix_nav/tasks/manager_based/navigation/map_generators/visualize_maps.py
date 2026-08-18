"""Demo script: generate 9 maps (3 difficulties x 3 seeds) and plot them.

Pure numpy/matplotlib, no simulator dependencies. A* is only used here for
visualization (it is NOT stored in MapSpec — the RL env computes it at
reset time). Saves a 3x3 preview grid to output/maps_preview.png, a pickled
MapSpec per map, and prints a summary of generation statistics.
"""

from __future__ import annotations

import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from .astar import astar
from .grid_utils import HALF, grid_to_world, inflate_grid
from .random_map_generator import BFS_INFLATION_CELLS, RandomMapGenerator

DIFFICULTIES = [1, 2, 3]
SEEDS = [42, 123, 777]
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "maps_preview.png")


def _path_length_meters(path_world: np.ndarray) -> float:
    if len(path_world) < 2:
        return 0.0
    deltas = np.diff(path_world, axis=0)
    return float(np.sum(np.hypot(deltas[:, 0], deltas[:, 1])))


def _plot_map(ax: plt.Axes, map_spec, requested_seed: int, retries: int) -> float:
    """Draw one map subplot. Returns the A*-visualized path length in meters."""
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
            (-HALF, -HALF),
            2 * HALF,
            2 * HALF,
            fill=False,
            edgecolor="black",
            linewidth=3.0,
            zorder=3,
        )
    )

    inflated = inflate_grid(map_spec.occupancy_grid, cells=BFS_INFLATION_CELLS)
    path_grid = astar(inflated, map_spec.start_grid, map_spec.goal_grid)
    path_len = 0.0
    if path_grid is not None:
        path_world = np.array([grid_to_world(r, c) for r, c in path_grid], dtype=float)
        ax.plot(path_world[:, 0], path_world[:, 1], "b-", linewidth=1.5, zorder=4)
        ax.plot(path_world[:, 0], path_world[:, 1], "b.", markersize=4, zorder=4)
        path_len = _path_length_meters(path_world)

    ax.scatter(*map_spec.start_position[:2], c="green", s=150, marker="o", zorder=5, label="Start")
    ax.scatter(*map_spec.goal_position[:2], c="red", s=150, marker="x", zorder=5, label="Goal")

    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_xticks(np.arange(-7, 7.01, 1))
    ax.set_yticks(np.arange(-7, 7.01, 1))
    ax.grid(True, alpha=0.15)
    ax.set_aspect("equal")

    retry_note = f" (+{retries} retry)" if retries else ""
    ax.set_title(
        f"D{map_spec.difficulty} | Seed {requested_seed}{retry_note} | "
        f"{map_spec.obstacle_count} obs | Path: {path_len:.1f}m",
        fontsize=9,
    )
    return path_len


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generator = RandomMapGenerator()

    fig, axes = plt.subplots(len(DIFFICULTIES), len(SEEDS), figsize=(15, 16))

    retries_by_difficulty: dict[int, list[int]] = {d: [] for d in DIFFICULTIES}
    obs_count_by_difficulty: dict[int, list[int]] = {d: [] for d in DIFFICULTIES}
    path_len_by_difficulty: dict[int, list[float]] = {d: [] for d in DIFFICULTIES}
    failures: list[str] = []

    start_time = time.perf_counter()
    for row, difficulty in enumerate(DIFFICULTIES):
        for col, seed in enumerate(SEEDS):
            ax = axes[row, col]
            try:
                map_spec = generator.generate_with_retry(difficulty, seed, max_attempts=20)
            except RuntimeError as exc:
                failures.append(f"difficulty={difficulty}, seed={seed}: {exc}")
                ax.text(0.5, 0.5, "GENERATION FAILED", ha="center", va="center", color="red")
                ax.set_xlim(-7, 7)
                ax.set_ylim(-7, 7)
                ax.set_aspect("equal")
                continue

            retries = map_spec.seed - seed
            path_len = _plot_map(ax, map_spec, seed, retries)

            retries_by_difficulty[difficulty].append(retries)
            obs_count_by_difficulty[difficulty].append(map_spec.obstacle_count)
            path_len_by_difficulty[difficulty].append(path_len)

            pickle_path = os.path.join(OUTPUT_DIR, f"mapspec_d{difficulty}_s{seed}.pkl")
            with open(pickle_path, "wb") as f:
                pickle.dump(map_spec, f)
    total_time = time.perf_counter() - start_time

    fig.tight_layout(rect=(0, 0.08, 1, 1))

    summary_lines = [f"Total generation time: {total_time:.3f}s"]
    for difficulty in DIFFICULTIES:
        retries = retries_by_difficulty[difficulty]
        obs_counts = obs_count_by_difficulty[difficulty]
        path_lens = path_len_by_difficulty[difficulty]
        if retries:
            summary_lines.append(
                f"D{difficulty}: avg retries={np.mean(retries):.2f}  "
                f"avg obs={np.mean(obs_counts):.1f}  "
                f"avg path={np.mean(path_lens):.2f}m"
            )
        else:
            summary_lines.append(f"D{difficulty}: no successful maps")
    if failures:
        summary_lines.append("Failed generations: " + "; ".join(failures))
    else:
        summary_lines.append("Failed generations: none")

    fig.text(0.5, 0.02, "\n".join(summary_lines), ha="center", va="bottom",
              fontsize=9, family="monospace")

    fig.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)

    for line in summary_lines:
        print(line)

    if failures:
        raise RuntimeError(f"{len(failures)} map(s) failed to generate: {failures}")

    print(f"OK — maps_preview.png saved to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
