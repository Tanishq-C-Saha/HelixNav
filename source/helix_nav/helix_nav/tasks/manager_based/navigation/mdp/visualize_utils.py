from isaaclab.envs import ManagerBasedEnv


import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Rectangle



RES = 0.2
HALF = 6.0
GRID_CELLS = 61

# visualizing the plot 
def plot_nav_state(
    occupancy_grid: np.ndarray,
    path_world: np.ndarray | None = None,
    start_world: np.ndarray | None = None,
    goal_world: np.ndarray | None = None,
    trajectory_world: np.ndarray | None = None,
    title: str = "Navigation State",
    save_path: str | None = None,
    dpi: int = 150,
) -> None:
    """Plot occupancy grid with path, start, goal, and (optionally) a travelled trajectory overlaid.

    Args:
        occupancy_grid: (61, 61) array, 1=occupied, 0=free
        path_world: (P, 2) array of (x, y) world coords, or None. Drawn as the planned A* path.
        start_world: (2,) or (3,) array of (x, y, [z]) world coords, or None
        goal_world: (2,) or (3,) array of (x, y, [z]) world coords, or None
        trajectory_world: (T, 2) array of (x, y) world coords, or None. Drawn as the actual
            path the agent travelled (e.g. recorded during play) — a separate line from the
            planned A* path so the two can be visually compared.
        title: plot title
        save_path: if provided, saves PNG to this path
        dpi: save resolution
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    # Derive the arena half-size from the grid actually passed in, rather
    # than the module-level HALF/GRID_CELLS constants (12m/61 cells) — those
    # only match the small RandomMapGenerator. UniversalRandomMapGenerator
    # (or any other size) produces a differently-shaped occupancy_grid, and
    # this keeps the drawn walls/extent/axes matching whatever arena the map
    # actually came from instead of always drawing a fixed 12m box.
    cells = occupancy_grid.shape[0]
    half = (cells - 1) / 2 * RES

    # Occupancy grid as image
    # grid[row, col] where row=Y, col=X
    # imshow expects (row, col) with row 0 at top, so flip vertically
    ax.imshow(
        occupancy_grid[::-1],
        extent=[-half, half, -half, half],
        cmap="Greys",
        alpha=0.6,
        zorder=1,
    )

    # Arena walls
    ax.add_patch(
        Rectangle(
            (-half, -half), 2 * half, 2 * half,
            fill=False, edgecolor="black", linewidth=3.0, zorder=3,
        )
    )

    # A* path
    if path_world is not None and len(path_world) > 0:
        ax.plot(path_world[:, 0], path_world[:, 1], "b-", linewidth=1.5, zorder=4, label="Planned (A*)")
        ax.plot(path_world[:, 0], path_world[:, 1], "b.", markersize=4, zorder=4)

    # Actual travelled trajectory
    if trajectory_world is not None and len(trajectory_world) > 0:
        ax.plot(
            trajectory_world[:, 0], trajectory_world[:, 1],
            "-", color="darkorange", linewidth=2.0, zorder=6, label="Travelled",
        )

    # Start
    if start_world is not None:
        ax.scatter(start_world[0], start_world[1],
                   c="green", s=200, marker="o", zorder=5, label="Start")

    # Goal
    if goal_world is not None:
        ax.scatter(goal_world[0], goal_world[1],
                   c="red", s=200, marker="x", zorder=5, label="Goal")

    view_bound = half + 1.0
    tick_step = max(1.0, round(2 * half / 20.0))
    ax.set_xlim(-view_bound, view_bound)
    ax.set_ylim(-view_bound, view_bound)
    ax.set_xticks(np.arange(-half, half + 0.01, tick_step))
    ax.set_yticks(np.arange(-half, half + 0.01, tick_step))
    ax.grid(True, alpha=0.15)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    if start_world is not None or goal_world is not None or path_world is not None or trajectory_world is not None:
        ax.legend(loc="upper right")

    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi)
        plt.close(fig)
    else:
        plt.close(fig)


# visualizing the MultiMesh raycaster 
def visualize_nav_states(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    hit_points_w: torch.Tensor,
    min_threshold: float = 0.01,
    max_threshold: float = 6.0,
    output_dir: str = "output",
):
    """Convert raycaster hits to occupancy grids and plot with A* path per env.
    
    Args:
        env: the environment (carries _paths_world(local), _start_positions, etc.)
        env_ids: (N,) tensor of env indices to visualize
        hit_points_w: (num_envs, 3721, 3) raycaster hit points in world frame
        min_threshold: min z to count as occupied
        max_threshold: max z to count as occupied
        output_dir: base directory for saved images
    """
    
    for env_id in env_ids.tolist():
        
        # ── Ray hits → occupancy grid ──
        points_z = hit_points_w[env_id, :, 2]
        hit_mask = (points_z > min_threshold) & (points_z < max_threshold)
        occupancy = torch.zeros_like(points_z)
        occupancy[hit_mask] = 1
        occupancy_grid = occupancy.cpu().numpy().reshape(GRID_CELLS, GRID_CELLS)
        
        # ── Extract path, start, goal from env state ──
        path_len = int(env._path_lengths[env_id].item())
        path_world = None
        if path_len > 0:
            path_world = env._paths_local[env_id, :path_len].cpu().numpy()
        
        start_world = env._start_positions_local[env_id].cpu().numpy()
        goal_world = env._goal_positions_local[env_id].cpu().numpy()
        
        reset_count = int(env._reset_counter[env_id].item())
        seed_used = int(env_id * env._global_seed + reset_count - 1)
        
        # ── Plot ──
        save_path = os.path.join(
            output_dir,
            "occupancy_maps",
            "multi_mesh_ray_caster",
            f"env_{env_id}",
            f"reset_{reset_count - 1:04d}.jpg",
        )
        
        plot_nav_state(
            occupancy_grid=occupancy_grid,
            path_world=path_world,
            start_world=start_world,
            goal_world=goal_world,
            title=f"MultiMeshRayCaster | Env {env_id} | D{env._current_difficulty} | Reset {reset_count - 1} | seed {seed_used}",
            save_path=save_path,
        )


# ── Episode-level metrics (success rate, SPL, path efficiency, reward) ──
# Standard navigation-RL evaluation metrics — see e.g. Anderson et al. 2018,
# "On Evaluation of Embodied Navigation Agents" for SPL (Success weighted by Path Length).

def compute_spl(success: bool, actual_path_length: float, optimal_path_length: float) -> float:
    """Success weighted by (normalized inverse) Path Length.

    SPL = success * optimal_path_length / max(actual_path_length, optimal_path_length, eps).
    0 for a failed episode; 1 for a successful episode that took the optimal path;
    shrinks toward 0 the more the travelled path exceeds the optimal one.
    """
    if not success:
        return 0.0
    denom = max(actual_path_length, optimal_path_length, 1e-6)
    return float(optimal_path_length / denom)


def path_length(points_local: np.ndarray) -> float:
    """Sum of consecutive euclidean distances along an (N, 2) array of (x, y) points."""
    if points_local is None or len(points_local) < 2:
        return 0.0
    deltas = np.diff(points_local, axis=0)
    return float(np.linalg.norm(deltas, axis=-1).sum())


def save_episode_metrics_csv(episode_metrics: list[dict], save_path: str) -> None:
    """Dump one row per episode to a CSV — raw data for supplementary tables/further analysis."""
    import csv

    if not episode_metrics:
        return
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fieldnames = list(episode_metrics[0].keys())
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episode_metrics)


def plot_episode_metrics_dashboard(
    episode_metrics: list[dict],
    title: str = "Episode Metrics",
    save_path: str | None = None,
    dpi: int = 150,
) -> None:
    """Summary dashboard over a batch of play episodes — the metrics typically reported
    for a navigation-RL policy evaluation: outcome rates, SPL, path length (policy vs.
    A*-optimal), path efficiency, time taken, reward, and a per-difficulty breakdown.

    Args:
        episode_metrics: list of per-episode dicts, each with keys "success" (bool),
            "collided" (bool), "timed_out" (bool), "steps" (int), "duration_s" (float),
            "spl" (float), "path_length_actual" (float), "path_length_optimal" (float),
            "cumulative_reward" (float), "difficulty" (int). Extra keys are ignored.
        title: figure title.
        save_path: if provided, saves PNG to this path.
        dpi: save resolution.
    """
    if not episode_metrics:
        return

    n = len(episode_metrics)
    successes = [1 if m["success"] else 0 for m in episode_metrics]
    collisions = [1 if m.get("collided") else 0 for m in episode_metrics]
    timeouts = [1 if m.get("timed_out") else 0 for m in episode_metrics]
    spls = np.array([m["spl"] for m in episode_metrics])
    durations = np.array([m.get("duration_s", m["steps"]) for m in episode_metrics])
    rewards = [m["cumulative_reward"] for m in episode_metrics]
    actual_lens = np.array([m["path_length_actual"] for m in episode_metrics])
    optimal_lens = np.array([m["path_length_optimal"] for m in episode_metrics])
    difficulties = np.array([m.get("difficulty", 0) for m in episode_metrics])

    success_rate = 100.0 * sum(successes) / n
    collision_rate = 100.0 * sum(collisions) / n
    timeout_rate = 100.0 * sum(timeouts) / n
    mean_spl = float(spls.mean())

    # path efficiency ratio: actual / optimal (only meaningful where a path was found;
    # 1.0 = travelled exactly the optimal distance, >1.0 = took a longer route)
    valid = optimal_lens > 1e-6
    efficiency_ratio = np.divide(actual_lens, optimal_lens, out=np.full_like(actual_lens, np.nan), where=valid)

    fig, axes = plt.subplots(3, 3, figsize=(16, 13))

    # ── outcome rates ──
    ax = axes[0, 0]
    outcomes = ["Success", "Collision", "Timeout"]
    rates = [success_rate, collision_rate, timeout_rate]
    colors = ["seagreen", "firebrick", "goldenrod"]
    ax.bar(outcomes, rates, color=colors)
    for i, r in enumerate(rates):
        ax.text(i, r + 1, f"{r:.1f}%", ha="center", va="bottom")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Episodes (%)")
    ax.set_title(f"Outcome rates (n={n})")

    # ── SPL distribution ──
    ax = axes[0, 1]
    ax.hist(spls, bins=np.linspace(0, 1, 11), color="steelblue", edgecolor="white")
    ax.axvline(mean_spl, color="black", linestyle="--", linewidth=1.5, label=f"mean={mean_spl:.3f}")
    ax.set_xlabel("SPL")
    ax.set_ylabel("Episodes")
    ax.set_title("Success weighted by Path Length")
    ax.legend()

    # ── time taken ──
    ax = axes[0, 2]
    ax.hist(durations, bins=20, color="mediumpurple", edgecolor="white")
    ax.axvline(float(durations.mean()), color="black", linestyle="--", linewidth=1.5,
               label=f"mean={durations.mean():.1f}s")
    ax.set_xlabel("Episode duration (s)")
    ax.set_ylabel("Episodes")
    ax.set_title("Time taken")
    ax.legend()

    # ── policy path length vs. A*-optimal path length ──
    ax = axes[1, 0]
    point_colors = ["seagreen" if s else "firebrick" for s in successes]
    ax.scatter(optimal_lens, actual_lens, c=point_colors, s=24, alpha=0.8, zorder=3)
    lim = float(max(actual_lens.max(initial=0), optimal_lens.max(initial=0), 1.0)) * 1.05
    ax.plot([0, lim], [0, lim], color="gray", linestyle="--", linewidth=1.0, zorder=2, label="y = x (optimal)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("A* optimal path length (m)")
    ax.set_ylabel("Policy travelled distance (m)")
    ax.set_title("Distance travelled vs. A* optimal (green=success, red=fail)")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")

    # ── path efficiency ratio (actual / optimal) ──
    ax = axes[1, 1]
    finite_ratio = efficiency_ratio[np.isfinite(efficiency_ratio)]
    if len(finite_ratio) > 0:
        ax.hist(np.clip(finite_ratio, 0, 3), bins=20, color="teal", edgecolor="white")
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="optimal (1.0)")
        ax.axvline(float(finite_ratio.mean()), color="darkorange", linestyle="-", linewidth=1.5,
                   label=f"mean={finite_ratio.mean():.2f}")
        ax.legend()
    ax.set_xlabel("Travelled / optimal distance ratio")
    ax.set_ylabel("Episodes")
    ax.set_title("Path efficiency (clipped at 3x)")

    # ── reward per episode ──
    ax = axes[1, 2]
    ep_idx = np.arange(n)
    ax.scatter(ep_idx, rewards, c=point_colors, s=20, zorder=3)
    ax.plot(ep_idx, rewards, color="gray", alpha=0.4, linewidth=1.0, zorder=2)
    ax.set_xlabel("Episode index")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Reward per episode")

    # ── success rate by difficulty ──
    ax = axes[2, 0]
    diff_levels = sorted(set(difficulties.tolist()))
    diff_success_rates = [
        100.0 * np.mean([s for s, d in zip(successes, difficulties) if d == level]) for level in diff_levels
    ]
    diff_counts = [int(np.sum(difficulties == level)) for level in diff_levels]
    ax.bar([str(d) for d in diff_levels], diff_success_rates, color="seagreen")
    for i, (r, c) in enumerate(zip(diff_success_rates, diff_counts)):
        ax.text(i, r + 1, f"{r:.0f}%\n(n={c})", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 115)
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Success rate by difficulty")

    # ── SPL by difficulty ──
    ax = axes[2, 1]
    diff_spl = [[spl for spl, d in zip(spls.tolist(), difficulties) if d == level] for level in diff_levels]
    ax.boxplot(diff_spl, labels=[str(d) for d in diff_levels])
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("SPL")
    ax.set_title("SPL by difficulty")

    # ── running (cumulative) success rate ──
    ax = axes[2, 2]
    running_success_rate = 100.0 * np.cumsum(successes) / np.arange(1, n + 1)
    ax.plot(ep_idx, running_success_rate, color="seagreen", linewidth=1.5)
    ax.axhline(success_rate, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Episode index")
    ax.set_ylabel("Running success rate (%)")
    ax.set_title("Success rate stability over the evaluation run")

    fig.suptitle(title)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi)
    plt.close(fig)