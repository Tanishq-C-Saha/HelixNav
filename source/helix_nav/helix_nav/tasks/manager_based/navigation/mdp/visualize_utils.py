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
    title: str = "Navigation State",
    save_path: str | None = None,
    dpi: int = 150,
) -> None:
    """Plot occupancy grid with path, start, and goal overlaid.
    
    Args:
        occupancy_grid: (61, 61) array, 1=occupied, 0=free
        path_world: (P, 2) array of (x, y) world coords, or None
        start_world: (2,) or (3,) array of (x, y, [z]) world coords, or None
        goal_world: (2,) or (3,) array of (x, y, [z]) world coords, or None
        title: plot title
        save_path: if provided, saves PNG to this path
        dpi: save resolution
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    # Occupancy grid as image
    # grid[row, col] where row=Y, col=X
    # imshow expects (row, col) with row 0 at top, so flip vertically
    ax.imshow(
        occupancy_grid[::-1],
        extent=[-HALF, HALF, -HALF, HALF],
        cmap="Greys",
        alpha=0.6,
        zorder=1,
    )

    # Arena walls
    ax.add_patch(
        Rectangle(
            (-HALF, -HALF), 2 * HALF, 2 * HALF,
            fill=False, edgecolor="black", linewidth=3.0, zorder=3,
        )
    )

    # A* path
    if path_world is not None and len(path_world) > 0:
        ax.plot(path_world[:, 0], path_world[:, 1], "b-", linewidth=1.5, zorder=4)
        ax.plot(path_world[:, 0], path_world[:, 1], "b.", markersize=4, zorder=4)

    # Start
    if start_world is not None:
        ax.scatter(start_world[0], start_world[1],
                   c="green", s=200, marker="o", zorder=5, label="Start")

    # Goal
    if goal_world is not None:
        ax.scatter(goal_world[0], goal_world[1],
                   c="red", s=200, marker="x", zorder=5, label="Goal")

    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_xticks(np.arange(-6, 6.01, 1))
    ax.set_yticks(np.arange(-6, 6.01, 1))
    ax.grid(True, alpha=0.15)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    if start_world is not None or goal_world is not None:
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
        env: the environment (carries _paths_world, _start_positions, etc.)
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
            path_world = env._paths_world[env_id, :path_len].cpu().numpy()
        
        start_world = env._start_positions[env_id].cpu().numpy()
        goal_world = env._goal_positions[env_id].cpu().numpy()
        
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