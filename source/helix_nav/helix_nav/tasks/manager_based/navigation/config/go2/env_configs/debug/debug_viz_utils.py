"""Reusable visualization utilities for HelixNav debug scripts."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def save_images_grid(
    images: list[torch.Tensor],
    cmap: str | None = "viridis",
    nrow: int = 1,
    subtitles: list[str] | None = None,
    title: str | None = None,
    filename: str | None = None,
):
    """Save images in a clean, professional grid."""

    n_images = len(images)
    ncol = int(np.ceil(n_images / nrow))

    fig, axes = plt.subplots(
        nrow, ncol,
        figsize=(4.0 * ncol, 3.2 * nrow),
        squeeze=False,
    )
    axes = axes.flatten()

    for idx, (img, ax) in enumerate(zip(images, axes)):
        img = img.detach().cpu().numpy()

        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]

        ax.imshow(img, cmap=cmap, interpolation="nearest", aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color("0.25")

        if subtitles and idx < len(subtitles):
            ax.set_title(subtitles[idx], fontsize=11, fontweight="medium", pad=8)

    for ax in axes[n_images:]:
        fig.delaxes(ax)

    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    fig.tight_layout(rect=(0, 0, 1, 0.94 if title else 1))

    if filename:
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(filename, dpi=200, bbox_inches="tight", pad_inches=0.15, facecolor="white")

    plt.close(fig)


def save_occupancy_map(
    hit_points_w: torch.Tensor,
    min_threshold: float,
    max_threshold: float,
    filename: str = "occupancy.jpg",
):
    """Render a 61x61 global occupancy grid from raycaster hit points and save to disk."""

    points = hit_points_w[0]
    points_z = points[:, 2]

    hit_mask = (points_z > min_threshold) & (points_z < max_threshold)
    occupancy = torch.zeros_like(points_z)
    occupancy[hit_mask] = 1

    occupancy_grid = occupancy.detach().cpu().numpy().reshape(61, 61)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(
        occupancy_grid,
        origin="lower",
        extent=[-6, 6, -6, 6],
        interpolation="nearest",
        cmap="Greys",
        vmin=0, vmax=1,
    )

    ax.set_xticks(np.arange(-6, 6.1, 1.0))
    ax.set_yticks(np.arange(-6, 6.1, 1.0))
    ax.set_xticks(np.arange(-6, 6.01, 0.2), minor=True)
    ax.set_yticks(np.arange(-6, 6.01, 0.2), minor=True)

    ax.grid(which="major", linewidth=0.8, alpha=0.35)
    ax.grid(which="minor", linewidth=0.25, alpha=0.25)

    ax.scatter(0, 0, marker="x", s=100, linewidths=2, label="Sensor")
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title("Occupancy Grid — MultiMeshRayCaster")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")

    fig.tight_layout()

    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)