from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_rgb_and_depth(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    normalized_depth: np.ndarray,
    save_path: str | Path | None = None,
    title: str = "RGB-D sample",
) -> None:
    """
    Displays and optionally saves RGB image, depth in meters and normalized depth.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb)
    axes[0].set_title("RGB image")
    axes[0].axis("off")

    depth_img = axes[1].imshow(depth_m, cmap="viridis")
    axes[1].set_title("Depth map, meters")
    axes[1].axis("off")
    fig.colorbar(depth_img, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(normalized_depth, cmap="gray")
    axes[2].set_title("Normalized depth")
    axes[2].axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


def plot_point_cloud_projections(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    save_path: str | Path | None = None,
    title: str = "Point cloud projections",
    max_points: int = 30000,
) -> None:
    """
    Plots three 2D projections of a 3D point cloud:
    - X-Z: top-like projection;
    - X-Y: front projection;
    - Z-Y: side projection.

    This is useful for saving point cloud visualization as a static image.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")

    if points.shape[0] == 0:
        raise ValueError("Point cloud is empty")

    if colors is not None:
        if colors.ndim != 2 or colors.shape[1] != 3:
            raise ValueError(f"Expected colors with shape (N, 3), got {colors.shape}")
        if colors.shape[0] != points.shape[0]:
            raise ValueError("points and colors must have the same number of rows")

    num_points = points.shape[0]

    if num_points > max_points:
        indices = np.random.choice(num_points, size=max_points, replace=False)
        points_to_plot = points[indices]
        colors_to_plot = colors[indices] if colors is not None else None
    else:
        points_to_plot = points
        colors_to_plot = colors

    if colors_to_plot is None:
        point_colors = None
    else:
        point_colors = np.clip(colors_to_plot, 0.0, 1.0)

    x = points_to_plot[:, 0]
    y = points_to_plot[:, 1]
    z = points_to_plot[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(x, z, s=1, c=point_colors)
    axes[0].set_title("X-Z projection")
    axes[0].set_xlabel("X, meters")
    axes[0].set_ylabel("Z, meters")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(x, -y, s=1, c=point_colors)
    axes[1].set_title("X-Y projection")
    axes[1].set_xlabel("X, meters")
    axes[1].set_ylabel("-Y, meters")
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(z, -y, s=1, c=point_colors)
    axes[2].set_title("Z-Y projection")
    axes[2].set_xlabel("Z, meters")
    axes[2].set_ylabel("-Y, meters")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()

def plot_floor_segmentation_projections(
    floor_points: np.ndarray,
    non_floor_points: np.ndarray,
    save_path: str | Path | None = None,
    title: str = "Floor segmentation projections",
    max_points_per_class: int = 30000,
) -> None:
    """
    Plots point cloud projections with separated floor and non-floor points.
    Floor points and non-floor points are shown as two different groups.

    Coordinate visualization is consistent with plot_point_cloud_projections:
    - X-Z projection;
    - X-(-Y) projection;
    - Z-(-Y) projection.
    """
    if floor_points.ndim != 2 or floor_points.shape[1] != 3:
        raise ValueError(f"Expected floor_points with shape (N, 3), got {floor_points.shape}")

    if non_floor_points.ndim != 2 or non_floor_points.shape[1] != 3:
        raise ValueError(
            f"Expected non_floor_points with shape (N, 3), got {non_floor_points.shape}"
        )

    def sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
        if points.shape[0] > max_points:
            indices = np.random.choice(points.shape[0], size=max_points, replace=False)
            return points[indices]
        return points

    floor_plot = sample_points(floor_points, max_points_per_class)
    non_floor_plot = sample_points(non_floor_points, max_points_per_class)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # X-Z projection
    axes[0].scatter(
        non_floor_plot[:, 0],
        non_floor_plot[:, 2],
        s=1,
        alpha=0.35,
        label="non-floor",
    )
    axes[0].scatter(
        floor_plot[:, 0],
        floor_plot[:, 2],
        s=1,
        alpha=0.8,
        label="floor",
    )
    axes[0].set_title("X-Z projection")
    axes[0].set_xlabel("X, meters")
    axes[0].set_ylabel("Z, meters")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(markerscale=5)

    # X-Y projection, displayed as X-(-Y)
    axes[1].scatter(
        non_floor_plot[:, 0],
        -non_floor_plot[:, 1],
        s=1,
        alpha=0.35,
        label="non-floor",
    )
    axes[1].scatter(
        floor_plot[:, 0],
        -floor_plot[:, 1],
        s=1,
        alpha=0.8,
        label="floor",
    )
    axes[1].set_title("X-Y projection")
    axes[1].set_xlabel("X, meters")
    axes[1].set_ylabel("-Y, meters")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(markerscale=5)

    # Z-Y projection, displayed as Z-(-Y)
    axes[2].scatter(
        non_floor_plot[:, 2],
        -non_floor_plot[:, 1],
        s=1,
        alpha=0.35,
        label="non-floor",
    )
    axes[2].scatter(
        floor_plot[:, 2],
        -floor_plot[:, 1],
        s=1,
        alpha=0.8,
        label="floor",
    )
    axes[2].set_title("Z-Y projection")
    axes[2].set_xlabel("Z, meters")
    axes[2].set_ylabel("-Y, meters")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(markerscale=5)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()