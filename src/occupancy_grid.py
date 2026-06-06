from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.floor_detection import compute_signed_distances_to_plane

UNKNOWN = 0
FREE = 1
OCCUPIED = 2


@dataclass
class OccupancyGridResult:
    grid: np.ndarray
    x_min: float
    x_max: float
    z_min: float
    z_max: float
    resolution: float
    obstacle_points: np.ndarray
    free_points: np.ndarray


def orient_plane_normal_up(plane_model: np.ndarray) -> np.ndarray:
    """
    Orients floor plane normal approximately upward in camera coordinates.

    In our point cloud convention:
    - Y grows downward in the image;
    - therefore upward direction corresponds to negative Y.

    We flip plane model if its normal has positive Y component.
    """
    oriented = plane_model.astype(np.float64).copy()

    if oriented[1] > 0:
        oriented = -oriented

    return oriented


def split_obstacles_by_height_above_floor(
    points: np.ndarray,
    plane_model: np.ndarray,
    min_height_m: float = 0.10,
    max_height_m: float = 2.00,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits points into obstacle points and non-obstacle points
    by height above the detected floor plane.

    The floor plane normal is oriented upward, so positive signed distance
    means the point is above the floor.
    """
    oriented_plane = orient_plane_normal_up(plane_model)

    heights = compute_signed_distances_to_plane(points, oriented_plane)

    obstacle_mask = (heights >= min_height_m) & (heights <= max_height_m)

    obstacle_points = points[obstacle_mask]
    non_obstacle_points = points[~obstacle_mask]

    return obstacle_points, non_obstacle_points


def _points_to_grid_indices(
    points: np.ndarray,
    x_min: float,
    z_min: float,
    resolution: float,
    grid_height: int,
    grid_width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts X-Z point coordinates to occupancy grid row/col indices.
    """
    x = points[:, 0]
    z = points[:, 2]

    cols = np.floor((x - x_min) / resolution).astype(np.int32)
    rows = np.floor((z - z_min) / resolution).astype(np.int32)

    valid = (rows >= 0) & (rows < grid_height) & (cols >= 0) & (cols < grid_width)

    return rows[valid], cols[valid]


def build_occupancy_grid(
    floor_points: np.ndarray,
    all_points: np.ndarray,
    plane_model: np.ndarray,
    resolution: float = 0.05,
    min_obstacle_height_m: float = 0.10,
    max_obstacle_height_m: float = 2.00,
    padding_m: float = 0.20,
) -> OccupancyGridResult:
    """
    Builds a local top-down occupancy grid from floor and obstacle points.

    Grid values:
        0 - unknown
        1 - free
        2 - occupied
    """
    if floor_points.ndim != 2 or floor_points.shape[1] != 3:
        raise ValueError(f"Expected floor_points with shape (N, 3), got {floor_points.shape}")

    if all_points.ndim != 2 or all_points.shape[1] != 3:
        raise ValueError(f"Expected all_points with shape (N, 3), got {all_points.shape}")

    obstacle_points, _ = split_obstacles_by_height_above_floor(
        points=all_points,
        plane_model=plane_model,
        min_height_m=min_obstacle_height_m,
        max_height_m=max_obstacle_height_m,
    )

    combined_points = np.vstack([floor_points, obstacle_points])

    x_min = float(np.min(combined_points[:, 0]) - padding_m)
    x_max = float(np.max(combined_points[:, 0]) + padding_m)
    z_min = float(np.min(combined_points[:, 2]) - padding_m)
    z_max = float(np.max(combined_points[:, 2]) + padding_m)

    grid_width = int(np.ceil((x_max - x_min) / resolution))
    grid_height = int(np.ceil((z_max - z_min) / resolution))

    grid = np.full((grid_height, grid_width), UNKNOWN, dtype=np.uint8)

    # Mark floor as free.
    free_rows, free_cols = _points_to_grid_indices(
        points=floor_points,
        x_min=x_min,
        z_min=z_min,
        resolution=resolution,
        grid_height=grid_height,
        grid_width=grid_width,
    )
    grid[free_rows, free_cols] = FREE

    min_obstacle_points_per_cell = 3

    if obstacle_points.shape[0] > 0:
        obstacle_rows, obstacle_cols = _points_to_grid_indices(
            points=obstacle_points,
            x_min=x_min,
            z_min=z_min,
            resolution=resolution,
            grid_height=grid_height,
            grid_width=grid_width,
        )

        obstacle_counts = np.zeros_like(grid, dtype=np.int32)
        np.add.at(obstacle_counts, (obstacle_rows, obstacle_cols), 1)

        grid[obstacle_counts >= min_obstacle_points_per_cell] = OCCUPIED

    return OccupancyGridResult(
        grid=grid,
        x_min=x_min,
        x_max=x_max,
        z_min=z_min,
        z_max=z_max,
        resolution=resolution,
        obstacle_points=obstacle_points,
        free_points=floor_points,
    )


def inflate_obstacles(grid: np.ndarray, robot_radius_m: float, resolution: float) -> np.ndarray:
    """
    Inflates occupied cells by robot radius.

    This converts an occupancy grid into a traversability grid:
    cells too close to obstacles become occupied as well.
    """
    inflated = grid.copy()

    radius_cells = int(np.ceil(robot_radius_m / resolution))

    if radius_cells <= 0:
        return inflated

    occupied_mask = (grid == OCCUPIED).astype(np.uint8)

    kernel_size = 2 * radius_cells + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    inflated_occupied = cv2.dilate(occupied_mask, kernel, iterations=1).astype(bool)

    inflated[inflated_occupied] = OCCUPIED

    return inflated


def compute_occupancy_metrics(grid: np.ndarray) -> dict:
    """
    Computes basic occupancy grid metrics.
    """
    total_cells = grid.size

    unknown_cells = int(np.sum(grid == UNKNOWN))
    free_cells = int(np.sum(grid == FREE))
    occupied_cells = int(np.sum(grid == OCCUPIED))

    known_cells = free_cells + occupied_cells

    if total_cells == 0:
        raise ValueError("Grid is empty")

    if known_cells == 0:
        free_space_ratio_known = 0.0
        obstacle_ratio_known = 0.0
    else:
        free_space_ratio_known = free_cells / known_cells
        obstacle_ratio_known = occupied_cells / known_cells

    return {
        "total_cells": total_cells,
        "known_cells": known_cells,
        "unknown_cells": unknown_cells,
        "free_cells": free_cells,
        "occupied_cells": occupied_cells,
        "unknown_ratio": unknown_cells / total_cells,
        "free_space_ratio_known": free_space_ratio_known,
        "obstacle_ratio_known": obstacle_ratio_known,
    }


def save_occupancy_grid_visualization(
    grid: np.ndarray,
    save_path: str | Path,
    title: str = "Occupancy grid",
) -> None:
    """
    Saves occupancy grid visualization.

    Colors:
        unknown - dark gray
        free - light gray
        occupied - black
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.float32)

    rgb[grid == UNKNOWN] = [0.25, 0.25, 0.25]
    rgb[grid == FREE] = [0.85, 0.85, 0.85]
    rgb[grid == OCCUPIED] = [0.0, 0.0, 0.0]

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb, origin="lower")
    plt.title(title)
    plt.xlabel("X grid coordinate")
    plt.ylabel("Z grid coordinate")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
