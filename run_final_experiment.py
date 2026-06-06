import time
from pathlib import Path
from typing import Optional

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    DATA_RAW_DIR,
    OUTPUTS_DIR,
    DEPTH_SCALE,
    MIN_DEPTH_M,
    MAX_DEPTH_M,
    FX,
    FY,
    CX,
    CY,
    DOWNSAMPLE_STEP,
    GRID_RESOLUTION_M,
    MIN_OBSTACLE_HEIGHT_M,
    MAX_OBSTACLE_HEIGHT_M,
    ROBOT_RADIUS_M,
)
from src.data_loading import NYU2KaggleDataset
from src.depth_preprocessing import (
    convert_depth_to_meters,
    clean_depth_map,
    normalize_depth_for_display,
    get_depth_stats,
)
from src.point_cloud import (
    create_point_cloud_from_rgbd,
    get_point_cloud_stats,
)
from src.floor_detection import detect_floor_plane_ransac
from src.occupancy_grid import (
    build_occupancy_grid,
    inflate_obstacles,
    compute_occupancy_metrics,
    UNKNOWN,
    FREE,
    OCCUPIED,
)


# =========================
# Experiment settings
# =========================

SPLIT = "test"
MAX_SAMPLES: Optional[int] = None

SAVE_TOP_N_VISUALIZATIONS = 40

BORDER_PX = 10

FLOOR_DISTANCE_THRESHOLD = 0.03
FLOOR_RANSAC_N = 3
FLOOR_NUM_ITERATIONS = 700
FLOOR_MAX_PLANES = 8
FLOOR_MIN_INLIER_RATIO = 0.02
FLOOR_MIN_NORMAL_Y_ABS = 0.45

GOOD_FLOOR_MIN_INLIER_RATIO = 0.08
GOOD_FLOOR_MIN_NORMAL_Y_ABS = 0.90
GOOD_FLOOR_MIN_MEAN_Y = 0.45

GOOD_MIN_OCCUPANCY_FREE_RATIO = 0.10
GOOD_MAX_OCCUPANCY_FREE_RATIO = 0.85
GOOD_MIN_TRAVERSABILITY_FREE_RATIO = 0.01

MIN_OCCUPIED_COMPONENT_SIZE = 3
CLOSE_FREE_SPACE = True

GRID_VISUALIZATION_DPI = 250
COMBINED_VISUALIZATION_DPI = 220


# =========================
# Output directories
# =========================

FINAL_OUTPUT_DIR = OUTPUTS_DIR / "final_experiment"
FINAL_METRICS_DIR = FINAL_OUTPUT_DIR / "metrics"
FINAL_FIGURES_DIR = FINAL_OUTPUT_DIR / "figures"
FINAL_OCCUPANCY_DIR = FINAL_OUTPUT_DIR / "occupancy_maps"
FINAL_COMBINED_DIR = FINAL_OUTPUT_DIR / "combined"


# =========================
# Utility functions
# =========================

def ensure_output_dirs() -> None:
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_OCCUPANCY_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_COMBINED_DIR.mkdir(parents=True, exist_ok=True)


def remove_depth_border(depth_m: np.ndarray, border_px: int = 10) -> np.ndarray:
    """
    Removes border pixels from depth map by setting them to NaN.

    This reduces artifacts caused by frame boundaries and unstable depth
    values near image borders.
    """
    if border_px <= 0:
        return depth_m

    cleaned = depth_m.copy()

    cleaned[:border_px, :] = np.nan
    cleaned[-border_px:, :] = np.nan
    cleaned[:, :border_px] = np.nan
    cleaned[:, -border_px:] = np.nan

    return cleaned


def clean_occupancy_grid(
    grid: np.ndarray,
    min_occupied_component_size: int = 3,
    close_free_space: bool = True,
) -> np.ndarray:
    """
    Cleans occupancy grid:
    - removes very small occupied components;
    - optionally closes small holes in free space.

    Grid values:
        0 - unknown
        1 - free
        2 - occupied
    """
    cleaned = grid.copy()

    occupied_mask = (cleaned == OCCUPIED).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        occupied_mask,
        connectivity=8,
    )

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area < min_occupied_component_size:
            cleaned[labels == label_id] = UNKNOWN

    if close_free_space:
        free_mask = (cleaned == FREE).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        free_closed = cv2.morphologyEx(
            free_mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

        cleaned[(free_closed == 1) & (cleaned == UNKNOWN)] = FREE

        # Occupied cells keep priority.
        cleaned[occupied_mask == 1] = OCCUPIED

    return cleaned


def grid_to_rgb(grid: np.ndarray) -> np.ndarray:
    """
    Converts occupancy grid to RGB image.

    Colors:
        unknown  - dark gray
        free     - light gray
        occupied - black
    """
    rgb = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.float32)

    rgb[grid == UNKNOWN] = [0.25, 0.25, 0.25]
    rgb[grid == FREE] = [0.85, 0.85, 0.85]
    rgb[grid == OCCUPIED] = [0.0, 0.0, 0.0]

    return rgb


def save_grid_image(
    grid: np.ndarray,
    save_path: Path,
    title: str,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = grid_to_rgb(grid)

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb, origin="lower", interpolation="nearest")
    plt.title(title)
    plt.xlabel("X grid coordinate")
    plt.ylabel("Z grid coordinate")
    plt.tight_layout()
    plt.savefig(save_path, dpi=GRID_VISUALIZATION_DPI, bbox_inches="tight")
    plt.close()


def save_rgb_depth_image(
    rgb: np.ndarray,
    depth_clean: np.ndarray,
    depth_display: np.ndarray,
    save_path: Path,
    title: str,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    depth_plot = axes[1].imshow(depth_clean, cmap="viridis")
    axes[1].set_title("Depth, meters")
    axes[1].axis("off")
    fig.colorbar(depth_plot, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(depth_display, cmap="gray")
    axes[2].set_title("Normalized depth")
    axes[2].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=COMBINED_VISUALIZATION_DPI, bbox_inches="tight")
    plt.close()


def save_point_cloud_projection_image(
    points: np.ndarray,
    colors: np.ndarray,
    save_path: Path,
    title: str,
    max_points: int = 30000,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if points.shape[0] > max_points:
        indices = np.random.choice(points.shape[0], size=max_points, replace=False)
        plot_points = points[indices]
        plot_colors = colors[indices]
    else:
        plot_points = points
        plot_colors = colors

    x = plot_points[:, 0]
    y = plot_points[:, 1]
    z = plot_points[:, 2]

    plot_colors = np.clip(plot_colors, 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(x, z, s=1, c=plot_colors)
    axes[0].set_title("X-Z projection")
    axes[0].set_xlabel("X, meters")
    axes[0].set_ylabel("Z, meters")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(x, -y, s=1, c=plot_colors)
    axes[1].set_title("X-Y projection")
    axes[1].set_xlabel("X, meters")
    axes[1].set_ylabel("-Y, meters")
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(z, -y, s=1, c=plot_colors)
    axes[2].set_title("Z-Y projection")
    axes[2].set_xlabel("Z, meters")
    axes[2].set_ylabel("-Y, meters")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=COMBINED_VISUALIZATION_DPI, bbox_inches="tight")
    plt.close()


def save_floor_segmentation_image(
    floor_points: np.ndarray,
    non_floor_points: np.ndarray,
    save_path: Path,
    title: str,
    max_points_per_class: int = 30000,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    def sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
        if points.shape[0] > max_points:
            indices = np.random.choice(points.shape[0], size=max_points, replace=False)
            return points[indices]
        return points

    floor_plot = sample_points(floor_points, max_points_per_class)
    non_floor_plot = sample_points(non_floor_points, max_points_per_class)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

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
    plt.savefig(save_path, dpi=COMBINED_VISUALIZATION_DPI, bbox_inches="tight")
    plt.close()


def save_combined_result_image(
    rgb: np.ndarray,
    depth_display: np.ndarray,
    floor_points: np.ndarray,
    non_floor_points: np.ndarray,
    occupancy_grid: np.ndarray,
    traversability_grid: np.ndarray,
    save_path: Path,
    title: str,
    max_points_per_class: int = 18000,
) -> None:
    """
    Saves one compact figure:
        RGB | Depth | Floor segmentation | Occupancy grid | Traversability grid
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    def sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
        if points.shape[0] > max_points:
            indices = np.random.choice(points.shape[0], size=max_points, replace=False)
            return points[indices]
        return points

    floor_plot = sample_points(floor_points, max_points_per_class)
    non_floor_plot = sample_points(non_floor_points, max_points_per_class)

    fig, axes = plt.subplots(1, 5, figsize=(24, 5))

    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    axes[1].imshow(depth_display, cmap="gray")
    axes[1].set_title("Depth")
    axes[1].axis("off")

    axes[2].scatter(
        non_floor_plot[:, 0],
        non_floor_plot[:, 2],
        s=1,
        alpha=0.25,
        label="non-floor",
    )
    axes[2].scatter(
        floor_plot[:, 0],
        floor_plot[:, 2],
        s=1,
        alpha=0.8,
        label="floor",
    )
    axes[2].set_title("Floor segmentation")
    axes[2].set_xlabel("X, meters")
    axes[2].set_ylabel("Z, meters")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(markerscale=5)

    axes[3].imshow(
        grid_to_rgb(occupancy_grid),
        origin="lower",
        interpolation="nearest",
    )
    axes[3].set_title("Occupancy grid")
    axes[3].set_xlabel("X")
    axes[3].set_ylabel("Z")

    axes[4].imshow(
        grid_to_rgb(traversability_grid),
        origin="lower",
        interpolation="nearest",
    )
    axes[4].set_title("Traversability grid")
    axes[4].set_xlabel("X")
    axes[4].set_ylabel("Z")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=COMBINED_VISUALIZATION_DPI, bbox_inches="tight")
    plt.close()


def compute_candidate_score(row: pd.Series) -> float:
    """
    Score for choosing visually useful examples.

    It prefers:
    - confident floor detection;
    - horizontal floor plane;
    - visible free space;
    - non-zero traversable space;
    - moderate amount of obstacles.
    """
    if not bool(row.get("floor_found", False)):
        return 0.0

    floor_inlier_ratio = float(row.get("floor_inlier_ratio", 0.0) or 0.0)
    floor_normal_y_abs = float(row.get("floor_normal_y_abs", 0.0) or 0.0)
    occupancy_free = float(row.get("occupancy_free_space_ratio_known", 0.0) or 0.0)
    occupancy_obstacle = float(row.get("occupancy_obstacle_ratio_known", 0.0) or 0.0)
    traversability_free = float(row.get("traversability_free_space_ratio_known", 0.0) or 0.0)

    balance = 1.0 - abs(occupancy_free - occupancy_obstacle)

    score = (
        3.0 * floor_inlier_ratio
        + 1.0 * floor_normal_y_abs
        + 1.0 * occupancy_free
        + 0.8 * traversability_free
        + 0.5 * balance
    )

    return float(score)


# =========================
# Main processing functions
# =========================

def process_one_sample(dataset: NYU2KaggleDataset, index: int) -> dict:
    start_time = time.perf_counter()

    rgb_path, depth_path = dataset.get_paths(index)

    result = {
        "index": index,
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "depth_valid_pixels": None,
        "depth_min_m": None,
        "depth_max_m": None,
        "depth_mean_m": None,
        "depth_median_m": None,
        "point_count": None,
        "point_x_min": None,
        "point_x_max": None,
        "point_y_min": None,
        "point_y_max": None,
        "point_z_min": None,
        "point_z_max": None,
        "floor_found": False,
        "floor_points": 0,
        "non_floor_points": 0,
        "floor_inlier_ratio": 0.0,
        "floor_mean_y": None,
        "floor_normal_y_abs": None,
        "plane_a": None,
        "plane_b": None,
        "plane_c": None,
        "plane_d": None,
        "obstacle_points": 0,
        "occupancy_total_cells": None,
        "occupancy_known_cells": None,
        "occupancy_unknown_cells": None,
        "occupancy_free_cells": None,
        "occupancy_occupied_cells": None,
        "occupancy_unknown_ratio": None,
        "occupancy_free_space_ratio_known": None,
        "occupancy_obstacle_ratio_known": None,
        "traversability_total_cells": None,
        "traversability_known_cells": None,
        "traversability_unknown_cells": None,
        "traversability_free_cells": None,
        "traversability_occupied_cells": None,
        "traversability_unknown_ratio": None,
        "traversability_free_space_ratio_known": None,
        "traversability_obstacle_ratio_known": None,
        "candidate_score": 0.0,
        "processing_time_sec": None,
        "error": None,
    }

    try:
        rgb, depth_raw = dataset[index]

        depth_m = convert_depth_to_meters(depth_raw, depth_scale=DEPTH_SCALE)

        depth_clean = clean_depth_map(
            depth_m,
            min_depth_m=MIN_DEPTH_M,
            max_depth_m=MAX_DEPTH_M,
            median_kernel_size=5,
        )

        depth_clean = remove_depth_border(depth_clean, border_px=BORDER_PX)

        depth_stats = get_depth_stats(depth_clean)

        result.update(
            {
                "depth_valid_pixels": depth_stats["valid_pixels"],
                "depth_min_m": depth_stats["min_m"],
                "depth_max_m": depth_stats["max_m"],
                "depth_mean_m": depth_stats["mean_m"],
                "depth_median_m": depth_stats["median_m"],
            }
        )

        points, colors = create_point_cloud_from_rgbd(
            rgb=rgb,
            depth_m=depth_clean,
            fx=FX,
            fy=FY,
            cx=CX,
            cy=CY,
            downsample_step=DOWNSAMPLE_STEP,
        )

        point_stats = get_point_cloud_stats(points)

        result.update(
            {
                "point_count": point_stats["num_points"],
                "point_x_min": point_stats["x_min"],
                "point_x_max": point_stats["x_max"],
                "point_y_min": point_stats["y_min"],
                "point_y_max": point_stats["y_max"],
                "point_z_min": point_stats["z_min"],
                "point_z_max": point_stats["z_max"],
            }
        )

        floor_result = detect_floor_plane_ransac(
            points=points,
            colors=colors,
            distance_threshold=FLOOR_DISTANCE_THRESHOLD,
            ransac_n=FLOOR_RANSAC_N,
            num_iterations=FLOOR_NUM_ITERATIONS,
            max_planes=FLOOR_MAX_PLANES,
            min_inlier_ratio=FLOOR_MIN_INLIER_RATIO,
            min_normal_y_abs=FLOOR_MIN_NORMAL_Y_ABS,
        )

        if floor_result is None:
            result["processing_time_sec"] = time.perf_counter() - start_time
            return result

        occupancy_result = build_occupancy_grid(
            floor_points=floor_result.floor_points,
            all_points=points,
            plane_model=floor_result.plane_model,
            resolution=GRID_RESOLUTION_M,
            min_obstacle_height_m=MIN_OBSTACLE_HEIGHT_M,
            max_obstacle_height_m=MAX_OBSTACLE_HEIGHT_M,
            padding_m=0.20,
        )

        cleaned_occupancy_grid = clean_occupancy_grid(
            occupancy_result.grid,
            min_occupied_component_size=MIN_OCCUPIED_COMPONENT_SIZE,
            close_free_space=CLOSE_FREE_SPACE,
        )

        traversability_grid = inflate_obstacles(
            grid=cleaned_occupancy_grid,
            robot_radius_m=ROBOT_RADIUS_M,
            resolution=GRID_RESOLUTION_M,
        )

        occupancy_metrics = compute_occupancy_metrics(cleaned_occupancy_grid)
        traversability_metrics = compute_occupancy_metrics(traversability_grid)

        a, b, c, d = floor_result.plane_model

        result.update(
            {
                "floor_found": True,
                "floor_points": int(floor_result.floor_points.shape[0]),
                "non_floor_points": int(floor_result.non_floor_points.shape[0]),
                "floor_inlier_ratio": float(floor_result.inlier_ratio),
                "floor_mean_y": float(floor_result.mean_floor_y),
                "floor_normal_y_abs": float(floor_result.normal_y_abs),
                "plane_a": float(a),
                "plane_b": float(b),
                "plane_c": float(c),
                "plane_d": float(d),
                "obstacle_points": int(occupancy_result.obstacle_points.shape[0]),
                "occupancy_total_cells": occupancy_metrics["total_cells"],
                "occupancy_known_cells": occupancy_metrics["known_cells"],
                "occupancy_unknown_cells": occupancy_metrics["unknown_cells"],
                "occupancy_free_cells": occupancy_metrics["free_cells"],
                "occupancy_occupied_cells": occupancy_metrics["occupied_cells"],
                "occupancy_unknown_ratio": occupancy_metrics["unknown_ratio"],
                "occupancy_free_space_ratio_known": occupancy_metrics[
                    "free_space_ratio_known"
                ],
                "occupancy_obstacle_ratio_known": occupancy_metrics[
                    "obstacle_ratio_known"
                ],
                "traversability_total_cells": traversability_metrics["total_cells"],
                "traversability_known_cells": traversability_metrics["known_cells"],
                "traversability_unknown_cells": traversability_metrics["unknown_cells"],
                "traversability_free_cells": traversability_metrics["free_cells"],
                "traversability_occupied_cells": traversability_metrics[
                    "occupied_cells"
                ],
                "traversability_unknown_ratio": traversability_metrics[
                    "unknown_ratio"
                ],
                "traversability_free_space_ratio_known": traversability_metrics[
                    "free_space_ratio_known"
                ],
                "traversability_obstacle_ratio_known": traversability_metrics[
                    "obstacle_ratio_known"
                ],
            }
        )

        result["processing_time_sec"] = time.perf_counter() - start_time

        return result

    except Exception as error:
        result["processing_time_sec"] = time.perf_counter() - start_time
        result["error"] = str(error)
        return result


def save_all_visualizations_for_sample(dataset: NYU2KaggleDataset, index: int) -> bool:
    """
    Recomputes pipeline for one sample and saves all useful visualizations.
    Returns True if visualizations were saved.
    """
    try:
        rgb, depth_raw = dataset[index]

        depth_m = convert_depth_to_meters(depth_raw, depth_scale=DEPTH_SCALE)

        depth_clean = clean_depth_map(
            depth_m,
            min_depth_m=MIN_DEPTH_M,
            max_depth_m=MAX_DEPTH_M,
            median_kernel_size=5,
        )

        depth_clean = remove_depth_border(depth_clean, border_px=BORDER_PX)
        depth_display = normalize_depth_for_display(depth_clean)

        points, colors = create_point_cloud_from_rgbd(
            rgb=rgb,
            depth_m=depth_clean,
            fx=FX,
            fy=FY,
            cx=CX,
            cy=CY,
            downsample_step=DOWNSAMPLE_STEP,
        )

        floor_result = detect_floor_plane_ransac(
            points=points,
            colors=colors,
            distance_threshold=FLOOR_DISTANCE_THRESHOLD,
            ransac_n=FLOOR_RANSAC_N,
            num_iterations=1000,
            max_planes=FLOOR_MAX_PLANES,
            min_inlier_ratio=FLOOR_MIN_INLIER_RATIO,
            min_normal_y_abs=FLOOR_MIN_NORMAL_Y_ABS,
        )

        if floor_result is None:
            return False

        occupancy_result = build_occupancy_grid(
            floor_points=floor_result.floor_points,
            all_points=points,
            plane_model=floor_result.plane_model,
            resolution=GRID_RESOLUTION_M,
            min_obstacle_height_m=MIN_OBSTACLE_HEIGHT_M,
            max_obstacle_height_m=MAX_OBSTACLE_HEIGHT_M,
            padding_m=0.20,
        )

        cleaned_occupancy_grid = clean_occupancy_grid(
            occupancy_result.grid,
            min_occupied_component_size=MIN_OCCUPIED_COMPONENT_SIZE,
            close_free_space=CLOSE_FREE_SPACE,
        )

        traversability_grid = inflate_obstacles(
            grid=cleaned_occupancy_grid,
            robot_radius_m=ROBOT_RADIUS_M,
            resolution=GRID_RESOLUTION_M,
        )

        save_rgb_depth_image(
            rgb=rgb,
            depth_clean=depth_clean,
            depth_display=depth_display,
            save_path=FINAL_FIGURES_DIR / f"sample_{index:05d}_rgb_depth.png",
            title=f"RGB-D sample {index:05d}",
        )

        save_point_cloud_projection_image(
            points=points,
            colors=colors,
            save_path=FINAL_FIGURES_DIR / f"sample_{index:05d}_point_cloud.png",
            title=f"Point cloud projections, sample {index:05d}",
        )

        save_floor_segmentation_image(
            floor_points=floor_result.floor_points,
            non_floor_points=floor_result.non_floor_points,
            save_path=FINAL_FIGURES_DIR / f"sample_{index:05d}_floor_segmentation.png",
            title=f"Floor segmentation, sample {index:05d}",
        )

        save_grid_image(
            grid=cleaned_occupancy_grid,
            save_path=FINAL_OCCUPANCY_DIR / f"sample_{index:05d}_occupancy_grid.png",
            title=f"Occupancy grid, sample {index:05d}",
        )

        save_grid_image(
            grid=traversability_grid,
            save_path=FINAL_OCCUPANCY_DIR / f"sample_{index:05d}_traversability_grid.png",
            title=f"Traversability grid, sample {index:05d}",
        )

        save_combined_result_image(
            rgb=rgb,
            depth_display=depth_display,
            floor_points=floor_result.floor_points,
            non_floor_points=floor_result.non_floor_points,
            occupancy_grid=cleaned_occupancy_grid,
            traversability_grid=traversability_grid,
            save_path=FINAL_COMBINED_DIR / f"sample_{index:05d}_combined.png",
            title=f"Final pipeline result, sample {index:05d}",
        )

        return True

    except Exception as error:
        print(f"Failed to save visualizations for sample {index}: {error}")
        return False


def build_summary(df: pd.DataFrame, valid_df: pd.DataFrame) -> dict:
    total = len(df)

    floor_found_count = int(df["floor_found"].sum()) if "floor_found" in df else 0
    error_count = int(df["error"].notna().sum()) if "error" in df else 0

    summary = {
        "split": SPLIT,
        "samples_total": total,
        "samples_with_errors": error_count,
        "floor_found_count": floor_found_count,
        "floor_found_ratio": floor_found_count / total if total > 0 else 0.0,
        "valid_candidate_count": int(len(valid_df)),
        "valid_candidate_ratio": len(valid_df) / total if total > 0 else 0.0,
        "mean_processing_time_sec_all": float(df["processing_time_sec"].dropna().mean()),
        "median_processing_time_sec_all": float(df["processing_time_sec"].dropna().median()),
        "mean_floor_inlier_ratio_found": float(
            df.loc[df["floor_found"] == True, "floor_inlier_ratio"].dropna().mean()
        )
        if floor_found_count > 0
        else None,
        "mean_floor_normal_y_abs_found": float(
            df.loc[df["floor_found"] == True, "floor_normal_y_abs"].dropna().mean()
        )
        if floor_found_count > 0
        else None,
        "mean_occupancy_free_space_ratio_known_valid": float(
            valid_df["occupancy_free_space_ratio_known"].dropna().mean()
        )
        if len(valid_df) > 0
        else None,
        "mean_occupancy_obstacle_ratio_known_valid": float(
            valid_df["occupancy_obstacle_ratio_known"].dropna().mean()
        )
        if len(valid_df) > 0
        else None,
        "mean_traversability_free_space_ratio_known_valid": float(
            valid_df["traversability_free_space_ratio_known"].dropna().mean()
        )
        if len(valid_df) > 0
        else None,
        "mean_traversability_obstacle_ratio_known_valid": float(
            valid_df["traversability_obstacle_ratio_known"].dropna().mean()
        )
        if len(valid_df) > 0
        else None,
        "grid_resolution_m": GRID_RESOLUTION_M,
        "min_obstacle_height_m": MIN_OBSTACLE_HEIGHT_M,
        "max_obstacle_height_m": MAX_OBSTACLE_HEIGHT_M,
        "robot_radius_m": ROBOT_RADIUS_M,
        "downsample_step": DOWNSAMPLE_STEP,
    }

    return summary


def main() -> None:
    ensure_output_dirs()

    sample_tag = "all" if MAX_SAMPLES is None else str(MAX_SAMPLES)

    dataset = NYU2KaggleDataset(
        data_root=DATA_RAW_DIR,
        split=SPLIT,
        max_samples=MAX_SAMPLES,
    )

    print("=" * 80)
    print("Final RGB-D free-space mapping experiment")
    print("=" * 80)
    print(f"Split: {SPLIT}")
    print(f"Samples: {len(dataset)}")
    print(f"Max samples setting: {MAX_SAMPLES}")
    print(f"Grid resolution: {GRID_RESOLUTION_M} m/cell")
    print(f"Obstacle height range: {MIN_OBSTACLE_HEIGHT_M} - {MAX_OBSTACLE_HEIGHT_M} m")
    print(f"Robot radius: {ROBOT_RADIUS_M} m")
    print(f"Output directory: {FINAL_OUTPUT_DIR}")
    print("=" * 80)
    print()

    rows = []

    for index in tqdm(range(len(dataset)), desc="Processing frames"):
        row = process_one_sample(dataset, index)
        rows.append(row)

    df = pd.DataFrame(rows)

    df["candidate_score"] = df.apply(compute_candidate_score, axis=1)

    metrics_path = FINAL_METRICS_DIR / f"final_metrics_{SPLIT}_{sample_tag}.csv"
    df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    valid_df = df[
        (df["floor_found"] == True)
        & (df["error"].isna())
        & (df["floor_inlier_ratio"] >= GOOD_FLOOR_MIN_INLIER_RATIO)
        & (df["floor_normal_y_abs"] >= GOOD_FLOOR_MIN_NORMAL_Y_ABS)
        & (df["floor_mean_y"] >= GOOD_FLOOR_MIN_MEAN_Y)
        & (df["occupancy_known_cells"] > 0)
        & (df["occupancy_free_space_ratio_known"] >= GOOD_MIN_OCCUPANCY_FREE_RATIO)
        & (df["occupancy_free_space_ratio_known"] <= GOOD_MAX_OCCUPANCY_FREE_RATIO)
        & (df["traversability_free_space_ratio_known"] >= GOOD_MIN_TRAVERSABILITY_FREE_RATIO)
    ].copy()

    valid_df = valid_df.sort_values(
        by="candidate_score",
        ascending=False,
    )

    selected_df = valid_df.head(SAVE_TOP_N_VISUALIZATIONS).copy()

    selected_path = FINAL_METRICS_DIR / f"final_selected_candidates_{SPLIT}_{sample_tag}.csv"
    selected_df.to_csv(selected_path, index=False, encoding="utf-8-sig")

    summary = build_summary(df, valid_df)
    summary_path = FINAL_METRICS_DIR / f"final_summary_{SPLIT}_{sample_tag}.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8-sig")

    print()
    print("=" * 80)
    print("Metrics saved")
    print("=" * 80)
    print(f"All metrics: {metrics_path}")
    print(f"Selected candidates: {selected_path}")
    print(f"Summary: {summary_path}")
    print()

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    for key, value in summary.items():
        print(f"{key}: {value}")

    print()
    print("=" * 80)
    print(f"Saving visualizations for top {len(selected_df)} candidates")
    print("=" * 80)

    selected_indices = selected_df["index"].astype(int).tolist()

    print("Selected indices:")
    print(selected_indices)
    print()

    saved_count = 0

    for index in tqdm(selected_indices, desc="Saving visualizations"):
        saved = save_all_visualizations_for_sample(dataset, index)

        if saved:
            saved_count += 1

    print()
    print("=" * 80)
    print("Finished")
    print("=" * 80)
    print(f"Saved visualizations: {saved_count} / {len(selected_indices)}")
    print(f"Combined figures: {FINAL_COMBINED_DIR}")
    print(f"Separate figures: {FINAL_FIGURES_DIR}")
    print(f"Occupancy maps: {FINAL_OCCUPANCY_DIR}")
    print(f"CSV metrics: {FINAL_METRICS_DIR}")


if __name__ == "__main__":
    main()