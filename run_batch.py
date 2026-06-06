import time
from pathlib import Path

import pandas as pd

from src.config import (
    DATA_RAW_DIR,
    FIGURES_DIR,
    OCCUPANCY_MAPS_DIR,
    METRICS_DIR,
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
)
from src.visualization import (
    plot_rgb_and_depth,
    plot_point_cloud_projections,
    plot_floor_segmentation_projections,
)
from src.occupancy_grid import save_occupancy_grid_visualization


def process_one_sample(dataset: NYU2KaggleDataset, index: int) -> dict:
    start_time = time.perf_counter()

    rgb, depth_raw = dataset[index]
    rgb_path, depth_path = dataset.get_paths(index)

    depth_m = convert_depth_to_meters(depth_raw, depth_scale=DEPTH_SCALE)

    depth_clean = clean_depth_map(
        depth_m,
        min_depth_m=MIN_DEPTH_M,
        max_depth_m=MAX_DEPTH_M,
        median_kernel_size=5,
    )

    depth_stats = get_depth_stats(depth_clean)

    points, colors = create_point_cloud_from_rgbd(
        rgb=rgb,
        depth_m=depth_clean,
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
        downsample_step=DOWNSAMPLE_STEP,
    )

    point_cloud_stats = get_point_cloud_stats(points)

    floor_result = detect_floor_plane_ransac(
        points=points,
        colors=colors,
        distance_threshold=0.03,
        ransac_n=3,
        num_iterations=700,
        max_planes=8,
        min_inlier_ratio=0.02,
        min_normal_y_abs=0.45,
    )

    base_result = {
        "index": index,
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "depth_valid_pixels": depth_stats["valid_pixels"],
        "depth_min_m": depth_stats["min_m"],
        "depth_max_m": depth_stats["max_m"],
        "depth_mean_m": depth_stats["mean_m"],
        "point_count": point_cloud_stats["num_points"],
        "floor_found": False,
        "floor_points": 0,
        "floor_inlier_ratio": 0.0,
        "floor_mean_y": None,
        "floor_normal_y_abs": None,
        "obstacle_points": 0,
        "occupancy_total_cells": None,
        "occupancy_known_cells": None,
        "occupancy_unknown_ratio": None,
        "occupancy_free_space_ratio_known": None,
        "occupancy_obstacle_ratio_known": None,
        "traversability_total_cells": None,
        "traversability_known_cells": None,
        "traversability_unknown_ratio": None,
        "traversability_free_space_ratio_known": None,
        "traversability_obstacle_ratio_known": None,
        "processing_time_sec": None,
        "error": None,
    }

    if floor_result is None:
        base_result["processing_time_sec"] = time.perf_counter() - start_time
        return base_result

    occupancy_result = build_occupancy_grid(
        floor_points=floor_result.floor_points,
        all_points=points,
        plane_model=floor_result.plane_model,
        resolution=GRID_RESOLUTION_M,
        min_obstacle_height_m=MIN_OBSTACLE_HEIGHT_M,
        max_obstacle_height_m=MAX_OBSTACLE_HEIGHT_M,
        padding_m=0.20,
    )

    inflated_grid = inflate_obstacles(
        grid=occupancy_result.grid,
        robot_radius_m=ROBOT_RADIUS_M,
        resolution=GRID_RESOLUTION_M,
    )

    occupancy_metrics = compute_occupancy_metrics(occupancy_result.grid)
    traversability_metrics = compute_occupancy_metrics(inflated_grid)

    base_result.update(
        {
            "floor_found": True,
            "floor_points": int(floor_result.floor_points.shape[0]),
            "floor_inlier_ratio": float(floor_result.inlier_ratio),
            "floor_mean_y": float(floor_result.mean_floor_y),
            "floor_normal_y_abs": float(floor_result.normal_y_abs),
            "obstacle_points": int(occupancy_result.obstacle_points.shape[0]),
            "occupancy_total_cells": occupancy_metrics["total_cells"],
            "occupancy_known_cells": occupancy_metrics["known_cells"],
            "occupancy_unknown_ratio": occupancy_metrics["unknown_ratio"],
            "occupancy_free_space_ratio_known": occupancy_metrics["free_space_ratio_known"],
            "occupancy_obstacle_ratio_known": occupancy_metrics["obstacle_ratio_known"],
            "traversability_total_cells": traversability_metrics["total_cells"],
            "traversability_known_cells": traversability_metrics["known_cells"],
            "traversability_unknown_ratio": traversability_metrics["unknown_ratio"],
            "traversability_free_space_ratio_known": traversability_metrics[
                "free_space_ratio_known"
            ],
            "traversability_obstacle_ratio_known": traversability_metrics[
                "obstacle_ratio_known"
            ],
        }
    )

    base_result["processing_time_sec"] = time.perf_counter() - start_time

    return base_result


def save_visualizations_for_sample(dataset: NYU2KaggleDataset, index: int) -> None:
    rgb, depth_raw = dataset[index]

    depth_m = convert_depth_to_meters(depth_raw, depth_scale=DEPTH_SCALE)
    depth_clean = clean_depth_map(
        depth_m,
        min_depth_m=MIN_DEPTH_M,
        max_depth_m=MAX_DEPTH_M,
        median_kernel_size=5,
    )
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
        distance_threshold=0.03,
        ransac_n=3,
        num_iterations=1000,
        max_planes=8,
        min_inlier_ratio=0.02,
        min_normal_y_abs=0.45,
    )

    if floor_result is None:
        return

    occupancy_result = build_occupancy_grid(
        floor_points=floor_result.floor_points,
        all_points=points,
        plane_model=floor_result.plane_model,
        resolution=GRID_RESOLUTION_M,
        min_obstacle_height_m=MIN_OBSTACLE_HEIGHT_M,
        max_obstacle_height_m=MAX_OBSTACLE_HEIGHT_M,
        padding_m=0.20,
    )

    inflated_grid = inflate_obstacles(
        grid=occupancy_result.grid,
        robot_radius_m=ROBOT_RADIUS_M,
        resolution=GRID_RESOLUTION_M,
    )

    plot_rgb_and_depth(
        rgb=rgb,
        depth_m=depth_clean,
        normalized_depth=depth_display,
        save_path=FIGURES_DIR / f"batch_sample_{index:05d}_rgb_depth.png",
        title=f"RGB-D sample {index:05d}",
    )

    plot_point_cloud_projections(
        points=points,
        colors=colors,
        save_path=FIGURES_DIR / f"batch_sample_{index:05d}_point_cloud.png",
        title=f"Point cloud projections, sample {index:05d}",
    )

    plot_floor_segmentation_projections(
        floor_points=floor_result.floor_points,
        non_floor_points=floor_result.non_floor_points,
        save_path=FIGURES_DIR / f"batch_sample_{index:05d}_floor_segmentation.png",
        title=f"Floor segmentation, sample {index:05d}",
    )

    save_occupancy_grid_visualization(
        grid=occupancy_result.grid,
        save_path=OCCUPANCY_MAPS_DIR / f"batch_sample_{index:05d}_occupancy_grid.png",
        title=f"Occupancy grid, sample {index:05d}",
    )

    save_occupancy_grid_visualization(
        grid=inflated_grid,
        save_path=OCCUPANCY_MAPS_DIR / f"batch_sample_{index:05d}_traversability_grid.png",
        title=f"Traversability grid, sample {index:05d}",
    )


def main() -> None:
    split = "test"
    max_samples = 100
    max_visualizations = 10

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OCCUPANCY_MAPS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = NYU2KaggleDataset(
        data_root=DATA_RAW_DIR,
        split=split,
        max_samples=max_samples,
    )

    print(f"Batch processing split={split}, samples={len(dataset)}")

    results = []

    for index in range(len(dataset)):
        print(f"Processing sample {index + 1}/{len(dataset)}...")

        try:
            result = process_one_sample(dataset, index)
        except Exception as error:
            result = {
                "index": index,
                "error": str(error),
                "floor_found": False,
            }

        results.append(result)

    df = pd.DataFrame(results)

    metrics_path = METRICS_DIR / f"batch_metrics_{split}_{max_samples}.csv"
    df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print()
    print(f"Saved metrics to: {metrics_path}")

    valid_df = df[
        (df["floor_found"] == True)
        & (df["floor_inlier_ratio"] >= 0.08)
        & (df["floor_normal_y_abs"] >= 0.90)
        & (df["floor_mean_y"] >= 0.45)
        & (df["occupancy_known_cells"] > 0)
    ].copy()

    valid_df = valid_df.sort_values(
        by=["floor_inlier_ratio", "floor_normal_y_abs"],
        ascending=False,
    )

    best_indices = valid_df["index"].head(max_visualizations).astype(int).tolist()

    print()
    print("Best samples selected for visualization:")
    print(best_indices)

    for index in best_indices:
        print(f"Saving visualizations for sample {index}...")
        save_visualizations_for_sample(dataset, index)

    summary = {
        "samples_total": len(df),
        "floor_found_count": int(df["floor_found"].sum()) if "floor_found" in df else 0,
        "floor_found_ratio": float(df["floor_found"].mean()) if "floor_found" in df else 0.0,
        "mean_processing_time_sec": float(df["processing_time_sec"].dropna().mean())
        if "processing_time_sec" in df
        else None,
        "mean_floor_inlier_ratio": float(valid_df["floor_inlier_ratio"].mean())
        if len(valid_df) > 0
        else None,
        "mean_occupancy_free_space_ratio_known": float(
            valid_df["occupancy_free_space_ratio_known"].mean()
        )
        if len(valid_df) > 0
        else None,
        "mean_traversability_free_space_ratio_known": float(
            valid_df["traversability_free_space_ratio_known"].mean()
        )
        if len(valid_df) > 0
        else None,
    }

    summary_path = METRICS_DIR / f"summary_{split}_{max_samples}.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8-sig")

    print()
    print("Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print()
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()