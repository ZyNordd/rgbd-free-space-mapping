from src.config import (
    DATA_RAW_DIR,
    FIGURES_DIR,
    POINT_CLOUDS_DIR,
    OCCUPANCY_MAPS_DIR,
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
    save_point_cloud_ply,
    get_point_cloud_stats,
)
from src.visualization import (
    plot_rgb_and_depth,
    plot_point_cloud_projections,
    plot_floor_segmentation_projections,
)
from src.floor_detection import (
    detect_floor_plane_ransac,
    print_plane_result,
    print_plane_candidates,
)
from src.occupancy_grid import (
    build_occupancy_grid,
    inflate_obstacles,
    compute_occupancy_metrics,
    save_occupancy_grid_visualization,
)


def main() -> None:
    sample_index = 48

    dataset = NYU2KaggleDataset(
        data_root=DATA_RAW_DIR,
        split="test",
        max_samples=50,
    )

    print(f"Dataset size: {len(dataset)}")

    rgb, depth_raw = dataset[sample_index]
    info = dataset.get_sample_info(sample_index)

    print("\nRaw sample info:")
    for key, value in info.items():
        print(f"{key}: {value}")

    depth_m = convert_depth_to_meters(depth_raw, depth_scale=DEPTH_SCALE)
    depth_clean = clean_depth_map(
        depth_m,
        min_depth_m=MIN_DEPTH_M,
        max_depth_m=MAX_DEPTH_M,
        median_kernel_size=5,
    )
    depth_display = normalize_depth_for_display(depth_clean)

    print("\nClean depth stats:")
    depth_stats = get_depth_stats(depth_clean)
    for key, value in depth_stats.items():
        print(f"{key}: {value}")

    rgb_depth_save_path = FIGURES_DIR / f"sample_{sample_index:05d}_rgb_depth.png"

    plot_rgb_and_depth(
        rgb=rgb,
        depth_m=depth_clean,
        normalized_depth=depth_display,
        save_path=rgb_depth_save_path,
        title=f"NYU Depth V2 sample {sample_index:05d}",
    )

    print(f"\nSaved RGB-D visualization to: {rgb_depth_save_path}")

    points, colors = create_point_cloud_from_rgbd(
        rgb=rgb,
        depth_m=depth_clean,
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
        downsample_step=DOWNSAMPLE_STEP,
    )

    print("\nPoint cloud stats:")
    point_cloud_stats = get_point_cloud_stats(points)
    for key, value in point_cloud_stats.items():
        print(f"{key}: {value}")

    point_cloud_save_path = POINT_CLOUDS_DIR / f"sample_{sample_index:05d}_point_cloud.ply"

    save_point_cloud_ply(
        points=points,
        colors=colors,
        save_path=point_cloud_save_path,
    )

    print(f"\nSaved point cloud to: {point_cloud_save_path}")

    projections_save_path = FIGURES_DIR / f"sample_{sample_index:05d}_point_cloud_projections.png"

    plot_point_cloud_projections(
        points=points,
        colors=colors,
        save_path=projections_save_path,
        title=f"Point cloud projections, sample {sample_index:05d}",
    )

    print(f"Saved point cloud projections to: {projections_save_path}")
    
    
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

    print()

    if floor_result is not None:
        print_plane_candidates(floor_result.candidates)

    print()
    print_plane_result(floor_result)

    if floor_result is not None:
        floor_segmentation_save_path = (
            FIGURES_DIR / f"sample_{sample_index:05d}_floor_segmentation.png"
        )

        plot_floor_segmentation_projections(
            floor_points=floor_result.floor_points,
            non_floor_points=floor_result.non_floor_points,
            save_path=floor_segmentation_save_path,
            title=f"Floor segmentation, sample {sample_index:05d}",
        )

        print(f"Saved floor segmentation visualization to: {floor_segmentation_save_path}")

    occupancy_result = build_occupancy_grid(
        floor_points=floor_result.floor_points,
        all_points=points,
        plane_model=floor_result.plane_model,
        resolution=GRID_RESOLUTION_M,
        min_obstacle_height_m=MIN_OBSTACLE_HEIGHT_M,
        max_obstacle_height_m=MAX_OBSTACLE_HEIGHT_M,
        padding_m=0.20,
    )

    occupancy_grid_save_path = (
        OCCUPANCY_MAPS_DIR / f"sample_{sample_index:05d}_occupancy_grid.png"
    )

    save_occupancy_grid_visualization(
        grid=occupancy_result.grid,
        save_path=occupancy_grid_save_path,
        title=f"Occupancy grid, sample {sample_index:05d}",
    )

    print(f"Saved occupancy grid to: {occupancy_grid_save_path}")

    inflated_grid = inflate_obstacles(
        grid=occupancy_result.grid,
        robot_radius_m=ROBOT_RADIUS_M,
        resolution=GRID_RESOLUTION_M,
    )

    traversability_grid_save_path = (
        OCCUPANCY_MAPS_DIR / f"sample_{sample_index:05d}_traversability_grid.png"
    )

    save_occupancy_grid_visualization(
        grid=inflated_grid,
        save_path=traversability_grid_save_path,
        title=f"Traversability grid, sample {sample_index:05d}",
    )

    print(f"Saved traversability grid to: {traversability_grid_save_path}")

    print("\nOccupancy metrics:")
    occupancy_metrics = compute_occupancy_metrics(occupancy_result.grid)
    for key, value in occupancy_metrics.items():
        print(f"{key}: {value}")

    print("\nTraversability metrics:")
    traversability_metrics = compute_occupancy_metrics(inflated_grid)
    for key, value in traversability_metrics.items():
        print(f"{key}: {value}")

    print("\nObstacle points:")
    print(f"obstacle_points: {occupancy_result.obstacle_points.shape[0]}")

if __name__ == "__main__":
    main()