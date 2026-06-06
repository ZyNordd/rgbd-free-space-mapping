from src.config import (CX, CY, DATA_RAW_DIR, DEPTH_SCALE, DOWNSAMPLE_STEP, FX,
                        FY, MAX_DEPTH_M, MIN_DEPTH_M)
from src.data_loading import NYU2KaggleDataset
from src.depth_preprocessing import clean_depth_map, convert_depth_to_meters
from src.floor_detection import detect_floor_plane_ransac
from src.point_cloud import create_point_cloud_from_rgbd


def process_sample(dataset: NYU2KaggleDataset, index: int) -> dict:
    rgb, depth_raw = dataset[index]

    depth_m = convert_depth_to_meters(depth_raw, depth_scale=DEPTH_SCALE)
    depth_clean = clean_depth_map(
        depth_m,
        min_depth_m=MIN_DEPTH_M,
        max_depth_m=MAX_DEPTH_M,
        median_kernel_size=5,
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

    if floor_result is None:
        return {
            "index": index,
            "status": "not_found",
            "floor_points": 0,
            "inlier_ratio": 0.0,
            "mean_y": None,
            "normal_y_abs": None,
        }

    return {
        "index": index,
        "status": "found",
        "floor_points": int(floor_result.floor_points.shape[0]),
        "inlier_ratio": float(floor_result.inlier_ratio),
        "mean_y": float(floor_result.mean_floor_y),
        "normal_y_abs": float(floor_result.normal_y_abs),
    }


def main() -> None:
    split = "test"
    max_samples = 50

    dataset = NYU2KaggleDataset(
        data_root=DATA_RAW_DIR,
        split=split,
        max_samples=max_samples,
    )

    print(f"Scanning split={split}, samples={len(dataset)}")
    print()

    results = []

    for index in range(len(dataset)):
        try:
            result = process_sample(dataset, index)
            results.append(result)

            print(
                f"index={result['index']:04d} | "
                f"status={result['status']:9s} | "
                f"floor_points={result['floor_points']:6d} | "
                f"ratio={result['inlier_ratio']:.4f} | "
                f"mean_y={result['mean_y']} | "
                f"|normal_y|={result['normal_y_abs']}"
            )

        except Exception as error:
            print(f"index={index:04d} | error: {error}")

    found = [item for item in results if item["status"] == "found"]

    print()
    print(f"Found floor-like planes: {len(found)} / {len(results)}")

    found_sorted = sorted(
        found,
        key=lambda item: (item["inlier_ratio"], item["normal_y_abs"]),
        reverse=True,
    )

    print()
    print("Best candidates:")
    for item in found_sorted[:10]:
        print(
            f"index={item['index']:04d} | "
            f"floor_points={item['floor_points']:6d} | "
            f"ratio={item['inlier_ratio']:.4f} | "
            f"mean_y={item['mean_y']:.4f} | "
            f"|normal_y|={item['normal_y_abs']:.4f}"
        )


if __name__ == "__main__":
    main()
