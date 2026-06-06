from src.config import (DATA_RAW_DIR, DEPTH_SCALE, FIGURES_DIR, MAX_DEPTH_M,
                        MIN_DEPTH_M)
from src.data_loading import NYU2KaggleDataset
from src.depth_preprocessing import (clean_depth_map, convert_depth_to_meters,
                                     get_depth_stats,
                                     normalize_depth_for_display)
from src.visualization import plot_rgb_and_depth


def main() -> None:
    dataset = NYU2KaggleDataset(
        data_root=DATA_RAW_DIR,
        split="test",
        max_samples=10,
    )

    print(f"Dataset size: {len(dataset)}")

    rgb, depth_raw = dataset[0]
    info = dataset.get_sample_info(0)

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
    stats = get_depth_stats(depth_clean)
    for key, value in stats.items():
        print(f"{key}: {value}")

    save_path = FIGURES_DIR / "sample_00000_rgb_depth.png"

    plot_rgb_and_depth(
        rgb=rgb,
        depth_m=depth_clean,
        normalized_depth=depth_display,
        save_path=save_path,
        title="NYU Depth V2 sample 00000",
    )

    print(f"\nSaved visualization to: {save_path}")


if __name__ == "__main__":
    main()
