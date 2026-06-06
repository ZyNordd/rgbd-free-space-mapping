import cv2
import numpy as np


def convert_depth_to_meters(depth_raw: np.ndarray, depth_scale: float = 1000.0) -> np.ndarray:
    """
    Converts raw depth map to meters.

    For the Kaggle NYU2 dataset, depth values usually look like millimeters:
    1798 -> 1.798 m.
    """
    if depth_raw.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got shape {depth_raw.shape}")

    depth_m = depth_raw.astype(np.float32) / float(depth_scale)
    return depth_m


def clean_depth_map(
    depth_m: np.ndarray,
    min_depth_m: float = 0.5,
    max_depth_m: float = 5.0,
    median_kernel_size: int = 5,
) -> np.ndarray:
    """
    Cleans depth map:
    - removes invalid values;
    - clips values outside depth range;
    - applies median filtering.

    Invalid pixels are stored as NaN.
    """
    if depth_m.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got shape {depth_m.shape}")

    cleaned = depth_m.astype(np.float32).copy()

    invalid_mask = (
        ~np.isfinite(cleaned)
        | (cleaned <= 0)
        | (cleaned < min_depth_m)
        | (cleaned > max_depth_m)
    )

    cleaned[invalid_mask] = np.nan

    filtered_input = np.nan_to_num(cleaned, nan=0.0).astype(np.float32)

    if median_kernel_size is not None and median_kernel_size > 1:
        filtered = cv2.medianBlur(filtered_input, median_kernel_size)
    else:
        filtered = filtered_input

    # Remove invalid values again after filtering.
    filtered[
        (~np.isfinite(filtered))
        | (filtered <= 0)
        | (filtered < min_depth_m)
        | (filtered > max_depth_m)
    ] = np.nan

    return filtered.astype(np.float32)


def normalize_depth_for_display(depth_m: np.ndarray) -> np.ndarray:
    """
    Normalizes depth map to [0, 1] for visualization.
    NaN values are displayed as 0.
    """
    valid_mask = np.isfinite(depth_m)

    if not np.any(valid_mask):
        return np.zeros_like(depth_m, dtype=np.float32)

    min_val = np.nanmin(depth_m)
    max_val = np.nanmax(depth_m)

    if max_val - min_val < 1e-6:
        return np.zeros_like(depth_m, dtype=np.float32)

    normalized = (depth_m - min_val) / (max_val - min_val)
    normalized = np.nan_to_num(normalized, nan=0.0)
    normalized = np.clip(normalized, 0.0, 1.0)

    return normalized.astype(np.float32)


def get_depth_stats(depth_m: np.ndarray) -> dict:
    """
    Returns basic statistics for valid depth values.
    """
    valid = depth_m[np.isfinite(depth_m)]

    if valid.size == 0:
        return {
            "valid_pixels": 0,
            "min_m": None,
            "max_m": None,
            "mean_m": None,
            "median_m": None,
        }

    return {
        "valid_pixels": int(valid.size),
        "min_m": float(np.min(valid)),
        "max_m": float(np.max(valid)),
        "mean_m": float(np.mean(valid)),
        "median_m": float(np.median(valid)),
    }