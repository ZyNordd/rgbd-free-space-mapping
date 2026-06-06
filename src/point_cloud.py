from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d


def create_point_cloud_from_rgbd(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    downsample_step: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a colored 3D point cloud from RGB image and depth map.

    Parameters
    ----------
    rgb:
        RGB image with shape (H, W, 3), values in [0, 255].
    depth_m:
        Depth map in meters with shape (H, W).
    fx, fy, cx, cy:
        Camera intrinsic parameters.
    downsample_step:
        Pixel step for downsampling. For example:
        1 - use every pixel;
        2 - use every second pixel;
        4 - use every fourth pixel.

    Returns
    -------
    points:
        Array of 3D points with shape (N, 3).
    colors:
        Array of RGB colors with shape (N, 3), values in [0, 1].
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb.shape}")

    if depth_m.ndim != 2:
        raise ValueError(f"Expected depth map with shape (H, W), got {depth_m.shape}")

    if rgb.shape[:2] != depth_m.shape:
        raise ValueError(f"RGB and depth shapes do not match: rgb={rgb.shape}, depth={depth_m.shape}")

    if downsample_step < 1:
        raise ValueError("downsample_step must be >= 1")

    height, width = depth_m.shape

    v_coords, u_coords = np.mgrid[0:height:downsample_step, 0:width:downsample_step]

    z = depth_m[0:height:downsample_step, 0:width:downsample_step]
    rgb_downsampled = rgb[0:height:downsample_step, 0:width:downsample_step, :]

    valid_mask = np.isfinite(z) & (z > 0)

    u_valid = u_coords[valid_mask].astype(np.float32)
    v_valid = v_coords[valid_mask].astype(np.float32)
    z_valid = z[valid_mask].astype(np.float32)

    x_valid = (u_valid - cx) * z_valid / fx
    y_valid = (v_valid - cy) * z_valid / fy

    points = np.stack([x_valid, y_valid, z_valid], axis=1)

    colors = rgb_downsampled[valid_mask].astype(np.float32) / 255.0

    return points, colors


def create_open3d_point_cloud(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Converts numpy arrays to Open3D PointCloud.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")

    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(f"Expected colors with shape (N, 3), got {colors.shape}")

    if points.shape[0] != colors.shape[0]:
        raise ValueError(
            f"Points and colors must have the same number of rows: "
            f"points={points.shape[0]}, colors={colors.shape[0]}"
        )

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    point_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    return point_cloud


def save_point_cloud_ply(
    points: np.ndarray,
    colors: np.ndarray,
    save_path: str | Path,
) -> None:
    """
    Saves colored point cloud to .ply file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    point_cloud = create_open3d_point_cloud(points, colors)

    success = o3d.io.write_point_cloud(str(save_path), point_cloud)

    if not success:
        raise RuntimeError(f"Failed to save point cloud to {save_path}")


def get_point_cloud_stats(points: np.ndarray) -> dict:
    """
    Returns basic statistics for point cloud.
    """
    if points.size == 0:
        return {
            "num_points": 0,
            "x_min": None,
            "x_max": None,
            "y_min": None,
            "y_max": None,
            "z_min": None,
            "z_max": None,
        }

    return {
        "num_points": int(points.shape[0]),
        "x_min": float(np.min(points[:, 0])),
        "x_max": float(np.max(points[:, 0])),
        "y_min": float(np.min(points[:, 1])),
        "y_max": float(np.max(points[:, 1])),
        "z_min": float(np.min(points[:, 2])),
        "z_max": float(np.max(points[:, 2])),
    }
