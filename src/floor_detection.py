from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import open3d as o3d

from src.point_cloud import create_open3d_point_cloud


@dataclass
class PlaneCandidate:
    plane_model: np.ndarray
    indices: np.ndarray
    inlier_ratio: float
    mean_y: float
    normal_y_abs: float
    num_inliers: int


@dataclass
class PlaneDetectionResult:
    plane_model: np.ndarray
    floor_indices: np.ndarray
    non_floor_indices: np.ndarray
    floor_points: np.ndarray
    floor_colors: np.ndarray
    non_floor_points: np.ndarray
    non_floor_colors: np.ndarray
    inlier_ratio: float
    mean_floor_y: float
    normal_y_abs: float
    candidates: List[PlaneCandidate]


def _normalize_plane_model(plane_model: np.ndarray) -> np.ndarray:
    """
    Normalizes plane model so that normal vector has unit length.

    Plane equation:
        ax + by + cz + d = 0
    """
    plane_model = np.asarray(plane_model, dtype=np.float64)
    normal = plane_model[:3]
    norm = np.linalg.norm(normal)

    if norm < 1e-12:
        raise ValueError("Plane normal has near-zero norm")

    return plane_model / norm


def _segment_single_plane(
    points: np.ndarray,
    colors: np.ndarray,
    distance_threshold: float,
    ransac_n: int,
    num_iterations: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segments one dominant plane from point cloud using Open3D RANSAC.
    Returns normalized plane model and local inlier indices.
    """
    point_cloud = create_open3d_point_cloud(points, colors)

    plane_model, inliers = point_cloud.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )

    plane_model = _normalize_plane_model(np.asarray(plane_model, dtype=np.float64))

    return plane_model, np.asarray(inliers, dtype=np.int64)


def extract_plane_candidates_ransac(
    points: np.ndarray,
    colors: np.ndarray,
    distance_threshold: float = 0.03,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    max_planes: int = 8,
    min_inlier_ratio: float = 0.02,
) -> List[PlaneCandidate]:
    """
    Extracts several large plane candidates from point cloud.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")

    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(f"Expected colors with shape (N, 3), got {colors.shape}")

    if points.shape[0] != colors.shape[0]:
        raise ValueError("points and colors must have the same number of rows")

    total_points = points.shape[0]

    remaining_points = points.copy()
    remaining_colors = colors.copy()
    remaining_original_indices = np.arange(total_points)

    candidates: List[PlaneCandidate] = []

    for _ in range(max_planes):
        if remaining_points.shape[0] < ransac_n:
            break

        plane_model, local_inliers = _segment_single_plane(
            points=remaining_points,
            colors=remaining_colors,
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )

        if local_inliers.size == 0:
            break

        original_inliers = remaining_original_indices[local_inliers]
        inlier_points = points[original_inliers]

        inlier_ratio = local_inliers.size / total_points

        if inlier_ratio >= min_inlier_ratio:
            normal_y_abs = float(abs(plane_model[1]))
            mean_y = float(np.mean(inlier_points[:, 1]))

            candidates.append(
                PlaneCandidate(
                    plane_model=plane_model,
                    indices=original_inliers,
                    inlier_ratio=float(inlier_ratio),
                    mean_y=mean_y,
                    normal_y_abs=normal_y_abs,
                    num_inliers=int(local_inliers.size),
                )
            )

        mask = np.ones(remaining_points.shape[0], dtype=bool)
        mask[local_inliers] = False

        remaining_points = remaining_points[mask]
        remaining_colors = remaining_colors[mask]
        remaining_original_indices = remaining_original_indices[mask]

    return candidates


def detect_floor_plane_ransac(
    points: np.ndarray,
    colors: np.ndarray,
    distance_threshold: float = 0.03,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    max_planes: int = 8,
    min_inlier_ratio: float = 0.02,
    min_normal_y_abs: float = 0.45,
) -> Optional[PlaneDetectionResult]:
    """
    Detects floor-like plane in point cloud.

    Improved strategy:
    1. Extract several large planes using RANSAC.
    2. Keep only planes with sufficiently large Y component of normal.
       This rejects vertical walls/cabinets.
    3. Among remaining candidates choose the one with largest mean Y,
       because in this camera coordinate system Y grows downward.

    Parameters
    ----------
    min_normal_y_abs:
        Minimal absolute Y component of plane normal.
        Larger value means stronger preference for horizontal planes.
        Useful range: 0.35 - 0.75.
    """
    total_points = points.shape[0]

    if total_points == 0:
        return None

    candidates = extract_plane_candidates_ransac(
        points=points,
        colors=colors,
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
        max_planes=max_planes,
        min_inlier_ratio=min_inlier_ratio,
    )

    if len(candidates) == 0:
        return None

    floor_like_candidates = [candidate for candidate in candidates if candidate.normal_y_abs >= min_normal_y_abs]

    if len(floor_like_candidates) == 0:
        return None

    best_candidate = max(
        floor_like_candidates,
        key=lambda item: (
            item.mean_y,
            item.inlier_ratio,
            item.normal_y_abs,
        ),
    )

    floor_indices = np.sort(best_candidate.indices)
    floor_mask = np.zeros(total_points, dtype=bool)
    floor_mask[floor_indices] = True

    non_floor_indices = np.where(~floor_mask)[0]

    floor_points = points[floor_indices]
    floor_colors = colors[floor_indices]

    non_floor_points = points[non_floor_indices]
    non_floor_colors = colors[non_floor_indices]

    return PlaneDetectionResult(
        plane_model=best_candidate.plane_model,
        floor_indices=floor_indices,
        non_floor_indices=non_floor_indices,
        floor_points=floor_points,
        floor_colors=floor_colors,
        non_floor_points=non_floor_points,
        non_floor_colors=non_floor_colors,
        inlier_ratio=best_candidate.inlier_ratio,
        mean_floor_y=best_candidate.mean_y,
        normal_y_abs=best_candidate.normal_y_abs,
        candidates=candidates,
    )


def compute_signed_distances_to_plane(points: np.ndarray, plane_model: np.ndarray) -> np.ndarray:
    """
    Computes signed distances from points to plane.

    Plane equation:
        ax + by + cz + d = 0
    """
    plane_model = _normalize_plane_model(plane_model)

    distances = points @ plane_model[:3] + plane_model[3]

    return distances


def print_plane_candidates(candidates: List[PlaneCandidate]) -> None:
    """
    Prints all extracted plane candidates.
    """
    if len(candidates) == 0:
        print("No plane candidates found.")
        return

    print("Plane candidates:")
    for i, candidate in enumerate(candidates):
        a, b, c, d = candidate.plane_model
        print(
            f"[{i}] "
            f"points={candidate.num_inliers}, "
            f"ratio={candidate.inlier_ratio:.4f}, "
            f"mean_y={candidate.mean_y:.4f}, "
            f"|normal_y|={candidate.normal_y_abs:.4f}, "
            f"plane={a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f}=0"
        )


def print_plane_result(result: Optional[PlaneDetectionResult]) -> None:
    """
    Prints floor plane detection result.
    """
    if result is None:
        print("Floor plane was not detected.")
        return

    a, b, c, d = result.plane_model

    print("Floor plane detection result:")
    print(f"plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    print(f"floor points: {result.floor_points.shape[0]}")
    print(f"non-floor points: {result.non_floor_points.shape[0]}")
    print(f"inlier ratio: {result.inlier_ratio:.4f}")
    print(f"mean floor Y: {result.mean_floor_y:.4f}")
    print(f"|normal Y|: {result.normal_y_abs:.4f}")
