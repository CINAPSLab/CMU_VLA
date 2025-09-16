#!/usr/bin/env python3
"""
Fusion utilities for projecting LiDAR points to equirectangular image plane
and estimating simple 3D bounding boxes from in-box points.

Minimal, dependency-light skeleton.
"""

from typing import Tuple
import numpy as np

from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2


def pointcloud2_to_xyz_array(cloud: PointCloud2) -> np.ndarray:
    """Convert sensor_msgs/PointCloud2 to Nx3 float32 ndarray.
    Ignores NaNs and invalid points.
    """
    pts = []
    for p in pc2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True):
        pts.append([p[0], p[1], p[2]])
    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def project_points_equirectangular(points_cam: np.ndarray, width: int, height: int) -> np.ndarray:
    """Project 3D points in camera coords to equirectangular (lon-lat) pixels.

    Assumes camera frame roughly aligned with LiDAR and equirectangular mapping.
    This is an approximation meant as a placeholder; tune as needed.

    Args:
      points_cam: Nx3 (x, y, z)
      width, height: image dimensions
    Returns:
      uv: Nx2 pixel coordinates (float)
    """
    if points_cam.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)

    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    r = np.linalg.norm(points_cam, axis=1) + 1e-9

    lon = np.arctan2(x, z)  # [-pi, pi]
    lat = np.arcsin(np.clip(y / r, -1.0, 1.0))  # [-pi/2, pi/2]

    u = (lon + np.pi) / (2 * np.pi) * width
    v = (lat + (np.pi / 2)) / np.pi * height
    return np.stack([u, v], axis=1).astype(np.float32)


def select_points_in_bbox(uv: np.ndarray, bbox: Tuple[int, int, int, int], width: int, height: int) -> np.ndarray:
    """Return boolean mask of uv points inside the bbox.

    Args:
      uv: Nx2 pixel coords
      bbox: (xmin, ymin, xmax, ymax)
    """
    if uv.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    xmin, ymin, xmax, ymax = bbox
    u, v = uv[:, 0], uv[:, 1]
    mask = (u >= max(0, xmin)) & (u <= min(width - 1, xmax)) & (v >= max(0, ymin)) & (v <= min(height - 1, ymax))
    return mask


def estimate_bbox_3d(points: np.ndarray) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Estimate simple axis-aligned 3D bbox (center, size) from points.

    If sklearn is available, could optionally cluster first. For skeleton, use AABB.
    """
    if points.shape[0] == 0:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    size = pmax - pmin
    center = (pmin + pmax) / 2.0
    return (float(center[0]), float(center[1]), float(center[2])), (
        float(size[0]),
        float(size[1]),
        float(size[2]),
    )

