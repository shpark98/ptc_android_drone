"""Depth map to point cloud conversion and coloring utilities."""

import numpy as np
import cv2
from typing import Tuple, Optional


def depth_to_points(
    depth: np.ndarray,
    K: np.ndarray,
    T_world_cam: np.ndarray,
    max_depth: float = 80.0,
    subsample: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert depth map to 3D points in world coordinates.

    Args:
        depth: (H, W) depth in meters
        K: (3, 3) camera intrinsic matrix
        T_world_cam: (4, 4) camera-to-world transformation
        max_depth: Maximum depth to include
        subsample: Use every Nth pixel (1 = all pixels)

    Returns:
        pts_world: (N, 3) points in world frame
        pixel_indices: (N, 2) corresponding (row, col) in original image
    """
    H, W = depth.shape

    # Create pixel grid
    if subsample > 1:
        rows = np.arange(0, H, subsample)
        cols = np.arange(0, W, subsample)
    else:
        rows = np.arange(H)
        cols = np.arange(W)

    uu, vv = np.meshgrid(cols, rows)  # (Hs, Ws)
    uu = uu.ravel()
    vv = vv.ravel()

    # Sample depth at these pixels
    d = depth[vv, uu]

    # Valid mask: positive depth, below max
    valid = (d > 0) & (d < max_depth) & np.isfinite(d)
    uu = uu[valid]
    vv = vv[valid]
    d = d[valid]

    if len(d) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)

    # Backproject to camera coordinates
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_cam = (uu - cx) * d / fx
    y_cam = (vv - cy) * d / fy
    z_cam = d

    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (N, 3)

    # Transform to world coordinates
    R = T_world_cam[:3, :3]
    t = T_world_cam[:3, 3]
    pts_world = (R @ pts_cam.T).T + t  # (N, 3)

    pixel_indices = np.stack([vv, uu], axis=-1).astype(np.int32)

    return pts_world.astype(np.float32), pixel_indices


def colorize_points(
    image: np.ndarray,
    pixel_indices: np.ndarray,
    depth_values: Optional[np.ndarray] = None,
    mode: str = "rgb",
    depth_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Generate colors for point cloud.

    Args:
        image: (H, W, 3) BGR image
        pixel_indices: (N, 2) pixel locations (row, col)
        depth_values: (N,) depth values (needed for colormap modes)
        mode: "rgb" | "turbo" | "viridis" | "plasma" | "inferno"
        depth_range: (min, max) for consistent colormap normalization

    Returns:
        colors: (N, 3) uint8 RGB colors
    """
    N = len(pixel_indices)
    if N == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    if mode == "rgb":
        rows, cols = pixel_indices[:, 0], pixel_indices[:, 1]
        # BGR -> RGB
        colors = image[rows, cols, ::-1].copy()
        return colors.astype(np.uint8)

    # Colormap modes
    if depth_values is None:
        raise ValueError(f"depth_values required for colormap mode '{mode}'")

    cmap_map = {
        "turbo": cv2.COLORMAP_TURBO,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "plasma": cv2.COLORMAP_PLASMA,
        "inferno": cv2.COLORMAP_INFERNO,
    }
    if mode not in cmap_map:
        raise ValueError(f"Unknown color mode: {mode}. Use: rgb, turbo, viridis, plasma, inferno")

    # Normalize depth to [0, 255]
    if depth_range is not None:
        d_min, d_max = depth_range
    else:
        d_min, d_max = depth_values.min(), depth_values.max()

    if d_max - d_min < 1e-6:
        d_max = d_min + 1.0

    norm = np.clip((depth_values - d_min) / (d_max - d_min), 0, 1)
    gray = (norm * 255).astype(np.uint8)

    # Apply colormap (returns BGR)
    colored = cv2.applyColorMap(gray.reshape(-1, 1), cmap_map[mode])
    # BGR -> RGB, squeeze
    colors = colored[:, 0, ::-1].copy()

    return colors.astype(np.uint8)
