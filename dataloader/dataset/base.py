"""
Base dataset class for all dataset loaders.
Defines the common interface that all loaders should implement.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np


class BaseDataset(ABC):
    """
    Abstract base class for all dataset loaders.

    All loaders should implement:
        - get_intrinsics() -> (fx, fy, cx, cy)
        - get(idx) -> dict with 'image', 'depth', 'position', 'rpy', etc.
        - __len__() -> number of frames

    Optional methods (with default implementations):
        - get_pose_matrix(idx) -> 4x4 transformation matrix
        - get_baseline(idx) -> translation magnitude between frames
        - get_relative_pose(idx) -> (R, t) relative pose between consecutive frames
    """

    @abstractmethod
    def get_intrinsics(self) -> Tuple[float, float, float, float]:
        """
        Get camera intrinsics.

        Returns:
            Tuple of (fx, fy, cx, cy)
        """
        pass

    @abstractmethod
    def get(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Get data for a single frame.

        Args:
            idx: Frame index

        Returns:
            Dictionary containing:
                - 'image': BGR image (H, W, 3) uint8
                - 'image_og': Original (uncropped) BGR image
                - 'depth': Depth map (H, W) float32, may be None
                - 'depth_og': Original depth map, may be None
                - 'position': Camera position [x, y, z]
                - 'rpy': Camera orientation [roll, pitch, yaw] in radians
            Returns None if frame cannot be loaded.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of frames in the dataset."""
        pass

    def get_pose_matrix(self, idx: int) -> np.ndarray:
        """
        Get 4x4 camera pose matrix (world <- camera transformation).

        Args:
            idx: Frame index

        Returns:
            4x4 transformationma trix T_world_camera
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_pose_matrix()"
        )

    def get_baseline(self, idx: int, clamp_y: bool = False, max_y_ratio: float = 0.3) -> float:
        """
        Get translation magnitude (baseline) between frame idx-1 and idx.

        Args:
            idx: Current frame index (must be >= 1)
            clamp_y: If True, clamp Y component to prevent GPS height spikes
            max_y_ratio: Maximum ratio of Y to XZ magnitude (default 0.3 = ~17 degrees)

        Returns:
            Baseline in meters
        """
        if idx < 1:
            return 0.0

        _, t_rel = self.get_relative_pose(idx)
        t = t_rel.copy()

        # Clamp Y (height) component to prevent GPS spikes
        if clamp_y:
            xz_mag = np.sqrt(t[0]**2 + t[2]**2)
            max_y = xz_mag * max_y_ratio
            t[1] = np.clip(t[1], -max_y, max_y)

        return float(np.linalg.norm(t))

    def get_relative_pose(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get relative pose between frame idx-1 and idx.

        Standard convention: T_rel transforms points from prev frame to curr frame.
        p_curr = R @ p_prev + t

        Args:
            idx: Current frame index (must be >= 1)

        Returns:
            Tuple of (R, t):
                - R: 3x3 rotation matrix
                - t: 3D translation vector (metric, in meters)
        """
        if idx < 1:
            return np.eye(3), np.zeros(3)

        T_curr = self.get_pose_matrix(idx)
        T_prev = self.get_pose_matrix(idx - 1)
        T_rel = np.linalg.inv(T_curr) @ T_prev

        R_rel = T_rel[:3, :3]
        t_rel = T_rel[:3, 3]

        return R_rel, t_rel

    def get_K_matrix(self) -> np.ndarray:
        """
        Get 3x3 camera intrinsic matrix K.

        Returns:
            3x3 intrinsic matrix
        """
        fx, fy, cx, cy = self.get_intrinsics()
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
