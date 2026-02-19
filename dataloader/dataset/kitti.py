"""
KITTI dataset loader with convenience methods for depth evaluation.
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
import pykitti

from .base import BaseDataset


# KITTI Odometry sequence mapping: seq_id -> (date, drive, start_frame, end_frame)
KITTI_ODOM_SEQUENCES = {
    "00": ("2011_10_03", "0027", 0, 4540),
    "01": ("2011_10_03", "0042", 0, 1100),
    "02": ("2011_10_03", "0034", 0, 4660),
    "03": ("2011_09_26", "0067", 0, 800),
    "04": ("2011_09_30", "0016", 0, 270),
    "05": ("2011_09_30", "0018", 0, 2760),
    "06": ("2011_09_30", "0020", 0, 1100),
    "07": ("2011_09_30", "0027", 0, 1100),
    "08": ("2011_09_30", "0028", 1100, 5170),
    "09": ("2011_09_30", "0033", 0, 1590),
    "10": ("2011_09_30", "0034", 0, 1200),
}


def _get_odom_seq_id(date: str, drive: str) -> Optional[str]:
    """Get odometry sequence ID from date and drive."""
    for seq_id, (d, dr, _, _) in KITTI_ODOM_SEQUENCES.items():
        if d == date and dr == drive:
            return seq_id
    return None


class KITTIEigenSplit(BaseDataset):
    """
    KITTI dataset loader following Eigen split conventions.

    Provides convenient access to:
        - RGB images (center-cropped and original)
        - GT depth maps (sparse or dense)
        - Camera poses from OXTS or VO GT (if available)
        - Baseline and relative pose computation

    Example:
        dataset = KITTIEigenSplit(
            rgb_path="/path/to/KITTI_RGB",
            date="2011_09_26",
            drive="0001",
            depth_path="/path/to/KITTI_Depth",
            data_type="pointcloud"
        )

        # Basic usage
        print(f"Total frames: {len(dataset)}")
        fx, fy, cx, cy = dataset.get_intrinsics()

        # Get frame data
        data = dataset.get(10)
        img = data['image_og']  # Original resolution
        depth_gt = data['depth_og']  # GT depth

        # Get baseline and relative pose for frame 10 vs 9
        baseline = dataset.get_baseline(10)  # meters
        R_rel, t_rel = dataset.get_relative_pose(10)
    """

    CROP_H: int = 352
    CROP_W: int = 1216

    def __init__(
        self,
        rgb_path: str,
        date: str,
        drive: str,
        depth_path: Optional[str] = None,
        data_type: str = "pointcloud",  # "pointcloud"(.png) or "densemap"(.npy)
        pose_source: str = "auto",  # "auto", "oxts", or "vo_gt"
        vo_gt_path: Optional[str] = None,  # Path to VO GT poses directory
    ) -> None:
        """
        Initialize KITTI dataset loader.

        Args:
            rgb_path: Path to KITTI RGB images (e.g., /path/to/KITTI_RGB_Image)
            date: Recording date (e.g., "2011_09_26")
            drive: Drive number (e.g., "0001")
            depth_path: Path to KITTI depth GT. If None, uses rgb_path.
            data_type: "pointcloud" for sparse .png, "densemap" for dense .npy
            pose_source: "auto" (prefer VO GT), "oxts", or "vo_gt"
            vo_gt_path: Path to VO GT poses directory. If None, uses dataloader/gt_poses/
        """
        self.raw = pykitti.raw(rgb_path, date, drive)
        self.date = date
        self.drive = drive
        self.pose_source_requested = pose_source

        # Depth directory
        if depth_path is None:
            depth_path = rgb_path
        self.depth_dir = os.path.join(
            depth_path, f"{date}_drive_{drive}_sync", "proj_depth", "groundtruth", "image_02"
        )

        if data_type not in ("pointcloud", "densemap"):
            raise ValueError("data_type must be 'pointcloud' or 'densemap'")
        self.data_type = data_type

        # Setup pose source
        self.vo_gt_poses: Optional[List[np.ndarray]] = None
        self.vo_gt_start_frame: int = 0
        self.pose_source_used: str = "oxts"  # Will be updated if VO GT is loaded

        # Try to load VO GT poses
        odom_seq_id = _get_odom_seq_id(date, drive)
        if pose_source in ("auto", "vo_gt") and odom_seq_id is not None:
            # Default path: dataloader/gt_poses/
            if vo_gt_path is None:
                vo_gt_path = str(Path(__file__).parent.parent / "gt_poses")

            vo_gt_file = os.path.join(vo_gt_path, f"{odom_seq_id}.txt")
            if os.path.exists(vo_gt_file):
                self.vo_gt_poses = self._load_vo_gt_poses(vo_gt_file)
                self.vo_gt_start_frame = KITTI_ODOM_SEQUENCES[odom_seq_id][2]
                self.pose_source_used = "vo_gt"
                print(f"[KITTI] Using VO GT poses for sequence {odom_seq_id} ({date}/{drive})")
            elif pose_source == "vo_gt":
                raise FileNotFoundError(f"VO GT file not found: {vo_gt_file}")

        # OXTS camera pose computation (lazy)
        self.T_imu_cam2 = np.linalg.inv(self.raw.calib.T_cam2_imu)
        self._oxts_pose_cache: Dict[int, np.ndarray] = {}

        if self.pose_source_used == "oxts":
            print(f"[KITTI] Using OXTS poses for {date}/{drive}")

        # Cache for lazy-loaded depths
        self._depth_cache: Dict[int, Optional[np.ndarray]] = {}

    @property
    def name(self) -> str:
        """Dataset name for display."""
        pose_info = f"[{self.pose_source_used}]"
        return f"KITTI_{self.date}_{self.drive} {pose_info}"

    def _load_vo_gt_poses(self, pose_file: str) -> List[np.ndarray]:
        """Load VO GT poses from KITTI odometry format file.

        Each line contains 12 values: flattened 3x4 transformation matrix (row-major).
        """
        poses = []
        with open(pose_file, 'r') as f:
            for line in f:
                values = [float(v) for v in line.strip().split()]
                if len(values) != 12:
                    continue
                # Reshape to 3x4, then extend to 4x4
                T = np.eye(4)
                T[:3, :] = np.array(values).reshape(3, 4)
                poses.append(T)
        return poses

    def __len__(self) -> int:
        """Return total number of frames."""
        return len(self.raw)

    def _load_depth(self, idx: int) -> Optional[np.ndarray]:
        """Lazy-load a single depth map with caching."""
        if idx in self._depth_cache:
            return self._depth_cache[idx]

        ext = "png" if self.data_type == "pointcloud" else "npy"
        path = os.path.join(self.depth_dir, f"{idx:010d}.{ext}")

        if not os.path.exists(path):
            self._depth_cache[idx] = None
            return None

        if ext == "png":
            depth = np.array(Image.open(path)).astype(np.float32).squeeze() / 256.0
        else:  # npy dense map
            depth = np.load(path).astype(np.float32).squeeze() / 256.0

        self._depth_cache[idx] = depth
        return depth

    def get_intrinsics(self) -> Tuple[float, float, float, float]:
        """Get camera intrinsics (fx, fy, cx, cy)."""
        K = self.raw.calib.K_cam2
        return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    def _get_oxts_pose(self, idx: int) -> np.ndarray:
        """Lazy-load OXTS camera pose with caching."""
        if idx in self._oxts_pose_cache:
            return self._oxts_pose_cache[idx]

        if idx < 0 or idx >= len(self.raw.oxts):
            raise IndexError(f"Frame index {idx} out of range [0, {len(self.raw.oxts)})")

        # Compute T_world_cam from OXTS
        T_world_cam = self.raw.oxts[idx].T_w_imu @ self.T_imu_cam2
        self._oxts_pose_cache[idx] = T_world_cam
        return T_world_cam

    def get_pose_matrix(self, idx: int) -> np.ndarray:
        """
        Get 4x4 camera pose matrix (world <- camera).

        Uses VO GT if available, otherwise falls back to OXTS.

        Args:
            idx: Frame index

        Returns:
            4x4 transformation matrix T_world_camera
        """
        # Use VO GT if available
        if self.vo_gt_poses is not None:
            vo_idx = idx - self.vo_gt_start_frame
            if 0 <= vo_idx < len(self.vo_gt_poses):
                return self.vo_gt_poses[vo_idx].copy()

        # Fallback to OXTS (lazy-loaded)
        return self._get_oxts_pose(idx).copy()

    @staticmethod
    def T_to_xyzrpy(T: np.ndarray, degrees: bool = False) -> Tuple[List[float], List[float]]:
        """
        Convert 4x4 transformation matrix to position and orientation.

        Args:
            T: 4x4 transformation matrix
            degrees: If True, return angles in degrees

        Returns:
            Tuple of ([x, y, z], [roll, pitch, yaw])
        """
        rot = R.from_matrix(T[:3, :3])
        roll, pitch, yaw = rot.as_euler("xyz", degrees=degrees)
        x, y, z = T[:3, 3]
        return [x, y, z], [roll, pitch, yaw]

    @staticmethod
    def _center_crop(img: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
        """Center crop to (crop_h, crop_w)."""
        h, w = img.shape[:2]
        if h < crop_h or w < crop_w:
            raise ValueError(f"Cannot center-crop ({h},{w}) to ({crop_h},{crop_w}).")
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        return img[top : top + crop_h, left : left + crop_w].copy()

    def get(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Get data for a single frame.

        Returns:
            Dictionary with keys:
                - 'image': Center-cropped BGR image
                - 'image_og': Original BGR image
                - 'depth': Center-cropped GT depth (or None)
                - 'depth_og': Original GT depth (or None)
                - 'position': [x, y, z] camera position
                - 'rpy': [roll, pitch, yaw] in radians
        """
        try:
            img_np = np.array(self.raw.get_cam2(idx))
            img_og = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            depth_og = self._load_depth(idx)  # may be None
            pos, rpy = self.T_to_xyzrpy(self.get_pose_matrix(idx))

            img = self._center_crop(img_og, self.CROP_H, self.CROP_W)
            depth = self._center_crop(depth_og, self.CROP_H, self.CROP_W) if depth_og is not None else None

            return {
                "image": img,
                "image_og": img_og,
                "depth": depth,
                "depth_og": depth_og,
                "position": pos,
                "rpy": rpy,
            }
        except (FileNotFoundError, OSError):
            return None

    # =========================================================================
    # Convenience methods for evaluation
    # =========================================================================

    def get_image(self, idx: int) -> np.ndarray:
        """Get original (uncropped) BGR image."""
        img_np = np.array(self.raw.get_cam2(idx))
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    def get_depth_gt(self, idx: int) -> Optional[np.ndarray]:
        """Get original GT depth map (None if not available)."""
        return self._load_depth(idx)

    def get_valid_depth_mask(self, idx: int) -> Optional[np.ndarray]:
        """Get valid mask for GT depth (depth > 0)."""
        depth = self._load_depth(idx)
        if depth is None:
            return None
        return depth > 0

    def get_image_size(self) -> Tuple[int, int]:
        """Get original image size (H, W)."""
        sample = self.get(0)
        if sample is None:
            return (375, 1242)  # Default KITTI size
        return sample['image_og'].shape[:2]

    def iter_frames(self, start: int = 0, end: Optional[int] = None, step: int = 1):
        """
        Iterate over frames.

        Args:
            start: Start frame index
            end: End frame index (exclusive). If None, uses len(dataset).
            step: Step size

        Yields:
            Tuple of (idx, data_dict)
        """
        if end is None:
            end = len(self)
        for idx in range(start, end, step):
            data = self.get(idx)
            if data is not None:
                yield idx, data

    def iter_pairs(self, start: int = 1, end: Optional[int] = None):
        """
        Iterate over consecutive frame pairs with baseline info.

        Args:
            start: Start frame index (must be >= 1)
            end: End frame index (exclusive)

        Yields:
            Dictionary with:
                - 'idx': Current frame index
                - 'prev': Previous frame data
                - 'curr': Current frame data
                - 'baseline': Translation magnitude
                - 'R_rel': 3x3 relative rotation
                - 't_rel': 3D unit translation vector
        """
        if end is None:
            end = len(self)

        for idx in range(max(1, start), end):
            prev_data = self.get(idx - 1)
            curr_data = self.get(idx)

            if prev_data is None or curr_data is None:
                continue

            R_rel, t_rel = self.get_relative_pose(idx)
            baseline = self.get_baseline(idx)

            yield {
                'idx': idx,
                'prev': prev_data,
                'curr': curr_data,
                'baseline': baseline,
                'R_rel': R_rel,
                't_rel': t_rel,
            }
