"""
Unified DataLoader wrapper for all dataset loaders.

Provides a consistent interface for loading data from different datasets
with common functionality like iteration, baseline computation, etc.
"""
from typing import Optional, Dict, Any, Iterator, Tuple
from .dataset import (
    BaseDataset,
    KITTIEigenSplit,
    MS2Loader,
    TartanairLoader,
    CampusLoader,
    WheelLoader,
    ETH3DLoader,
    EuRoCLoader,
    VOIDLoader,
)


# Registry of available datasets
DATASET_REGISTRY = {
    "kitti": KITTIEigenSplit,
    "ms2": MS2Loader,
    "tartanair": TartanairLoader,
    "campus": CampusLoader,
    "wheel": WheelLoader,
    "eth3d": ETH3DLoader,
    "euroc": EuRoCLoader,
    "void": VOIDLoader,
}


class DataLoader:
    """
    Unified wrapper for all dataset loaders.

    Example:
        # Using KITTI
        loader = DataLoader("kitti", {
            "rgb_path": "/path/to/KITTI_RGB",
            "date": "2011_09_26",
            "drive": "0001",
            "depth_path": "/path/to/KITTI_Depth",
        })

        # Basic info
        print(f"Total frames: {len(loader)}")
        fx, fy, cx, cy = loader.get_intrinsics()

        # Get single frame
        data = loader[10]

        # Iterate over all frames
        for idx, data in loader.iter_frames():
            process(data['image'])

        # Iterate over consecutive pairs (for depth refinement)
        for pair in loader.iter_pairs(start=1, end=100):
            baseline = pair['baseline']
            curr_img = pair['curr']['image_og']
    """

    def __init__(self, dataset_name: str, args: Optional[Dict] = None):
        """
        Initialize DataLoader with a specific dataset.

        Args:
            dataset_name: One of 'kitti', 'ms2', 'tartanair', 'campus',
                         'wheel', 'eth3d', 'euroc', 'void'
            args: Dataset-specific arguments as a dictionary
        """
        if dataset_name not in DATASET_REGISTRY:
            available = ", ".join(DATASET_REGISTRY.keys())
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")

        args = args or {}
        self.dataset_name = dataset_name
        self.dataset: BaseDataset = DATASET_REGISTRY[dataset_name](**args)

    def __len__(self) -> int:
        """Return total number of frames."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get data for a single frame (same as dataset.get())."""
        return self.dataset.get(idx)

    def get_intrinsics(self) -> Tuple[float, float, float, float]:
        """Get camera intrinsics (fx, fy, cx, cy)."""
        return self.dataset.get_intrinsics()

    def get(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get data for a single frame."""
        return self.dataset.get(idx)

    def get_baseline(self, idx: int) -> float:
        """Get baseline (translation magnitude) between frame idx-1 and idx."""
        return self.dataset.get_baseline(idx)

    def get_relative_pose(self, idx: int) -> Tuple:
        """Get relative pose (R, t) between frame idx-1 and idx."""
        return self.dataset.get_relative_pose(idx)

    def get_pose_matrix(self, idx: int):
        """Get 4x4 camera pose matrix for frame idx."""
        return self.dataset.get_pose_matrix(idx)

    def iter_frames(
        self,
        start: int = 0,
        end: Optional[int] = None,
        step: int = 1
    ) -> Iterator[Tuple[int, Dict[str, Any]]]:
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
        for idx in range(start, min(end, len(self)), step):
            data = self.get(idx)
            if data is not None:
                yield idx, data

    def iter_pairs(
        self,
        start: int = 1,
        end: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Iterate over consecutive frame pairs with motion information.

        This is the primary iterator for depth refinement evaluation.

        Args:
            start: Start frame index (must be >= 1)
            end: End frame index (exclusive)

        Yields:
            Dictionary with:
                - 'idx': Current frame index
                - 'prev': Previous frame data dict
                - 'curr': Current frame data dict
                - 'baseline': Translation magnitude (meters)
                - 'R_rel': 3x3 relative rotation matrix
                - 't_rel': 3D unit translation vector
        """
        if end is None:
            end = len(self)

        for idx in range(max(1, start), min(end, len(self))):
            prev_data = self.get(idx - 1)
            curr_data = self.get(idx)

            if prev_data is None or curr_data is None:
                continue

            try:
                R_rel, t_rel = self.get_relative_pose(idx)
                baseline = self.get_baseline(idx)
            except NotImplementedError:
                # Dataset doesn't support pose - use zeros
                import numpy as np
                R_rel = np.eye(3)
                t_rel = np.zeros(3)
                baseline = 0.0

            yield {
                'idx': idx,
                'prev': prev_data,
                'curr': curr_data,
                'baseline': baseline,
                'R_rel': R_rel,
                't_rel': t_rel,
            }

    # Legacy methods for backwards compatibility
    def load_data_3frame(self, idx: int) -> Dict[str, Any]:
        """Load 3 consecutive frames (legacy interface)."""
        return {
            "prev": self.get(idx),
            "curr": self.get(idx + 1),
            "next": self.get(idx + 2)
        }

    def load_data(self, idx: int) -> Dict[str, Any]:
        """Load single frame (legacy interface)."""
        return {"prev": self.get(idx)}


def create_kitti_loader(
    rgb_path: str = "/home/nas/Dataset2/KITTI/KITTI_RGB_Image",
    date: str = "2011_09_26",
    drive: str = "0001",
    depth_path: Optional[str] = None,
    data_type: str = "pointcloud"
) -> DataLoader:
    """
    Convenience function to create a KITTI DataLoader.

    Args:
        rgb_path: Path to KITTI RGB images
        date: Recording date (e.g., "2011_09_26")
        drive: Drive number (e.g., "0001")
        depth_path: Path to depth GT (uses rgb_path if None)
        data_type: "pointcloud" or "densemap"

    Returns:
        DataLoader configured for KITTI
    """
    if depth_path is None:
        depth_path = rgb_path

    return DataLoader("kitti", {
        "rgb_path": rgb_path,
        "date": date,
        "drive": drive,
        "depth_path": depth_path,
        "data_type": data_type,
    })
