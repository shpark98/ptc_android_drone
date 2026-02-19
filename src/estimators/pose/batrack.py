"""BaTrack-based pose estimator wrapper."""

import sys
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any

from .base import PoseEstimator, PoseResult


class BaTrackEstimator(PoseEstimator):
    """Pose estimator using BaTrack (Bundle-Adjusting Track).

    BaTrack performs visual odometry with bundle adjustment,
    providing camera poses and dense depth maps.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        device: str = "cuda",
    ):
        """Initialize BaTrack estimator.

        Args:
            config_path: Path to BaTrack config file
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device
        self.config_path = config_path

        # Add batrack to path
        batrack_root = Path(__file__).parent.parent.parent.parent / "external" / "batrack"
        if str(batrack_root) not in sys.path:
            sys.path.insert(0, str(batrack_root))

        self._model = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization of BaTrack model."""
        if self._initialized:
            return

        from main.batrack import BATRACK as BaTrack
        from omegaconf import OmegaConf

        # Load config
        if self.config_path:
            cfg = OmegaConf.load(self.config_path)
        else:
            cfg = OmegaConf.create({
                "model": {
                    "depth_model": "depth_anything_v2",
                    "depth_model_vitl": True,
                },
                "slam": {
                    "frontend": {
                        "num_iters": 4,
                        "window_size": 5,
                    },
                    "backend": {
                        "num_iters": 2,
                    }
                }
            })

        self._model = BaTrack(cfg, self.device)
        self._initialized = True

    def reset(self):
        """Reset the tracker state for a new sequence."""
        if self._model is not None:
            self._model.reset()

    def estimate(
        self,
        img0: np.ndarray,
        img1: np.ndarray,
        depth0: np.ndarray,
        depth1: np.ndarray,
        flow: np.ndarray,
        K: Optional[np.ndarray] = None,
        baseline: Optional[float] = None,
    ) -> PoseResult:
        """Estimate relative pose using BaTrack.

        Note: BaTrack is designed for video sequences, not image pairs.
        For pairwise estimation, it processes both frames sequentially.

        Args:
            img0: First image (H, W, 3) BGR uint8
            img1: Second image (H, W, 3) BGR uint8
            depth0: Depth map for first image (H, W) - not used, BaTrack estimates depth
            depth1: Depth map for second image (H, W) - not used
            flow: Optical flow from img0 to img1 (H, W, 2) - not used
            K: Camera intrinsics (3, 3)
            baseline: Ground truth baseline (optional)

        Returns:
            PoseResult with R, t, and estimated depth
        """
        self._lazy_init()

        try:
            # Reset for pair-wise estimation
            self.reset()

            # Convert BGR to RGB
            rgb0 = img0[:, :, ::-1].copy()
            rgb1 = img1[:, :, ::-1].copy()

            # Process first frame
            self._model.track(rgb0, K)

            # Process second frame
            result = self._model.track(rgb1, K)

            if result is None or len(self._model.poses) < 2:
                return PoseResult(
                    R=np.eye(3),
                    t=np.zeros(3),
                    success=False,
                    num_inliers=0,
                    extra={"method": "batrack", "error": "tracking_failed"}
                )

            # Get relative pose (T_1_0: transform from frame 0 to frame 1)
            pose0 = self._model.poses[0]  # T_w_0
            pose1 = self._model.poses[1]  # T_w_1

            # Relative pose: T_1_0 = T_1_w @ T_w_0 = inv(T_w_1) @ T_w_0
            T_w_0 = np.eye(4)
            T_w_0[:3, :3] = pose0[:3, :3]
            T_w_0[:3, 3] = pose0[:3, 3]

            T_w_1 = np.eye(4)
            T_w_1[:3, :3] = pose1[:3, :3]
            T_w_1[:3, 3] = pose1[:3, 3]

            T_rel = np.linalg.inv(T_w_1) @ T_w_0

            R = T_rel[:3, :3]
            t = T_rel[:3, 3]

            # Get estimated depth
            est_depth = None
            if hasattr(self._model, 'depths') and len(self._model.depths) > 0:
                est_depth = self._model.depths[-1]

            return PoseResult(
                R=R,
                t=t,
                success=True,
                num_inliers=-1,  # BaTrack doesn't report inliers
                extra={
                    "method": "batrack",
                    "estimated_depth": est_depth,
                }
            )

        except Exception as e:
            return PoseResult(
                R=np.eye(3),
                t=np.zeros(3),
                success=False,
                num_inliers=0,
                extra={"method": "batrack", "error": str(e)}
            )

    @property
    def name(self) -> str:
        return "BaTrack"

    @property
    def needs_baseline(self) -> bool:
        return False  # BaTrack estimates scale
