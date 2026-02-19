"""MADPose pose estimator wrapper."""

from typing import Optional
import numpy as np
from .base import PoseEstimator, PoseResult


def _make_grid_points(W: int, H: int, stride: int) -> np.ndarray:
    """Create grid points for sampling."""
    xs = np.arange(0, W, stride, dtype=np.float32)
    ys = np.arange(0, H, stride, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


class MADPoseEstimator(PoseEstimator):
    """MADPose estimator using HybridEstimatePoseScaleOffset.

    MADPose internally estimates scale and offset from depth, so it does
    NOT need external baseline. It requires metric depth (e.g., from UniDepth).
    """

    def __init__(
        self,
        K: np.ndarray,
        grid_stride: int = 12,
        reproj_pix_thres: float = 5.0,
        epipolar_pix_thres: float = 2.5,
        epipolar_weight: float = 1.0,
        min_iterations: int = 100,
        max_iterations: int = 1000,
    ):
        """Initialize MADPose estimator.

        Args:
            K: Camera intrinsic matrix (3, 3)
            grid_stride: Stride for grid sampling from flow
            reproj_pix_thres: Reprojection error threshold (pixels)
            epipolar_pix_thres: Epipolar error threshold (pixels)
            epipolar_weight: Weight for epipolar constraint
            min_iterations: Minimum RANSAC iterations
            max_iterations: Maximum RANSAC iterations
        """
        import madpose
        from madpose.utils import get_depths

        self.K = K.astype(np.float64)
        self.grid_stride = grid_stride
        self._get_depths = get_depths

        # RANSAC options
        self.options = madpose.HybridLORansacOptions()
        self.options.min_num_iterations = min_iterations
        self.options.max_num_iterations = max_iterations
        self.options.final_least_squares = True
        self.options.threshold_multiplier = 5.0
        self.options.num_lo_steps = 4
        self.options.squared_inlier_thresholds = [
            reproj_pix_thres**2,
            epipolar_pix_thres**2
        ]
        self.options.data_type_weights = [1.0, epipolar_weight]
        self.options.random_seed = 0

        # Estimator config
        self.est_config = madpose.EstimatorConfig()
        self.est_config.min_depth_constraint = True
        self.est_config.use_shift = True
        self.est_config.ceres_num_threads = 8

        self._madpose = madpose

    def _flow_to_keypoints(
        self,
        flow: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert dense flow to matched keypoints.

        Args:
            flow: Optical flow (H, W, 2)

        Returns:
            mkpts0: Keypoints in frame 0 (N, 2)
            mkpts1: Corresponding keypoints in frame 1 (N, 2)
        """
        H, W = flow.shape[:2]
        pts0 = _make_grid_points(W, H, self.grid_stride)

        # Sample flow at grid points
        xi = np.clip(np.rint(pts0[:, 0]).astype(np.int32), 0, W - 1)
        yi = np.clip(np.rint(pts0[:, 1]).astype(np.int32), 0, H - 1)
        flow_s = flow[yi, xi]

        pts1 = pts0 + flow_s

        # Validity checks
        valid = (
            np.isfinite(flow_s).all(axis=1) &
            (pts1[:, 0] >= 0) & (pts1[:, 0] < W) &
            (pts1[:, 1] >= 0) & (pts1[:, 1] < H)
        )

        return pts0[valid].astype(np.float32), pts1[valid].astype(np.float32)

    def estimate(
        self,
        img0: np.ndarray,
        img1: np.ndarray,
        depth0: np.ndarray,
        depth1: np.ndarray,
        flow: np.ndarray,
        baseline: Optional[float] = None,
    ) -> PoseResult:
        """Estimate relative pose using MADPose.

        Note: MADPose expects METRIC depth (meters), not inverse depth.
        Use UniDepth output directly. Baseline is ignored (MADPose
        estimates scale internally).

        Args:
            img0, img1: BGR images (H, W, 3)
            depth0, depth1: Metric depth maps (H, W) in meters
            flow: Optical flow from img0 to img1 (H, W, 2)
            baseline: Ignored (MADPose estimates scale internally)

        Returns:
            PoseResult with R, t, success status
        """
        # Convert flow to matched keypoints
        mkpts0, mkpts1 = self._flow_to_keypoints(flow)

        if len(mkpts0) < 10:
            return PoseResult(
                R=np.eye(3),
                t=np.zeros(3),
                success=False,
                num_inliers=len(mkpts0)
            )

        # Handle invalid depth values
        d0 = depth0.copy()
        d1 = depth1.copy()
        d0[~np.isfinite(d0) | (d0 <= 0)] = 1000.0
        d1[~np.isfinite(d1) | (d1 <= 0)] = 1000.0

        # Sample depth at keypoint locations
        depths0 = self._get_depths(img0, d0, mkpts0)
        depths1 = self._get_depths(img1, d1, mkpts1)

        try:
            pose, stats = self._madpose.HybridEstimatePoseScaleOffset(
                mkpts0, mkpts1,
                depths0, depths1,
                [float(depths0.min()), float(depths1.min())],
                self.K, self.K,
                self.options, self.est_config
            )

            R = pose.R()
            t = pose.t()

            return PoseResult(
                R=R,
                t=t,
                success=True,
                num_inliers=len(mkpts0),
                extra={
                    'scale': getattr(pose, 'scale', None),
                    'offset': getattr(pose, 'offset', None),
                }
            )
        except Exception as e:
            import traceback
            return PoseResult(
                R=np.eye(3),
                t=np.zeros(3),
                success=False,
                num_inliers=0,
                extra={'error': str(e), 'traceback': traceback.format_exc()}
            )

    @property
    def name(self) -> str:
        return "MADPose"

    @property
    def needs_baseline(self) -> bool:
        return False
