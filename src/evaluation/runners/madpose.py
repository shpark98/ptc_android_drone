"""MADPose method runner."""

import numpy as np
from typing import Optional, Dict, Any

from ..base import BaseMethodRunner, PoseResult


class MADPoseRunner(BaseMethodRunner):
    """Runner for MADPose (Metric-Aware Depth Pose).

    MADPose uses metric depth from UniDepth and hybrid RANSAC
    for pose estimation. Outputs metric scale.
    """

    def __init__(
        self,
        device: str = "cuda",
        stride: int = 16,
        min_iterations: int = 100,
        max_iterations: int = 1000,
    ):
        """Initialize MADPose runner.

        Args:
            device: Device for computation
            stride: Grid stride for keypoint sampling
            min_iterations: RANSAC minimum iterations
            max_iterations: RANSAC maximum iterations
        """
        super().__init__(device)
        self.stride = stride
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations

        self._madpose = None
        self._options = None
        self._est_config = None
        self._K = None
        self._H = None
        self._W = None

        # Flow estimator
        self._flow_estimator = None

    @property
    def name(self) -> str:
        return "MADPose"

    @property
    def requires_gt_baseline(self) -> bool:
        return False  # Outputs metric scale

    @property
    def is_metric(self) -> bool:
        return True

    def initialize(
        self,
        H: int,
        W: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        **kwargs
    ):
        """Initialize MADPose."""
        try:
            import madpose
            from madpose.utils import get_depths
            self._madpose = madpose
            self._get_depths = get_depths
        except ImportError as e:
            raise ImportError(f"MADPose not found: {e}")

        # Import C++ triangulation
        try:
            import pr_depth_cpp as cpp
            self._cpp = cpp
        except ImportError as e:
            raise ImportError(f"PR-Depth C++ module not found: {e}")

        # Initialize flow estimator
        from src.estimators.flow import DISFlowEstimator
        self._flow_estimator = DISFlowEstimator(preset='medium')

        # Camera parameters
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy

        # Camera matrix
        self._K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        self._H = H
        self._W = W

        # MADPose options
        self._options = madpose.HybridLORansacOptions()
        self._options.min_num_iterations = self.min_iterations
        self._options.max_num_iterations = self.max_iterations
        self._options.final_least_squares = True
        self._options.threshold_multiplier = 5.0
        self._options.num_lo_steps = 4
        self._options.squared_inlier_thresholds = [25.0, 6.25]
        self._options.data_type_weights = [1.0, 1.0]
        self._options.random_seed = 0

        self._est_config = madpose.EstimatorConfig()
        self._est_config.min_depth_constraint = True
        self._est_config.use_shift = True

        self._initialized = True

    def process_frame(
        self,
        img_curr: np.ndarray,
        img_prev: Optional[np.ndarray] = None,
        depth_curr: Optional[np.ndarray] = None,
        depth_prev: Optional[np.ndarray] = None,
        baseline: float = 1.0,
        **kwargs
    ) -> PoseResult:
        """Process frame with MADPose.

        Args:
            img_curr: Current BGR image (H, W, 3)
            img_prev: Previous BGR image
            depth_curr: Current metric depth
            depth_prev: Previous metric depth
            baseline: Not used (metric scale from depth)

        Returns:
            PoseResult with estimated pose
        """
        if not self._initialized:
            raise RuntimeError("Runner not initialized. Call initialize() first.")

        if img_prev is None or depth_curr is None or depth_prev is None:
            return PoseResult(
                R=np.eye(3),
                t=np.zeros(3),
                success=False,
            )

        try:
            # Compute optical flow
            flow = self._flow_estimator.compute(img_prev, img_curr)

            # Sample keypoints on grid
            H, W = flow.shape[:2]
            xs = np.arange(0, W, self.stride, dtype=np.float32)
            ys = np.arange(0, H, self.stride, dtype=np.float32)
            gx, gy = np.meshgrid(xs, ys)
            pts0 = np.stack([gx.ravel(), gy.ravel()], axis=1)

            # Get flow at grid points
            xi = np.clip(np.rint(pts0[:, 0]).astype(np.int32), 0, W - 1)
            yi = np.clip(np.rint(pts0[:, 1]).astype(np.int32), 0, H - 1)
            flow_s = flow[yi, xi]
            pts1 = pts0 + flow_s

            # Filter valid points
            valid = (
                np.isfinite(flow_s).all(axis=1) &
                (pts1[:, 0] >= 0) & (pts1[:, 0] < W) &
                (pts1[:, 1] >= 0) & (pts1[:, 1] < H)
            )

            mkpts0 = pts0[valid].astype(np.float32)
            mkpts1 = pts1[valid].astype(np.float32)

            if len(mkpts0) < 10:
                return PoseResult(R=np.eye(3), t=np.zeros(3), success=False)

            # Get depths at keypoints
            depths0 = self._get_depths(img_prev, depth_prev.astype(np.float32), mkpts0)
            depths1 = self._get_depths(img_curr, depth_curr.astype(np.float32), mkpts1)

            # Run MADPose
            pose, stats = self._madpose.HybridEstimatePoseScaleOffset(
                mkpts0, mkpts1,
                depths0, depths1,
                [float(depths0.min()), float(depths1.min())],
                self._K, self._K,
                self._options, self._est_config
            )

            R_est = pose.R()
            t_est = pose.t()

            # pose.scale is a property, not a method
            scale = pose.scale if hasattr(pose, 'scale') else 1.0
            if callable(scale):
                scale = scale()

            # Triangulate depth using C++ code (same as PR-Depth)
            # Create dense flow grid (u0, v0, u1, v1)
            ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            u0 = xs.astype(np.float32)
            v0 = ys.astype(np.float32)
            u1 = (u0 + flow[:, :, 0]).astype(np.float32)
            v1 = (v0 + flow[:, :, 1]).astype(np.float32)

            # MADPose returns forward motion: p_curr = R @ p_prev + t
            # For triangulation, need to invert (same as PR-Depth C++ code):
            #   R_for_tri = R^T
            #   t_for_tri = -R^T * t
            # MADPose t is already metric scale (not unit vector)
            R_for_tri = R_est.T
            t_for_tri = -R_for_tri @ t_est

            # Call C++ triangulation with inverted pose
            tri_result = self._cpp.triangulate_depth(
                u0, v0, u1, v1,
                R_for_tri.astype(np.float64),
                t_for_tri.astype(np.float64),
                self._fx, self._fy, self._cx, self._cy,
                H, W
            )
            z_triangulation = tri_result['z1_tri']
            num_valid_tri = tri_result.get('num_valid', 0)

            # UniDepth metric depth (from previous frame)
            z_unidepth = depth_prev.copy().astype(np.float32)

            return PoseResult(
                R=R_est,
                t=t_est,
                success=True,
                num_inliers=num_valid_tri,  # Triangulation valid pixel count
                extra={
                    'scale': float(scale),
                    'z_tri': z_triangulation,
                    'z_unidepth': z_unidepth,
                }
            )

        except Exception as e:
            return PoseResult(
                R=np.eye(3),
                t=np.zeros(3),
                success=False,
                extra={'error': str(e)}
            )

    def reset(self):
        """Reset state (MADPose is stateless per-frame)."""
        pass
