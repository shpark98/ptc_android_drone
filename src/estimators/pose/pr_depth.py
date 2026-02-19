"""PR-Depth pose estimator (C++ implementation)."""

from typing import Optional
import numpy as np
from .base import PoseEstimator, PoseResult


class PRDepthEstimator(PoseEstimator):
    """PR-Depth pose estimation using C++ pipeline.

    Uses motion field decomposition with RANSAC for robust pose estimation.
    Requires baseline for metric scale.
    """

    def __init__(
        self,
        H: int,
        W: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        **kwargs
    ):
        """Initialize PR-Depth estimator.

        Args:
            H, W: Image dimensions
            fx, fy, cx, cy: Camera intrinsics
            **kwargs: Additional config options for DepthRefinementConfig
        """
        import pr_depth_cpp as cpp

        self.H = H
        self.W = W

        # Create config
        config = cpp.DepthRefinementConfig()
        config.H = H
        config.W = W
        config.fx = fx
        config.fy = fy
        config.cx = cx
        config.cy = cy

        # Default settings
        config.ransac_max_iters = kwargs.get('ransac_max_iters', 500)
        config.ransac_min_sample = kwargs.get('ransac_min_sample', 6)
        config.ransac_thresh_ratio = kwargs.get('ransac_thresh_ratio', 1.5)
        config.min_flow_px = kwargs.get('min_flow_px', 0.01)
        config.max_points = kwargs.get('max_points', 2000)
        config.max_depth = kwargs.get('max_depth', 80.0)
        config.use_baseline_guard = kwargs.get('use_baseline_guard', False)
        config.min_baseline = kwargs.get('min_baseline', 0.05)
        config.min_scale_overlap = kwargs.get('min_scale_overlap', 2000)
        config.scale_tol_median = kwargs.get('scale_tol_median', 0.3)
        config.use_segmentation = kwargs.get('use_segmentation', True)
        config.seg_sigma = kwargs.get('seg_sigma', 0.5)
        config.seg_k = kwargs.get('seg_k', 500.0)
        config.seg_min_size = kwargs.get('seg_min_size', 200)
        config.seg_down = kwargs.get('seg_down', 0.5)
        config.enable_iterative_refinement = kwargs.get('iterative', False)
        config.iterative_refinement_iters = kwargs.get('iterative_iters', 1)

        self.pipeline = cpp.DepthRefinement(config)
        self._initialized = False

    def estimate(
        self,
        img0: np.ndarray,
        img1: np.ndarray,
        depth0: np.ndarray,
        depth1: np.ndarray,
        flow: np.ndarray,
        baseline: Optional[float] = None,
    ) -> PoseResult:
        """Estimate relative pose.

        Note: PR-Depth uses inv_depth internally, so depth should be
        inverse depth from DepthAnything (0=far, 1=close).
        """
        if baseline is None:
            baseline = 0.0

        # PR-Depth expects inverse depth
        inv_depth = depth1  # Assume already inverse depth from DepthAnything

        # First frame initialization
        if not self._initialized:
            self.pipeline.refine(img0, depth0.astype(np.float32), 0.0)
            self._initialized = True

        # Run pipeline
        result = self.pipeline.refine(img1, inv_depth.astype(np.float32), baseline)

        # Build extra dict with all available debug info
        extra = {
            'z_tri': result.get('z_tri'),
            'z_refined': result.get('z_refined'),
            'num_valid_tri': result.get('num_valid_tri', 0),
            # DC scores
            'dc_score_forward': result.get('dc_score_forward', -1),
            'dc_score_backward': result.get('dc_score_backward', -1),
            'depth_consistency_score': result.get('depth_consistency_score', -1),
            'depth_consistency_rejected': result.get('depth_consistency_rejected', False),
            # Backward estimation info
            'used_backward': result.get('used_backward', False),
            'metric_scale_forward': result.get('metric_scale_forward', -1),
            'metric_scale_backward': result.get('metric_scale_backward', -1),
            # Warp depths
            'z_warp_flow': result.get('z_warp_flow'),
            'z_warp_pose': result.get('z_warp_pose'),
            # Other
            'tri_disabled': result.get('tri_disabled', False),
            'baseline': result.get('baseline', baseline),
        }

        if result['num_matches'] < 50:
            return PoseResult(
                R=result['R'],
                t=result['t'],
                success=False,
                num_inliers=result['num_matches'],
                extra=extra
            )

        return PoseResult(
            R=result['R'],
            t=result['t'],
            success=True,
            num_inliers=result['num_matches'],
            extra=extra
        )

    def reset(self):
        """Reset pipeline state for new sequence."""
        self._initialized = False

    @property
    def name(self) -> str:
        return "PR-Depth"

    @property
    def needs_baseline(self) -> bool:
        return True
