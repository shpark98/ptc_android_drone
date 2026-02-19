"""Video Depth Anything runner for depth-only evaluation."""

import numpy as np
from typing import Optional, Dict, Any

from ..base import BaseMethodRunner, PoseResult


class VideoDepthAnythingRunner(BaseMethodRunner):
    """Runner for Video Depth Anything depth estimation.

    This runner wraps Video Depth Anything for depth estimation evaluation.
    It does NOT perform pose estimation - it's meant to be used for:
    1. Comparing depth quality between streaming vs offline modes
    2. Comparing metric vs relative depth models
    3. Benchmarking against other depth estimators

    For pose evaluation with Video Depth Anything depth, use PRDepthRunner
    with a VideoDepthAnythingEstimator as the depth source.
    """

    def __init__(
        self,
        device: str = "cuda",
        encoder: str = "vitl",
        metric: bool = True,
        streaming: bool = True,
        input_size: int = 518,
        max_res: int = 1280,
        fp32: bool = False,
    ):
        """Initialize Video Depth Anything runner.

        Args:
            device: Device for computation ('cuda' or 'cpu')
            encoder: Encoder type ('vits', 'vitb', 'vitl')
            metric: Use metric depth model (meters) vs relative depth
            streaming: Use streaming mode (real-time) vs offline mode (batch)
            input_size: Input size for model (default 518)
            max_res: Maximum resolution (default 1280)
            fp32: Use float32 precision (default float16)
        """
        super().__init__(device)
        self.encoder = encoder
        self.metric = metric
        self.streaming = streaming
        self.input_size = input_size
        self.max_res = max_res
        self.fp32 = fp32

        self._depth_estimator = None
        self._prev_depth = None

    @property
    def name(self) -> str:
        mode = "stream" if self.streaming else "offline"
        depth_type = "metric" if self.metric else "rel"
        return f"VDA-{self.encoder}-{depth_type}-{mode}"

    @property
    def requires_gt_baseline(self) -> bool:
        # Metric mode doesn't need GT baseline for depth
        # Relative mode would need it for scale
        return not self.metric

    @property
    def is_metric(self) -> bool:
        return self.metric

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
        """Initialize Video Depth Anything estimator."""
        from src.estimators.depth import VideoDepthAnythingEstimator

        self._depth_estimator = VideoDepthAnythingEstimator(
            encoder=self.encoder,
            metric=self.metric,
            streaming=self.streaming,
            input_size=self.input_size,
            max_res=self.max_res,
            fp32=self.fp32,
            device=self.device,
        )

        self._H = H
        self._W = W
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._initialized = True
        self._prev_depth = None

    def process_frame(
        self,
        img_curr: np.ndarray,
        img_prev: Optional[np.ndarray] = None,
        depth_curr: Optional[np.ndarray] = None,
        depth_prev: Optional[np.ndarray] = None,
        baseline: float = 1.0,
        **kwargs
    ) -> PoseResult:
        """Process frame and estimate depth.

        This runner focuses on depth estimation. It returns identity pose
        since it doesn't do pose estimation. The depth is stored in extra.

        Args:
            img_curr: Current BGR image (H, W, 3)
            img_prev: Previous BGR image (unused)
            depth_curr: External depth (unused - we compute our own)
            depth_prev: Previous depth (unused)
            baseline: GT translation magnitude (for relative depth scaling)

        Returns:
            PoseResult with identity pose and depth in extra
        """
        if not self._initialized:
            raise RuntimeError("Runner not initialized. Call initialize() first.")

        # Estimate depth
        depth = self._depth_estimator.infer(img_curr)

        # For relative depth, optionally apply scale
        if not self.metric and baseline > 0:
            # Convert to metric using baseline as reference
            # This is a simple median-based scaling for comparison
            pass  # Keep raw depth for now

        # Identity pose (no pose estimation)
        R = np.eye(3, dtype=np.float32)
        t = np.zeros(3, dtype=np.float32)

        result = PoseResult(
            R=R,
            t=t,
            success=True,  # Depth estimation always succeeds
            num_inliers=0,
            extra={
                'depth': depth,
                'depth_type': 'metric' if self.metric else 'relative',
                'mode': 'streaming' if self.streaming else 'offline',
            }
        )

        self._prev_depth = depth
        return result

    def reset(self):
        """Reset estimator state for new sequence."""
        if self._depth_estimator is not None:
            self._depth_estimator.reset()
        self._prev_depth = None

    def finalize(self) -> Optional[Dict[str, Any]]:
        """Flush remaining frames in offline mode."""
        if self._depth_estimator is not None and not self.streaming:
            remaining_depths = self._depth_estimator.flush()
            return {'remaining_depths': remaining_depths}
        return None

    def get_depth_estimator(self):
        """Get the underlying depth estimator for external use.

        Useful when you want to use VDA depth with another pose estimator.
        """
        return self._depth_estimator


class VideoDepthAnythingPoseRunner(BaseMethodRunner):
    """Combined runner using Video Depth Anything for depth + PR-Depth for pose.

    This runner combines:
    - Video Depth Anything for depth estimation (streaming or offline)
    - PR-Depth C++ pipeline for pose estimation

    Allows comparing pose accuracy with different depth sources.
    """

    def __init__(
        self,
        device: str = "cuda",
        # VDA options
        encoder: str = "vitl",
        vda_metric: bool = True,
        streaming: bool = True,
        input_size: int = 518,
        max_res: int = 1280,
        fp32: bool = False,
        # PR-Depth options
        iterative: bool = True,
        use_depth_consistency: bool = True,
        depth_consistency_threshold: float = 0.65,
    ):
        """Initialize combined VDA + PR-Depth runner.

        Args:
            device: Device for computation
            encoder: VDA encoder type ('vits', 'vitb', 'vitl')
            vda_metric: Use VDA metric depth model
            streaming: Use VDA streaming mode
            input_size: VDA input size
            max_res: VDA maximum resolution
            fp32: Use float32 precision
            iterative: Enable PR-Depth iterative refinement
            use_depth_consistency: Enable depth consistency check
            depth_consistency_threshold: DC threshold
        """
        super().__init__(device)

        # VDA settings
        self.encoder = encoder
        self.vda_metric = vda_metric
        self.streaming = streaming
        self.input_size = input_size
        self.max_res = max_res
        self.fp32 = fp32

        # PR-Depth settings
        self.iterative = iterative
        self.use_depth_consistency = use_depth_consistency
        self.depth_consistency_threshold = depth_consistency_threshold

        self._depth_estimator = None
        self._pipeline = None
        self._config = None

    @property
    def name(self) -> str:
        mode = "stream" if self.streaming else "offline"
        depth_type = "metric" if self.vda_metric else "rel"
        return f"PRD-VDA-{self.encoder}-{depth_type}-{mode}"

    @property
    def requires_gt_baseline(self) -> bool:
        # If using VDA metric depth, we could potentially avoid GT baseline
        # But PR-Depth still benefits from knowing the scale
        return True

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
        """Initialize both VDA and PR-Depth components."""
        from src.estimators.depth import VideoDepthAnythingEstimator

        # Initialize VDA
        self._depth_estimator = VideoDepthAnythingEstimator(
            encoder=self.encoder,
            metric=self.vda_metric,
            streaming=self.streaming,
            input_size=self.input_size,
            max_res=self.max_res,
            fp32=self.fp32,
            device=self.device,
        )

        # Initialize PR-Depth C++
        try:
            import pr_depth_cpp as cpp
        except ImportError as e:
            raise ImportError(f"PR-Depth C++ module not found: {e}")

        self._config = cpp.DepthRefinementConfig()
        self._config.H = H
        self._config.W = W
        self._config.fx = fx
        self._config.fy = fy
        self._config.cx = cx
        self._config.cy = cy
        self._config.enable_iterative_refinement = self.iterative
        self._config.use_depth_consistency = self.use_depth_consistency
        self._config.depth_consistency_threshold = self.depth_consistency_threshold

        self._pipeline = cpp.DepthRefinement(self._config)
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
        """Process frame with VDA depth and PR-Depth pose.

        Args:
            img_curr: Current BGR image (H, W, 3)
            img_prev: Previous BGR image (unused)
            depth_curr: External depth (unused - we use VDA)
            depth_prev: Previous depth (unused)
            baseline: GT translation magnitude

        Returns:
            PoseResult with estimated pose
        """
        if not self._initialized:
            raise RuntimeError("Runner not initialized. Call initialize() first.")

        # Get depth from VDA
        vda_depth = self._depth_estimator.infer(img_curr)

        # Convert to inverse depth for PR-Depth
        # PR-Depth expects normalized inverse depth [0, 1] where 1=close, 0=far
        if self.vda_metric:
            # Metric depth (meters) -> inverse depth
            inv_depth = 1.0 / (vda_depth + 1e-8)
            # Normalize to [0, 1]
            inv_depth = (inv_depth - inv_depth.min()) / (inv_depth.max() - inv_depth.min() + 1e-8)
        else:
            # VDA relative depth is already in a useful form
            # Normalize to [0, 1]
            inv_depth = (vda_depth - vda_depth.min()) / (vda_depth.max() - vda_depth.min() + 1e-8)

        inv_depth = inv_depth.astype(np.float32)

        # Run PR-Depth pipeline
        result = self._pipeline.refine(
            img_curr, inv_depth, float(baseline),
            np.array([], dtype=np.int32)
        )

        num_matches = result['num_matches']
        min_baseline_for_pose = 0.05

        if baseline < min_baseline_for_pose:
            success = False
        elif num_matches >= 50:
            success = True
        else:
            success = False

        return PoseResult(
            R=result['R'],
            t=result['t'],
            success=success,
            num_inliers=num_matches,
            extra={
                'vda_depth': vda_depth,
                'inv_depth_used': inv_depth,
                'z_tri': result.get('z_tri'),
                'z_refined': result.get('z_refined'),
                'depth_consistency_score': result.get('depth_consistency_score', 1.0),
            }
        )

    def reset(self):
        """Reset both VDA and PR-Depth state."""
        if self._depth_estimator is not None:
            self._depth_estimator.reset()
        if self._pipeline is not None and self._config is not None:
            import pr_depth_cpp as cpp
            self._pipeline = cpp.DepthRefinement(self._config)
