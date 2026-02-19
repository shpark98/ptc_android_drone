"""PR-Depth method runner."""

import numpy as np
from typing import Optional, Dict, Any

from ..base import BaseMethodRunner, PoseResult


class PRDepthRunner(BaseMethodRunner):
    """Runner for PR-Depth pipeline.

    PR-Depth uses relative depth from DepthAnything and estimates pose
    via motion field decomposition with RANSAC.

    Requires GT baseline for metric scale since it uses relative depth.

    Optionally accepts external rotation (e.g., from IMU) to improve accuracy.
    """

    def __init__(
        self,
        device: str = "cuda",
        iterative: bool = True,
        iterative_iters: int = 1,
        use_external_rotation: bool = False,
        use_baseline_guard: bool = True,
        min_baseline: float = 0.3,
        baseline_ema_beta: float = 0.9,
        # Fusion ablation options
        use_segmentation: bool = True,
        use_rgb_guide: bool = True,
        metric_scale_mode: int = 2,
        # RANSAC scoring mode
        use_magsac_scoring: bool = True,
        # GT pose fallback mode
        use_gt_pose_fallback: bool = False,
        gt_pose_rotation_threshold_deg: float = 3.0,
        # Ablation options
        skip_temporal_fusion: bool = False,
        use_gt_R: bool = False,
        # Pixel-count thresholds (for different resolutions)
        min_scale_overlap: int = 2000,
        seg_min_size: int = 200,
        max_points: int = 2000,
    ):
        """Initialize PR-Depth runner.

        Args:
            device: Device for computation
            iterative: Enable iterative refinement
            iterative_iters: Number of refinement iterations
            use_external_rotation: Use external rotation (e.g., IMU) instead of motion field R
            use_baseline_guard: Enable baseline guard (skip triangulation when baseline too short)
            min_baseline: Minimum baseline for triangulation (default: 0.05m)
            baseline_ema_beta: EMA smoothing factor for baseline (default: 0.9)
            use_segmentation: Enable edge-aware segmentation for fusion
            use_rgb_guide: Enable RGB guiding for edge-aware fusion
            metric_scale_mode: Scale estimation mode (0=off, 1=global, 2=per-segment)
            use_magsac_scoring: Use MAGSAC++ soft scoring (True) or paper MAD-based binary (False)
            use_gt_pose_fallback: Use GT pose when rotation exceeds threshold
            gt_pose_rotation_threshold_deg: Rotation threshold for GT pose fallback (degrees)
            skip_temporal_fusion: Skip Bayesian update, only triangulation + solve_metric (ablation)
            use_gt_R: Always use GT rotation when provided (ablation)
            min_scale_overlap: Minimum overlapping pixels for scale matching
            seg_min_size: Minimum segment size for segmentation
            max_points: Maximum points for motion estimation
        """
        super().__init__(device)
        self.iterative = iterative
        self.iterative_iters = iterative_iters
        self.use_external_rotation = use_external_rotation
        self.use_baseline_guard = use_baseline_guard
        self.min_baseline = min_baseline
        self.baseline_ema_beta = baseline_ema_beta
        # Fusion options
        self.use_segmentation = use_segmentation
        self.use_rgb_guide = use_rgb_guide
        self.metric_scale_mode = metric_scale_mode
        # RANSAC scoring mode
        self.use_magsac_scoring = use_magsac_scoring
        # GT pose fallback mode
        self.use_gt_pose_fallback = use_gt_pose_fallback
        self.gt_pose_rotation_threshold_deg = gt_pose_rotation_threshold_deg
        # Ablation options
        self.skip_temporal_fusion = skip_temporal_fusion
        self.use_gt_R = use_gt_R
        # Pixel-count thresholds
        self.min_scale_overlap = min_scale_overlap
        self.seg_min_size = seg_min_size
        self.max_points = max_points

        self._pipeline = None
        self._config = None

    @property
    def name(self) -> str:
        if self.use_external_rotation:
            return "PR-Depth-IMU"
        return "PR-Depth"

    @property
    def requires_gt_baseline(self) -> bool:
        return True  # Uses relative depth, needs GT baseline

    @property
    def is_metric(self) -> bool:
        return True  # t is scaled by baseline in C++

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
        """Initialize PR-Depth pipeline."""
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
        self._config.iterative_refinement_iters = self.iterative_iters
        self._config.use_baseline_guard = self.use_baseline_guard
        self._config.min_baseline = self.min_baseline
        self._config.baseline_ema_beta = self.baseline_ema_beta
        # Fusion options
        self._config.use_segmentation = self.use_segmentation
        self._config.use_rgb_guide = self.use_rgb_guide
        self._config.metric_scale_mode = self.metric_scale_mode
        # RANSAC scoring mode
        self._config.use_magsac_scoring = self.use_magsac_scoring
        # GT pose fallback mode
        self._config.use_gt_pose_fallback = self.use_gt_pose_fallback
        self._config.gt_pose_rotation_threshold_deg = self.gt_pose_rotation_threshold_deg
        # Ablation options
        self._config.skip_temporal_fusion = self.skip_temporal_fusion
        self._config.use_gt_R = self.use_gt_R
        # Fusion parameters: use C++ FusionConfig struct defaults
        # (kcap_floor=0.35, lambda_forget=0.4, min_var=5e-3)
        # Pixel-count thresholds
        self._config.min_scale_overlap = self.min_scale_overlap
        self._config.seg_min_size = self.seg_min_size
        self._config.max_points = self.max_points

        self._pipeline = cpp.DepthRefinement(self._config)
        self._initialized = True
        self._first_frame = True

    def process_frame(
        self,
        img_curr: np.ndarray,
        img_prev: Optional[np.ndarray] = None,
        depth_curr: Optional[np.ndarray] = None,
        depth_prev: Optional[np.ndarray] = None,
        baseline: float = 1.0,
        external_R: Optional[np.ndarray] = None,
        external_flow: Optional[np.ndarray] = None,
        gt_R: Optional[np.ndarray] = None,
        gt_t: Optional[np.ndarray] = None,
        **kwargs
    ) -> PoseResult:
        """Process frame with PR-Depth.

        Args:
            img_curr: Current BGR image (H, W, 3)
            img_prev: Previous BGR image (unused - pipeline maintains state)
            depth_curr: Current inverse depth [0, 1]
            depth_prev: Previous inverse depth (unused - pipeline maintains state)
            baseline: GT translation magnitude
            external_R: Optional external rotation matrix (3x3), e.g., from IMU.
                       If provided and use_external_rotation=True, this R is used
                       instead of motion field estimated R.
            external_flow: Optional external optical flow (H, W, 2), e.g., GT flow.
                          If provided, use this instead of computing DIS flow.
            gt_R: Optional GT rotation matrix (3x3) for GT pose fallback mode.
            gt_t: Optional GT translation vector (3,) for GT pose fallback mode.

        Returns:
            PoseResult with estimated pose
        """
        if not self._initialized:
            raise RuntimeError("Runner not initialized. Call initialize() first.")

        if depth_curr is None:
            raise ValueError("PR-Depth requires depth_curr (inverse depth)")

        # Ensure correct dtype
        inv_depth = depth_curr.astype(np.float32)

        # Prepare GT pose arrays (empty if not provided)
        gt_R_arr = gt_R.astype(np.float64) if gt_R is not None else np.array([], dtype=np.float64)
        gt_t_arr = gt_t.astype(np.float64) if gt_t is not None else np.array([], dtype=np.float64)

        # Run pipeline
        result = self._pipeline.refine(
            img_curr, inv_depth, float(baseline),
            np.array([], dtype=np.int32),  # seg_labels
            gt_R_arr,
            gt_t_arr
        )

        num_matches = result['num_matches']
        tri_disabled = result.get('tri_disabled', False)

        success = num_matches >= 50

        return PoseResult(
            R=result['R'],
            t=result['t'],
            success=success,
            num_inliers=num_matches,
            extra={
                'z_tri': result.get('z_tri'),
                'z_refined': result.get('z_refined'),
                'z_warp_pose': result.get('z_warp_pose'),  # 3D warped prior
                'z_warp_flow': result.get('z_warp_flow'),  # Flow warped prior
                'prev_depth_used': result.get('prev_depth_used'),  # Previous frame depth
                'iteration_info': result.get('iteration_info'),  # Per-iteration depth maps
                'used_gt_pose': result.get('used_gt_pose', False),  # GT pose fallback used
                'rotation_angle_deg': result.get('rotation_angle_deg', 0.0),  # Estimated rotation angle
                'z_warp_gt': result.get('z_warp_gt'),  # Dense 3D warp from GT pose
            }
        )

    def reset(self):
        """Reset pipeline state."""
        if self._pipeline is not None and self._config is not None:
            import pr_depth_cpp as cpp
            self._pipeline = cpp.DepthRefinement(self._config)
        self._first_frame = True
