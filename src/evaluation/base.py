"""Base classes for evaluation framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class PoseResult:
    """Result from pose estimation."""
    R: np.ndarray  # (3, 3) rotation matrix
    t: np.ndarray  # (3,) unit translation vector
    success: bool = True
    num_inliers: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameMetrics:
    """Metrics for a single frame."""
    frame_idx: int
    baseline: float
    success: bool
    num_inliers: int = 0

    # Pose errors
    rot_error: Optional[float] = None
    trans_error: Optional[float] = None

    # Depth metrics (triangulation)
    tri_MAE: Optional[float] = None
    tri_RMSE: Optional[float] = None
    tri_AbsRel: Optional[float] = None
    tri_d105: Optional[float] = None
    tri_d115: Optional[float] = None
    tri_d125: Optional[float] = None

    # Depth metrics (refined)
    ref_MAE: Optional[float] = None
    ref_RMSE: Optional[float] = None
    ref_AbsRel: Optional[float] = None
    ref_d105: Optional[float] = None
    ref_d115: Optional[float] = None
    ref_d125: Optional[float] = None

    # Depth metrics (relative/monocular)
    rel_MAE: Optional[float] = None
    rel_RMSE: Optional[float] = None
    rel_AbsRel: Optional[float] = None
    rel_d105: Optional[float] = None
    rel_d115: Optional[float] = None
    rel_d125: Optional[float] = None

    # Depth metrics (3D warped prior = z_warp_pose)
    wp_MAE: Optional[float] = None
    wp_RMSE: Optional[float] = None
    wp_AbsRel: Optional[float] = None
    wp_d105: Optional[float] = None
    wp_d115: Optional[float] = None
    wp_d125: Optional[float] = None

    # Depth metrics (flow warped prior = z_warp_flow)
    wf_MAE: Optional[float] = None
    wf_RMSE: Optional[float] = None
    wf_AbsRel: Optional[float] = None
    wf_d105: Optional[float] = None
    wf_d115: Optional[float] = None
    wf_d125: Optional[float] = None

    # Per-iteration metrics (iter0 = forward, iter1 = backward/refined)
    # Triangulation depth for each iteration
    iter0_tri_AbsRel: Optional[float] = None
    iter0_tri_d125: Optional[float] = None
    iter1_tri_AbsRel: Optional[float] = None
    iter1_tri_d125: Optional[float] = None

    # Refined depth for each iteration
    iter0_ref_AbsRel: Optional[float] = None
    iter0_ref_d125: Optional[float] = None
    iter1_ref_AbsRel: Optional[float] = None
    iter1_ref_d125: Optional[float] = None

    # Sparse fusion depth for each iteration (before solve_metric_from_rel)
    iter0_fused_sparse_AbsRel: Optional[float] = None
    iter0_fused_sparse_d125: Optional[float] = None
    iter1_fused_sparse_AbsRel: Optional[float] = None
    iter1_fused_sparse_d125: Optional[float] = None

    # GT pose warp error: warp prev GT depth to current frame using GT pose,
    # compare with current GT depth. High values indicate unreliable GT pose.
    gt_warp_absrel: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class EvalSummary:
    """Summary of evaluation results."""
    method_name: str
    dataset_name: str
    total_frames: int
    success_frames: int
    elapsed_time: float
    fps: float

    # Pose summary
    rot_error_mean: Optional[float] = None
    rot_error_median: Optional[float] = None
    trans_error_mean: Optional[float] = None
    trans_error_median: Optional[float] = None

    # Trajectory summary (ATE)
    ATE_RMSE: Optional[float] = None
    ATE_mean: Optional[float] = None
    final_drift: Optional[float] = None
    trajectory_length: Optional[float] = None

    # Trajectory summary (RPE)
    RPE_trans_RMSE: Optional[float] = None
    RPE_trans_mean: Optional[float] = None
    RPE_rot_RMSE: Optional[float] = None
    RPE_rot_mean: Optional[float] = None

    # KITTI VO metrics
    t_err: Optional[float] = None  # Translation error (%)
    r_err: Optional[float] = None  # Rotation error (deg/100m)

    # Pose AUC (MADPose style)
    AUC_5: Optional[float] = None   # % of frames with max(rot_err, trans_err) < 5°
    AUC_10: Optional[float] = None  # % of frames with max(rot_err, trans_err) < 10°
    AUC_20: Optional[float] = None  # % of frames with max(rot_err, trans_err) < 20°

    # Depth summary (triangulation)
    tri_d125_mean: Optional[float] = None
    tri_MAE_mean: Optional[float] = None

    # Depth summary (refined)
    ref_d125_mean: Optional[float] = None
    ref_MAE_mean: Optional[float] = None

    # Temporal consistency (TAE from Video Depth Anything)
    TAE: Optional[float] = None
    TAE_forward: Optional[float] = None
    TAE_backward: Optional[float] = None

    extra: Dict[str, Any] = field(default_factory=dict)


class BaseMethodRunner(ABC):
    """Abstract base class for pose estimation methods."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Method name for display."""
        pass

    @property
    def requires_gt_baseline(self) -> bool:
        """Whether this method requires GT baseline for metric scale."""
        return False

    @property
    def is_metric(self) -> bool:
        """Whether this method outputs metric scale depths/translations."""
        return False

    @abstractmethod
    def initialize(self, H: int, W: int, fx: float, fy: float, cx: float, cy: float, **kwargs):
        """Initialize the method with camera parameters.

        Args:
            H, W: Image dimensions
            fx, fy, cx, cy: Camera intrinsics
            **kwargs: Additional method-specific parameters
        """
        pass

    @abstractmethod
    def process_frame(
        self,
        img_curr: np.ndarray,
        img_prev: Optional[np.ndarray] = None,
        depth_curr: Optional[np.ndarray] = None,
        depth_prev: Optional[np.ndarray] = None,
        baseline: float = 1.0,
        **kwargs
    ) -> PoseResult:
        """Process a frame and return pose estimate.

        Args:
            img_curr: Current frame (H, W, 3) BGR
            img_prev: Previous frame (H, W, 3) BGR (optional for some methods)
            depth_curr: Depth/inverse-depth for current frame (optional)
            depth_prev: Depth/inverse-depth for previous frame (optional)
            baseline: Translation magnitude (GT or estimated)
            **kwargs: Additional method-specific parameters

        Returns:
            PoseResult with R, t, and metadata
        """
        pass

    def reset(self):
        """Reset internal state (for SLAM-like methods)."""
        pass

    def finalize(self) -> Optional[Dict[str, Any]]:
        """Finalize processing and return any accumulated results."""
        return None


class BaseDataset(ABC):
    """Abstract base class for datasets."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def get(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get frame data.

        Returns:
            Dict with keys: image_og, depth_og (optional), etc.
        """
        pass

    @abstractmethod
    def get_image_size(self) -> Tuple[int, int]:
        """Return (H, W)."""
        pass

    @abstractmethod
    def get_intrinsics(self) -> Tuple[float, float, float, float]:
        """Return (fx, fy, cx, cy)."""
        pass

    @abstractmethod
    def get_baseline(self, idx: int) -> float:
        """Get translation magnitude between frame idx-1 and idx."""
        pass

    @abstractmethod
    def get_relative_pose(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get GT relative pose (R, t) from frame idx-1 to idx."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset identifier."""
        pass
