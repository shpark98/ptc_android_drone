"""Base class for pose estimators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class PoseResult:
    """Result from pose estimation."""
    R: np.ndarray              # (3, 3) rotation matrix
    t: np.ndarray              # (3,) translation vector (unit or scaled)
    success: bool              # Whether estimation succeeded
    num_inliers: int = 0       # Number of inliers
    extra: dict = None         # Additional data (triangulated depth, etc.)

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


class PoseEstimator(ABC):
    """Abstract base class for relative pose estimation.

    All pose estimators should inherit from this class and implement
    the estimate() method.
    """

    @abstractmethod
    def estimate(
        self,
        img0: np.ndarray,
        img1: np.ndarray,
        depth0: np.ndarray,
        depth1: np.ndarray,
        flow: np.ndarray,
        baseline: Optional[float] = None,
    ) -> PoseResult:
        """Estimate relative pose between two frames.

        Args:
            img0: First image (H, W, 3) BGR uint8
            img1: Second image (H, W, 3) BGR uint8
            depth0: Depth map for first image (H, W)
            depth1: Depth map for second image (H, W)
            flow: Optical flow from img0 to img1 (H, W, 2)
            baseline: Ground truth baseline (optional, for scaling)

        Returns:
            PoseResult with R, t, and additional info
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the estimator."""
        pass

    @property
    @abstractmethod
    def needs_baseline(self) -> bool:
        """Whether this estimator requires baseline for scaling."""
        pass
