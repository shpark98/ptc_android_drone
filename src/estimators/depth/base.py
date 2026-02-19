"""Base class for depth estimators."""

from abc import ABC, abstractmethod
import numpy as np


class DepthEstimator(ABC):
    """Abstract base class for monocular depth estimation.

    All depth estimators should inherit from this class and implement
    the infer() method.
    """

    @abstractmethod
    def infer(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth from a single image.

        Args:
            image: BGR image (H, W, 3) uint8

        Returns:
            depth: Depth map (H, W) float32, units depend on model
                   - Metric models (UniDepth): meters
                   - Relative models (DepthAnything): normalized inverse depth
        """
        pass

    @property
    @abstractmethod
    def is_metric(self) -> bool:
        """Whether this estimator outputs metric depth."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the estimator."""
        pass

    def to_metric_depth(self, output: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Convert model output to metric depth.

        For metric models, returns output directly.
        For relative models, applies scale factor.

        Args:
            output: Model output from infer()
            scale: Scale factor for relative depth models

        Returns:
            Metric depth in meters
        """
        if self.is_metric:
            return output
        else:
            # For inverse depth models like DepthAnything
            depth = 1.0 / (output + 1e-8)
            return depth * scale
