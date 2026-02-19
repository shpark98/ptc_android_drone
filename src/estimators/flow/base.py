"""Base class for optical flow estimators."""

from abc import ABC, abstractmethod
import numpy as np


class FlowEstimator(ABC):
    """Abstract base class for optical flow estimation."""

    @abstractmethod
    def compute(self, img0: np.ndarray, img1: np.ndarray) -> np.ndarray:
        """Compute optical flow from img0 to img1.

        Args:
            img0: First image (H, W, 3) BGR uint8
            img1: Second image (H, W, 3) BGR uint8

        Returns:
            flow: Optical flow (H, W, 2), where flow[y, x] = (dx, dy)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the flow estimator."""
        pass
