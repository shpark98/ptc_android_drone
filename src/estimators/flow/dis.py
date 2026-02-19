"""DIS (Dense Inverse Search) optical flow estimator."""

import cv2
import numpy as np
from .base import FlowEstimator


class DISFlowEstimator(FlowEstimator):
    """DIS optical flow using OpenCV.

    Uses PRESET_MEDIUM as specified in project constraints.
    """

    def __init__(self, preset: str = 'medium'):
        """Initialize DIS flow estimator.

        Args:
            preset: Quality preset ('ultrafast', 'fast', 'medium')
                    IMPORTANT: 'medium' is required for accuracy.
        """
        presets = {
            'ultrafast': cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
            'fast': cv2.DISOPTICAL_FLOW_PRESET_FAST,
            'medium': cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
        }

        if preset not in presets:
            raise ValueError(f"Invalid preset: {preset}. Choose from {list(presets.keys())}")

        self._preset_name = preset
        self._dis = cv2.DISOpticalFlow_create(presets[preset])

    def compute(self, img0: np.ndarray, img1: np.ndarray) -> np.ndarray:
        """Compute optical flow from img0 to img1.

        Args:
            img0: First image (H, W, 3) BGR uint8
            img1: Second image (H, W, 3) BGR uint8

        Returns:
            flow: Optical flow (H, W, 2)
        """
        # Convert to grayscale
        gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # Compute flow
        flow = self._dis.calc(gray0, gray1, None)
        return flow

    @property
    def name(self) -> str:
        return f"DIS-{self._preset_name}"
