"""UniDepth metric depth estimator."""

import sys
import numpy as np
from pathlib import Path
from .base import DepthEstimator

# Add project root and UniDepth to path for external imports
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_UNIDEPTH_PATH = _PROJECT_ROOT / "external" / "UniDepth"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_UNIDEPTH_PATH) not in sys.path:
    sys.path.insert(0, str(_UNIDEPTH_PATH))


class UniDepthEstimator(DepthEstimator):
    """UniDepth V1 metric depth estimation.

    Outputs metric depth in meters directly - no scale factor needed.
    Required for MADPose which expects metric depth.
    """

    def __init__(
        self,
        model_path: str = None,
        K: np.ndarray = None,
    ):
        """Initialize UniDepth estimator.

        Args:
            model_path: Path to UniDepth model. If None, uses default.
            K: Camera intrinsic matrix (3, 3). Required for UniDepth.
        """
        import torch
        from external.UniDepth.unidepth.models import UniDepthV1
        from external.UniDepth.unidepth.utils.camera import Pinhole

        if K is None:
            raise ValueError("UniDepth requires camera intrinsic matrix K")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path is None:
            repo_root = Path(__file__).parent.parent.parent.parent
            model_path = repo_root / "weights" / "uni_depth"

        # Setup camera intrinsics
        intrinsics_torch = torch.from_numpy(K.astype(np.float32))
        camera = Pinhole(K=intrinsics_torch.unsqueeze(0))

        # Load model from local directory
        self.model = UniDepthV1.from_pretrained(str(model_path), local_files_only=True)
        self.model.to(self.device).eval()

        # UniDepthV1 expects K tensor directly
        self.camera = camera.K.squeeze(0)
        self._torch = torch

    def infer(self, image: np.ndarray) -> np.ndarray:
        """Estimate metric depth from image.

        Args:
            image: BGR image (H, W, 3) uint8

        Returns:
            Metric depth in meters, shape (H, W)
        """
        with self._torch.no_grad():
            # UniDepth expects RGB
            rgb = image[..., ::-1].copy()  # BGR to RGB
            rgb_torch = self._torch.from_numpy(rgb).permute(2, 0, 1)

            pred = self.model.infer(rgb_torch, self.camera)
            depth = pred["depth"].squeeze().cpu().numpy()

        return depth

    @property
    def is_metric(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "UniDepth"
