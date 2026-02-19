"""MonST3R-based pose estimator wrapper."""

import sys
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Tuple

from .base import PoseEstimator, PoseResult


class MonST3REstimator(PoseEstimator):
    """Pose estimator using MonST3R (Motion-aware Stereo 3D Reconstruction).

    MonST3R extends DUSt3R with motion-aware features for dynamic scenes,
    providing camera poses and dense depth/point cloud estimation.
    """

    def __init__(
        self,
        model_name: str = "Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt",
        weights_path: Optional[str] = None,
        device: str = "cuda",
        image_size: int = 512,
    ):
        """Initialize MonST3R estimator.

        Args:
            model_name: HuggingFace model name or local path
            weights_path: Path to model weights (optional, will download if not provided)
            device: Device to run on ('cuda' or 'cpu')
            image_size: Input image size (512 or 224)
        """
        self.model_name = model_name
        self.weights_path = weights_path
        self.device = device
        self.image_size = image_size

        # Add monst3r to path
        monst3r_root = Path(__file__).parent.parent.parent.parent / "external" / "monst3r"
        if str(monst3r_root) not in sys.path:
            sys.path.insert(0, str(monst3r_root))

        self._model = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization of MonST3R model."""
        if self._initialized:
            return

        from dust3r.model import AsymmetricCroCo3DStereo

        # Load model
        if self.weights_path and Path(self.weights_path).exists():
            self._model = AsymmetricCroCo3DStereo.from_pretrained(self.weights_path)
        else:
            self._model = AsymmetricCroCo3DStereo.from_pretrained(self.model_name)

        self._model = self._model.to(self.device)
        self._model.eval()
        self._initialized = True

    def _prepare_images(self, imgs: List[np.ndarray]) -> List[dict]:
        """Prepare images for MonST3R inference.

        Args:
            imgs: List of images (H, W, 3) BGR uint8

        Returns:
            List of image dicts for dust3r
        """
        from dust3r.utils.image import load_images
        import tempfile
        import cv2

        # Save images temporarily and load via dust3r utils
        prepared = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, img in enumerate(imgs):
                # Convert BGR to RGB and save
                rgb = img[:, :, ::-1]
                path = f"{tmpdir}/img_{i}.png"
                cv2.imwrite(path, img)

            # Load using dust3r utility
            paths = [f"{tmpdir}/img_{i}.png" for i in range(len(imgs))]
            prepared = load_images(paths, size=self.image_size)

        return prepared

    def estimate(
        self,
        img0: np.ndarray,
        img1: np.ndarray,
        depth0: np.ndarray,
        depth1: np.ndarray,
        flow: np.ndarray,
        K: Optional[np.ndarray] = None,
        baseline: Optional[float] = None,
    ) -> PoseResult:
        """Estimate relative pose using MonST3R.

        Args:
            img0: First image (H, W, 3) BGR uint8
            img1: Second image (H, W, 3) BGR uint8
            depth0: Depth map for first image (H, W) - not used
            depth1: Depth map for second image (H, W) - not used
            flow: Optical flow (H, W, 2) - not used
            K: Camera intrinsics (3, 3)
            baseline: Ground truth baseline (optional)

        Returns:
            PoseResult with R, t, and estimated depth/pointcloud
        """
        self._lazy_init()

        try:
            from dust3r.inference import inference
            from dust3r.image_pairs import make_pairs
            from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
            from dust3r.utils.device import to_numpy

            # Prepare images
            images = self._prepare_images([img0, img1])

            # Create pairs
            pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)

            # Run inference
            with torch.no_grad():
                output = inference(pairs, self._model, self.device, batch_size=1)

            # Global alignment
            scene = global_aligner(
                output,
                device=self.device,
                mode=GlobalAlignerMode.PairViewer,
            )

            # Get poses (cam2world)
            poses = scene.get_im_poses().cpu().numpy()

            if len(poses) < 2:
                return PoseResult(
                    R=np.eye(3),
                    t=np.zeros(3),
                    success=False,
                    num_inliers=0,
                    extra={"method": "monst3r", "error": "insufficient_poses"}
                )

            # Compute relative pose
            # poses[0] = T_w_0 (frame 0 to world)
            # poses[1] = T_w_1 (frame 1 to world)
            T_w_0 = poses[0]  # (4, 4)
            T_w_1 = poses[1]  # (4, 4)

            # Standard convention: T_rel transforms points from frame 0 to frame 1
            # T_rel = inv(T_w_1) @ T_w_0
            T_rel = np.linalg.inv(T_w_1) @ T_w_0

            R = T_rel[:3, :3]
            t = T_rel[:3, 3]

            # Get depth and point cloud
            pts3d = to_numpy(scene.get_pts3d())
            depths = to_numpy(scene.get_depthmaps())

            return PoseResult(
                R=R,
                t=t,
                success=True,
                num_inliers=-1,
                extra={
                    "method": "monst3r",
                    "pts3d": pts3d,
                    "depths": depths,
                    "focals": scene.get_focals().cpu().numpy() if hasattr(scene, 'get_focals') else None,
                }
            )

        except Exception as e:
            import traceback
            return PoseResult(
                R=np.eye(3),
                t=np.zeros(3),
                success=False,
                num_inliers=0,
                extra={"method": "monst3r", "error": str(e), "traceback": traceback.format_exc()}
            )

    @property
    def name(self) -> str:
        return "MonST3R"

    @property
    def needs_baseline(self) -> bool:
        return False  # MonST3R estimates scale
