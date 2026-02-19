"""MonST3R method runner."""

import sys
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, Any, List

from ..base import BaseMethodRunner, PoseResult


class MonST3RRunner(BaseMethodRunner):
    """Runner for MonST3R (Monocular Structure from Motion with DUSt3R).

    MonST3R processes frames in a global optimization manner.
    For fair comparison, we use pairwise processing mode.
    """

    def __init__(
        self,
        device: str = "cuda",
        monst3r_root: Optional[Path] = None,
        target_size: tuple = (512, 384),
    ):
        """Initialize MonST3R runner.

        Args:
            device: Device for computation
            monst3r_root: Path to MonST3R installation
            target_size: (W, H) for image resizing
        """
        super().__init__(device)

        if monst3r_root is None:
            monst3r_root = Path(__file__).parent.parent.parent.parent.parent / 'external' / 'monst3r'
        self.monst3r_root = Path(monst3r_root)
        self.target_size = target_size

        self._model = None
        self._ImgNorm = None
        self._ToTensor = None

    @property
    def name(self) -> str:
        return "MonST3R"

    @property
    def requires_gt_baseline(self) -> bool:
        return False  # Up-to-scale

    @property
    def is_metric(self) -> bool:
        return False  # Up-to-scale

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
        """Initialize MonST3R."""
        # Save original path
        original_path = sys.path.copy()

        # Remove conflicting paths
        conflicting_paths = ['batrack', 'Depth-Anything']
        sys.path = [p for p in sys.path if not any(c in p for c in conflicting_paths)]

        # Clear cached modules
        modules_to_remove = [k for k in sys.modules.keys()
                           if k.startswith('utils') or k.startswith('dust3r') or k.startswith('main')]
        for mod in modules_to_remove:
            del sys.modules[mod]

        # Add MonST3R path
        sys.path.insert(0, str(self.monst3r_root))

        try:
            import torch
            import torchvision.transforms as tvf
            import argparse

            # PyTorch 2.6+ safe globals
            torch.serialization.add_safe_globals([argparse.Namespace])

            from dust3r.model import AsymmetricCroCo3DStereo

            self._ImgNorm = tvf.Compose([
                tvf.ToTensor(),
                tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            self._ToTensor = tvf.ToTensor()

            # Load model
            model_path = self.monst3r_root / 'checkpoints' / 'MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth'
            if model_path.exists():
                self._model = AsymmetricCroCo3DStereo.from_pretrained(str(model_path)).to(self.device)
            else:
                self._model = AsymmetricCroCo3DStereo.from_pretrained(
                    'Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt'
                ).to(self.device)
            self._model.eval()

            self._initialized = True
            self._H = H
            self._W = W

        except ImportError as e:
            sys.path = original_path
            raise ImportError(f"MonST3R dependencies not found: {e}")

        # Restore path but keep MonST3R
        sys.path = [str(self.monst3r_root)] + original_path

    def _prepare_image(self, img_bgr: np.ndarray) -> Dict:
        """Prepare image for MonST3R."""
        import torch
        import PIL.Image

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(img_rgb)
        pil_img = pil_img.resize(self.target_size, PIL.Image.LANCZOS)
        W, H = pil_img.size

        img_tensor = self._ImgNorm(pil_img)[None]
        mask = ~(self._ToTensor(pil_img)[None].sum(1) <= 0.01)

        return {
            'img': img_tensor,
            'true_shape': np.int32([[H, W]]),
            'idx': 0,
            'instance': 'frame',
            'mask': mask,
            'dynamic_mask': torch.zeros_like(mask),
        }

    def process_frame(
        self,
        img_curr: np.ndarray,
        img_prev: Optional[np.ndarray] = None,
        depth_curr: Optional[np.ndarray] = None,
        depth_prev: Optional[np.ndarray] = None,
        baseline: float = 1.0,
        **kwargs
    ) -> PoseResult:
        """Process frame pair with MonST3R.

        Uses pairwise mode for fair comparison with other methods.
        """
        if not self._initialized:
            raise RuntimeError("Runner not initialized. Call initialize() first.")

        if img_prev is None:
            return PoseResult(R=np.eye(3), t=np.zeros(3), success=False)

        try:
            from dust3r.inference import inference
            from dust3r.image_pairs import make_pairs
            from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

            # Prepare images
            img0_dict = self._prepare_image(img_prev)
            img0_dict['idx'] = 0
            img0_dict['instance'] = 'frame_0'

            img1_dict = self._prepare_image(img_curr)
            img1_dict['idx'] = 1
            img1_dict['instance'] = 'frame_1'

            imgs = [img0_dict, img1_dict]

            # Make pairs
            pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)

            # Inference
            output = inference(pairs, self._model, self.device, batch_size=1, verbose=False)

            # Global alignment (pair mode)
            scene = global_aligner(output, device=self.device,
                                  mode=GlobalAlignerMode.PairViewer, verbose=False)

            # Get poses
            poses_c2w = scene.get_im_poses().detach().cpu().numpy()

            if len(poses_c2w) < 2:
                return PoseResult(R=np.eye(3), t=np.zeros(3), success=False)

            # Compute relative pose
            T0 = poses_c2w[0]
            T1 = poses_c2w[1]
            T_rel = np.linalg.inv(T1) @ T0

            R_est = T_rel[:3, :3]
            t_est = T_rel[:3, 3]

            # Normalize translation (up-to-scale)
            t_norm = np.linalg.norm(t_est)
            if t_norm > 1e-6:
                t_est = t_est / t_norm

            return PoseResult(
                R=R_est,
                t=t_est,
                success=True,
                extra={
                    'scale': float(t_norm),
                }
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return PoseResult(
                R=np.eye(3),
                t=np.zeros(3),
                success=False,
                extra={'error': str(e)}
            )

    def reset(self):
        """Reset state (MonST3R is stateless per-pair)."""
        pass
