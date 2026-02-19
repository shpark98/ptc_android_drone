"""BaTrack method runner."""

import sys
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from ..base import BaseMethodRunner, PoseResult


class BaTrackRunner(BaseMethodRunner):
    """Runner for BaTrack (Bundle-Adjusting Tracker).

    BaTrack is a SLAM system that accumulates poses over time.
    Results are only available after finalize() is called.
    """

    def __init__(
        self,
        device: str = "cuda",
        batrack_root: Optional[Path] = None,
    ):
        """Initialize BaTrack runner.

        Args:
            device: Device for computation
            batrack_root: Path to BaTrack installation
        """
        super().__init__(device)

        if batrack_root is None:
            # Try to find it relative to this file
            batrack_root = Path(__file__).parent.parent.parent.parent.parent / 'external' / 'batrack'
        self.batrack_root = Path(batrack_root)

        self._tracker = None
        self._depth_model = None
        self._intrinsics_tensor = None
        self._poses_7dof = None
        self._timestamps = []

    @property
    def name(self) -> str:
        return "BaTrack"

    @property
    def requires_gt_baseline(self) -> bool:
        return False  # SLAM scale

    @property
    def is_metric(self) -> bool:
        return False  # Up-to-scale SLAM

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
        """Initialize BaTrack."""
        # Add BaTrack paths
        sys.path.insert(0, str(self.batrack_root / 'main'))
        sys.path.insert(0, str(self.batrack_root / 'Depth-Anything'))

        try:
            from omegaconf import OmegaConf
            from batrack import BATRACK
            from depth_anything_v2.dpt import DepthAnythingV2
            import torch
        except ImportError as e:
            raise ImportError(f"BaTrack dependencies not found: {e}")

        # Load config
        config_path = self.batrack_root / 'configs' / 'sintel.yaml'
        cfg = OmegaConf.load(config_path)
        cfg.model.init_dir = str(self.batrack_root / 'checkpoints' / 'md_tracker.pth')
        cfg.model.depth_model_dir = str(
            self.batrack_root / 'Depth-Anything' / 'checkpoints' / 'depth_anything_v2_vitl.pth'
        )

        # Initialize depth model
        self._depth_model = DepthAnythingV2(
            encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024]
        ).to(self.device)
        self._depth_model.load_state_dict(
            torch.load(cfg.model.depth_model_dir, map_location=self.device, weights_only=False)
        )
        self._depth_model.eval()

        # Initialize tracker
        self._tracker = BATRACK(cfg, ht=H, wd=W)

        # Intrinsics tensor
        self._intrinsics_tensor = torch.tensor(
            [fx, fy, cx, cy], dtype=torch.float32, device=self.device
        )

        self._H = H
        self._W = W
        self._initialized = True
        self._frame_idx = 0
        self._timestamps = []

    def process_frame(
        self,
        img_curr: np.ndarray,
        img_prev: Optional[np.ndarray] = None,
        depth_curr: Optional[np.ndarray] = None,
        depth_prev: Optional[np.ndarray] = None,
        baseline: float = 1.0,
        **kwargs
    ) -> PoseResult:
        """Process frame with BaTrack.

        Note: BaTrack accumulates frames internally. Real poses
        are only available after finalize().

        Returns dummy success=True to indicate frame was processed.
        """
        if not self._initialized:
            raise RuntimeError("Runner not initialized. Call initialize() first.")

        import torch
        import cv2

        try:
            img_rgb = cv2.cvtColor(img_curr, cv2.COLOR_BGR2RGB)

            # Compute depth
            with torch.no_grad():
                depth = self._depth_model.infer_image(img_curr)

            # Convert to tensors
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).to(self.device).float()
            depth_tensor = torch.from_numpy(depth).unsqueeze(0).to(self.device).float()
            depth_tensor = depth_tensor.clip(1e-2, 1e2)

            # Feed to tracker
            self._tracker(float(self._frame_idx), img_tensor, depth_tensor, self._intrinsics_tensor)
            self._timestamps.append(self._frame_idx)
            self._frame_idx += 1

            # Return placeholder (real poses from finalize)
            return PoseResult(
                R=np.eye(3),
                t=np.zeros(3),
                success=True,
                extra={'accumulated': True}
            )

        except Exception as e:
            return PoseResult(
                R=np.eye(3),
                t=np.zeros(3),
                success=False,
                extra={'error': str(e)}
            )

    def finalize(self) -> Optional[Dict[str, Any]]:
        """Finalize and get accumulated poses."""
        if self._tracker is None:
            return None

        from scipy.spatial.transform import Rotation

        try:
            poses_7dof, timestamps = self._tracker.terminate()

            if poses_7dof is None or len(poses_7dof) < 2:
                return None

            # Convert to relative poses
            relative_poses = []
            for i in range(1, len(poses_7dof)):
                t0, t1 = poses_7dof[i-1], poses_7dof[i]

                # Build transformation matrices
                R0 = Rotation.from_quat([t0[4], t0[5], t0[6], t0[3]]).as_matrix()
                R1 = Rotation.from_quat([t1[4], t1[5], t1[6], t1[3]]).as_matrix()

                T0 = np.eye(4)
                T0[:3, :3] = R0
                T0[:3, 3] = t0[:3]

                T1 = np.eye(4)
                T1[:3, :3] = R1
                T1[:3, 3] = t1[:3]

                T_rel = np.linalg.inv(T1) @ T0
                relative_poses.append((T_rel[:3, :3], T_rel[:3, 3]))

            return {
                'poses_7dof': poses_7dof,
                'timestamps': timestamps,
                'relative_poses': relative_poses,
            }

        except Exception as e:
            return {'error': str(e)}

    def reset(self):
        """Reset tracker state."""
        if self._tracker is not None and self._initialized:
            from omegaconf import OmegaConf
            from batrack import BATRACK

            config_path = self.batrack_root / 'configs' / 'sintel.yaml'
            cfg = OmegaConf.load(config_path)
            cfg.model.init_dir = str(self.batrack_root / 'checkpoints' / 'md_tracker.pth')
            cfg.model.depth_model_dir = str(
                self.batrack_root / 'Depth-Anything' / 'checkpoints' / 'depth_anything_v2_vitl.pth'
            )
            self._tracker = BATRACK(cfg, ht=self._H, wd=self._W)

        self._frame_idx = 0
        self._timestamps = []
