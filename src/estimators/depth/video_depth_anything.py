"""Video Depth Anything depth estimator with streaming and offline modes."""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
from typing import Optional, List
from .base import DepthEstimator

# Add external video_depth_anything to path
EXTERNAL_VDA_PATH = Path(__file__).parent.parent.parent.parent / "external" / "video_depth_anything"

# Initialize VDA path once at module load
_VDA_PATH_INITIALIZED = False


def _ensure_vda_path():
    """Ensure VDA path is in sys.path for imports."""
    global _VDA_PATH_INITIALIZED
    if _VDA_PATH_INITIALIZED:
        return

    vda_path = str(EXTERNAL_VDA_PATH)
    if vda_path not in sys.path:
        # Insert at beginning to take precedence
        sys.path.insert(0, vda_path)

    _VDA_PATH_INITIALIZED = True


class VideoDepthAnythingEstimator(DepthEstimator):
    """Video Depth Anything estimator with streaming and offline modes.

    Streaming mode: Processes frames one at a time using cached temporal context.
                   Lower accuracy but suitable for real-time applications.

    Offline mode: Processes frames in batches with full temporal context.
                 Higher accuracy but requires buffering frames.

    Supports both relative depth (normalized) and metric depth (meters).
    """

    def __init__(
        self,
        encoder: str = 'vitl',
        metric: bool = True,
        streaming: bool = True,
        input_size: int = 518,
        max_res: int = 1280,
        fp32: bool = False,
        device: str = 'cuda',
        checkpoint_dir: Optional[str] = None,
    ):
        """Initialize Video Depth Anything estimator.

        Args:
            encoder: Encoder type ('vits', 'vitb', 'vitl').
            metric: Use metric depth model (meters) vs relative depth.
            streaming: Must be True. For offline batch mode, use
                      tools/eval/evaluate_vda_offline.py instead.
            input_size: Input size for the model (default 518).
            max_res: Maximum resolution (default 1280).
            fp32: Use float32 precision (default float16).
            device: Device to use ('cuda' or 'cpu').
            checkpoint_dir: Directory containing model checkpoints.
        """
        import torch

        if not streaming:
            raise ValueError(
                "Offline mode not supported via infer(). "
                "Use tools/eval/evaluate_vda_offline.py for batch processing."
            )

        self.encoder = encoder
        self.metric = metric
        self.streaming = True
        self.input_size = input_size
        self.max_res = max_res
        self.fp32 = fp32
        self.device = device

        # Set checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = EXTERNAL_VDA_PATH / "checkpoints"
        self.checkpoint_dir = Path(checkpoint_dir)

        # Build model name
        mode_str = "streaming" if streaming else "offline"
        depth_str = "metric" if metric else "relative"
        self._name = f"VideoDepthAnything-{encoder}-{depth_str}-{mode_str}"

        # Add to path and import
        if str(EXTERNAL_VDA_PATH) not in sys.path:
            sys.path.insert(0, str(EXTERNAL_VDA_PATH))

        # Model configurations
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        # Load model
        self._load_model()

        # For offline mode, we buffer frames
        self._frame_buffer: List[np.ndarray] = []
        self._depth_buffer: List[np.ndarray] = []
        self._buffer_idx = 0

    def _load_model(self):
        """Load the Video Depth Anything model."""
        import torch

        # Ensure VDA path is set up
        _ensure_vda_path()

        if self.streaming:
            from video_depth_anything.video_depth_stream import VideoDepthAnything
        else:
            from video_depth_anything.video_depth import VideoDepthAnything

        # Get checkpoint name
        checkpoint_name = 'metric_video_depth_anything' if self.metric else 'video_depth_anything'
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}_{self.encoder}.pth"

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Download from: https://huggingface.co/depth-anything/"
            )

        # Create model
        config = self.model_configs[self.encoder].copy()
        if not self.streaming:
            config['metric'] = self.metric

        self.model = VideoDepthAnything(**config)
        self.model.load_state_dict(
            torch.load(str(checkpoint_path), map_location='cpu'),
            strict=True
        )
        self.model = self.model.to(self.device).eval()

    def reset(self):
        """Reset internal state for new video sequence.

        Call this when starting a new video to clear cached temporal context.
        """
        self._frame_buffer.clear()
        self._depth_buffer.clear()
        self._buffer_idx = 0

        # Reset streaming model state
        if self.streaming and hasattr(self.model, 'frame_cache_list'):
            self.model.frame_cache_list = []
            self.model.frame_id_list = []
            self.model.id = -1
            self.model.transform = None

    def infer(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth from a single image.

        Args:
            image: BGR image (H, W, 3) uint8

        Returns:
            depth: Depth map (H, W) float32
                   - Metric mode: depth in meters
                   - Relative mode: raw depth values (not normalized)
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            frame = image

        if self.streaming:
            return self._infer_streaming(frame)
        else:
            return self._infer_offline(frame)

    def _infer_streaming(self, frame: np.ndarray) -> np.ndarray:
        """Streaming inference - process one frame at a time."""
        depth = self.model.infer_video_depth_one(
            frame,
            input_size=self.input_size,
            device=self.device,
            fp32=self.fp32
        )
        return depth.astype(np.float32)

    def _infer_offline(self, frame: np.ndarray) -> np.ndarray:
        """Offline inference with frame buffering.

        Buffers frames and processes in batches for better temporal consistency.
        Returns depth for frames as they become available.
        """
        import torch
        import torch.nn.functional as F

        INFER_LEN = 32
        OVERLAP = 10

        self._frame_buffer.append(frame)

        # If we have enough frames for a batch, process them
        if len(self._frame_buffer) >= INFER_LEN:
            frames = np.stack(self._frame_buffer[:INFER_LEN], axis=0)

            # Run offline inference
            depths, _ = self.model.infer_video_depth(
                frames,
                target_fps=30,  # Not used for output
                input_size=self.input_size,
                device=self.device,
                fp32=self.fp32
            )

            # Store depths and clear processed frames
            self._depth_buffer.extend([d for d in depths])

            # Keep overlap frames for next batch
            self._frame_buffer = self._frame_buffer[INFER_LEN - OVERLAP:]

        # Return depth if available, otherwise return placeholder
        if len(self._depth_buffer) > self._buffer_idx:
            depth = self._depth_buffer[self._buffer_idx]
            self._buffer_idx += 1
            return depth.astype(np.float32)
        else:
            # Not enough frames yet, return simple single-frame estimate
            # This happens at the start before we have enough frames
            return self._single_frame_fallback(frame)

    def _single_frame_fallback(self, frame: np.ndarray) -> np.ndarray:
        """Fallback for offline mode before enough frames are buffered."""
        import torch
        import torch.nn.functional as F
        from torchvision.transforms import Compose
        from video_depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

        h, w = frame.shape[:2]

        # Adjust input size for aspect ratio
        ratio = max(h, w) / min(h, w)
        input_size = self.input_size
        if ratio > 1.78:
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14

        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        # Transform and add batch/time dimensions
        img_tensor = torch.from_numpy(
            transform({'image': frame.astype(np.float32) / 255.0})['image']
        ).unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, C, H, W]

        with torch.no_grad():
            with torch.autocast(device_type=self.device, enabled=(not self.fp32)):
                depth = self.model(img_tensor)  # [1, 1, H, W]

        # Resize to original size
        depth = F.interpolate(
            depth.flatten(0, 1).unsqueeze(1),
            size=(h, w),
            mode='bilinear',
            align_corners=True
        )

        return depth[0, 0].cpu().numpy().astype(np.float32)

    def flush(self) -> List[np.ndarray]:
        """Flush remaining frames in offline mode.

        Call this at the end of a video to get depths for remaining buffered frames.

        Returns:
            List of depth maps for remaining frames.
        """
        if self.streaming or len(self._frame_buffer) == 0:
            return []

        # Pad with last frame to complete batch
        INFER_LEN = 32
        while len(self._frame_buffer) < INFER_LEN:
            self._frame_buffer.append(self._frame_buffer[-1].copy())

        frames = np.stack(self._frame_buffer[:INFER_LEN], axis=0)
        depths, _ = self.model.infer_video_depth(
            frames,
            target_fps=30,
            input_size=self.input_size,
            device=self.device,
            fp32=self.fp32
        )

        # Return only the depths for actual frames (not padding)
        remaining = len(self._frame_buffer) - (INFER_LEN - len(self._frame_buffer))
        result = [d.astype(np.float32) for d in depths[:remaining]]

        self._frame_buffer.clear()
        return result

    @property
    def is_metric(self) -> bool:
        return self.metric

    @property
    def name(self) -> str:
        return self._name
