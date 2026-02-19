"""DepthAnything v2 depth estimator."""

import cv2
import numpy as np
from pathlib import Path
from .base import DepthEstimator


def _preprocess(image: np.ndarray, input_size: int = 518) -> tuple:
    """Preprocess image for DepthAnything v2."""
    h, w = image.shape[:2]

    # BGR to RGB and normalize
    if len(image.shape) == 3 and image.shape[2] == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        img = image
    img = img.astype(np.float32) / 255.0

    # Fixed size resize
    img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    # HWC -> NCHW
    img = img.transpose(2, 0, 1)[np.newaxis, ...]

    return img.astype(np.float32), (h, w)


def _postprocess(depth: np.ndarray, orig_size: tuple) -> np.ndarray:
    """Postprocess depth output."""
    h, w = orig_size

    if len(depth.shape) == 4:
        depth = depth[0, 0]
    elif len(depth.shape) == 3:
        depth = depth[0]

    # Resize to original
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 1] (inverse depth: 0=far, 1=close)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    return depth


class DepthAnythingEstimator(DepthEstimator):
    """DepthAnything v2 with ONNX Runtime.

    Outputs normalized inverse depth [0, 1] where 1 is close, 0 is far.
    NOT metric depth - requires scale factor for metric conversion.
    """

    def __init__(
        self,
        model_path: str = None,
        input_size: int = 518,
        encoder: str = 'vitl',
        reinit_interval: int = 200  # Reinitialize session every N frames to prevent memory leak
    ):
        """Initialize DepthAnything estimator.

        Args:
            model_path: Path to ONNX model. If None, uses default.
            input_size: Input size for the model.
            encoder: Encoder type (vits, vitb, vitl).
            reinit_interval: Reinitialize ONNX session every N frames to prevent CUDA memory leak.
        """
        import onnxruntime as ort
        self._ort = ort

        if model_path is None:
            repo_root = Path(__file__).parent.parent.parent.parent
            model_path = repo_root / "weights" / "depth_anything_v2" / f"depth_anything_v2_{encoder}_{input_size}.onnx"

        self.model_path = str(model_path)
        self.input_size = input_size
        self._name = f"DepthAnything-{encoder}"
        self.reinit_interval = reinit_interval
        self._infer_count = 0

        self._create_session()

    def _create_session(self):
        """Create or recreate the ONNX session."""
        import gc

        # Clean up old session if exists
        if hasattr(self, 'session') and self.session is not None:
            # Force sync before deletion
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except ImportError:
                pass

            del self.session
            self.session = None
            gc.collect()
            gc.collect()  # Double collect for thorough cleanup

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # Additional cleanup
                    torch.cuda.ipc_collect()
            except (ImportError, AttributeError):
                pass

        ort = self._ort

        # ONNX Runtime setup with memory-efficient settings
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested',
                'cudnn_conv_algo_search': 'HEURISTIC',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider'
        ]
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = False

        self.session = ort.InferenceSession(self.model_path, sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Warmup
        dummy = np.random.randn(1, 3, self.input_size, self.input_size).astype(np.float32)
        self.session.run([self.output_name], {self.input_name: dummy})

    def infer(self, image: np.ndarray) -> np.ndarray:
        """Estimate inverse depth from image.

        Args:
            image: BGR image (H, W, 3) uint8

        Returns:
            Normalized inverse depth [0, 1], shape (H, W)
            0 = far, 1 = close
        """
        # Reinitialize session periodically to prevent CUDA memory leak
        self._infer_count += 1
        if self.reinit_interval > 0 and self._infer_count % self.reinit_interval == 0:
            self._create_session()

        input_tensor, orig_size = _preprocess(image, self.input_size)
        output = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        return _postprocess(output, orig_size)

    @property
    def is_metric(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return self._name
