"""Data source abstraction for depth visualization."""

import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class DepthSource:
    """A named depth map for a single frame."""
    name: str
    depth: np.ndarray       # (H, W) float32, meters
    mask: np.ndarray         # (H, W) bool, valid pixels


@dataclass
class FrameData:
    """All data for one frame."""
    idx: int
    image: np.ndarray            # (H, W, 3) BGR uint8
    T_world_cam: np.ndarray      # (4, 4) world <- camera
    K: np.ndarray                # (3, 3) intrinsics
    depth_sources: List[DepthSource] = field(default_factory=list)


class DepthMethod(ABC):
    """Interface for a depth estimation method."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def estimate(
        self,
        image: np.ndarray,
        prev_image: Optional[np.ndarray] = None,
        baseline: float = 1.0,
        gt_depth: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """Return depth map (H, W) in meters."""
        pass

    def initialize(self, H: int, W: int, fx: float, fy: float, cx: float, cy: float):
        """Optional initialization with camera parameters."""
        pass

    def reset(self):
        """Reset internal state (for sequential methods like PR-Depth)."""
        pass


class PRDepthMethod(DepthMethod):
    """PR-Depth live inference wrapper."""

    def __init__(self, device: str = "cuda", encoder: str = "vitl"):
        self._device = device
        self._encoder = encoder
        self._runner = None
        self._depth_estimator = None
        self._prev_image = None
        self._prev_depth = None

    @property
    def name(self) -> str:
        return "PR-Depth"

    def initialize(self, H, W, fx, fy, cx, cy):
        from src.evaluation.runners.pr_depth import PRDepthRunner
        from src.estimators.depth import DepthAnythingEstimator

        self._runner = PRDepthRunner(device=self._device, min_baseline=0.05)
        self._runner.initialize(H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy)
        self._depth_estimator = DepthAnythingEstimator(encoder=self._encoder)

    def reset(self):
        if self._runner is not None:
            self._runner._pipeline.reset()
        self._prev_image = None
        self._prev_depth = None

    def estimate(self, image, prev_image=None, baseline=1.0, gt_depth=None,
                 gt_R=None, gt_t=None, **kwargs):
        inv_depth = self._depth_estimator.infer(image)

        prev_inv = None
        if self._prev_image is not None:
            prev_inv = self._prev_depth

        result = self._runner.process_frame(
            img_curr=image,
            img_prev=self._prev_image,
            depth_curr=inv_depth,
            depth_prev=prev_inv,
            baseline=baseline,
            gt_R=gt_R,
            gt_t=gt_t,
        )

        self._prev_image = image
        self._prev_depth = inv_depth

        if result.success and result.extra.get('z_refined') is not None:
            return result.extra['z_refined']
        elif result.extra.get('z_tri') is not None:
            return result.extra['z_tri']
        else:
            return np.zeros_like(inv_depth)


class DepthAnythingMethod(DepthMethod):
    """Depth Anything v2 (relative depth, median-scaled to GT)."""

    def __init__(self, encoder: str = "vitl"):
        self._encoder = encoder
        self._estimator = None

    @property
    def name(self) -> str:
        return "DA-v2"

    def initialize(self, H, W, fx, fy, cx, cy):
        from src.estimators.depth import DepthAnythingEstimator
        self._estimator = DepthAnythingEstimator(encoder=self._encoder)

    def estimate(self, image, prev_image=None, baseline=1.0, gt_depth=None, **kwargs):
        inv_depth = self._estimator.infer(image)  # [0, 1], 0=far 1=close

        # Convert inverse depth to metric depth via median scaling with GT
        inv_depth = np.clip(inv_depth, 1e-6, 1.0)
        rel_depth = 1.0 / inv_depth  # Relative metric-like depth

        if gt_depth is not None:
            gt_valid = gt_depth > 0
            if gt_valid.sum() > 100:
                scale = np.median(gt_depth[gt_valid]) / np.median(rel_depth[gt_valid])
                return (rel_depth * scale).astype(np.float32)

        return rel_depth.astype(np.float32)


class VideoDepthAnythingMethod(DepthMethod):
    """Video Depth Anything (metric depth)."""

    def __init__(self, encoder: str = "vitl", device: str = "cuda"):
        self._encoder = encoder
        self._device = device
        self._estimator = None

    @property
    def name(self) -> str:
        return "VDA"

    def initialize(self, H, W, fx, fy, cx, cy):
        from src.estimators.depth import VideoDepthAnythingEstimator
        self._estimator = VideoDepthAnythingEstimator(
            encoder=self._encoder, metric=True, device=self._device
        )

    def reset(self):
        if self._estimator is not None:
            self._estimator.reset()

    def estimate(self, image, **kwargs):
        return self._estimator.infer(image)


class UniDepthMethod(DepthMethod):
    """UniDepth (metric depth)."""

    def __init__(self):
        self._estimator = None

    @property
    def name(self) -> str:
        return "UniDepth"

    def initialize(self, H, W, fx, fy, cx, cy):
        from src.estimators.depth import UniDepthEstimator
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self._estimator = UniDepthEstimator(K=K)

    def estimate(self, image, **kwargs):
        return self._estimator.infer(image)


# Registry
METHOD_REGISTRY = {
    "pr_depth": PRDepthMethod,
    "da_v2": DepthAnythingMethod,
    "vda": VideoDepthAnythingMethod,
    "unidepth": UniDepthMethod,
}


def build_methods(names: List[str], device: str = "cuda", encoder: str = "vitl") -> Dict[str, DepthMethod]:
    """Build depth method instances from CLI names."""
    methods = {}
    for name in names:
        if name == "gt":
            continue
        if name not in METHOD_REGISTRY:
            raise ValueError(f"Unknown method: {name}. Available: {list(METHOD_REGISTRY.keys())}")
        if name in ("pr_depth", "vda"):
            methods[name] = METHOD_REGISTRY[name](device=device, encoder=encoder)
        elif name == "da_v2":
            methods[name] = METHOD_REGISTRY[name](encoder=encoder)
        else:
            methods[name] = METHOD_REGISTRY[name]()
    return methods


class DatasetSource:
    """Loads frames from BaseDataset and runs live depth estimation."""

    def __init__(
        self,
        dataset,
        methods: Dict[str, DepthMethod],
        include_gt: bool = True,
        max_depth: float = 80.0,
    ):
        self.dataset = dataset
        self.methods = methods
        self.include_gt = include_gt
        self.max_depth = max_depth

        # Build intrinsics
        fx, fy, cx, cy = dataset.get_intrinsics()
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        # Get image size
        H, W = dataset.get_image_size()
        self.H, self.W = H, W

        # Initialize all methods
        for m in self.methods.values():
            m.initialize(H, W, fx, fy, cx, cy)

        self._prev_image = None
        self._prev_idx = None

    def __len__(self) -> int:
        return len(self.dataset)

    def get_frame(self, idx: int) -> Optional[FrameData]:
        """Get frame data with all depth sources computed live."""
        sample = self.dataset.get(idx)
        if sample is None:
            return None

        image = sample['image_og']
        T_world_cam = self.dataset.get_pose_matrix(idx)
        gt_depth = sample.get('depth_og', None)

        # Compute baseline and relative pose if we have a previous frame
        baseline = 1.0
        gt_R, gt_t = None, None
        if idx > 0:
            try:
                baseline = self.dataset.get_baseline(idx)
                gt_R, gt_t = self.dataset.get_relative_pose(idx)
            except Exception:
                pass

        depth_sources = []

        # GT depth
        if self.include_gt and gt_depth is not None:
            mask = (gt_depth > 0) & (gt_depth < self.max_depth)
            depth_sources.append(DepthSource(name="GT", depth=gt_depth, mask=mask))

        # Each method
        for method_name, method in self.methods.items():
            try:
                depth = method.estimate(
                    image=image,
                    prev_image=self._prev_image,
                    baseline=baseline,
                    gt_depth=gt_depth,
                    gt_R=gt_R,
                    gt_t=gt_t,
                )
                mask = (depth > 0) & (depth < self.max_depth) & np.isfinite(depth)
                depth_sources.append(DepthSource(
                    name=method.name, depth=depth, mask=mask
                ))
            except Exception as e:
                print(f"[{method_name}] Error at frame {idx}: {e}")

        self._prev_image = image
        self._prev_idx = idx

        return FrameData(
            idx=idx,
            image=image,
            T_world_cam=T_world_cam,
            K=self.K,
            depth_sources=depth_sources,
        )

    def reset(self):
        """Reset all methods' internal state."""
        for m in self.methods.values():
            m.reset()
        self._prev_image = None
        self._prev_idx = None
