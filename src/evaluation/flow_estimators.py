"""Optical flow estimators for PR-Depth evaluation."""

import sys
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class BaseFlowEstimator(ABC):
    """Base class for optical flow estimators."""

    @abstractmethod
    def estimate(self, img0: np.ndarray, img1: np.ndarray) -> np.ndarray:
        """Estimate optical flow from img0 to img1.

        Args:
            img0: First image (H, W, 3) BGR uint8
            img1: Second image (H, W, 3) BGR uint8

        Returns:
            Optical flow (H, W, 2) float32, (dx, dy) in pixels
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the flow estimator."""
        pass


class NeuFlowV2Estimator(BaseFlowEstimator):
    """NeuFlow_v2 optical flow estimator.

    Fast and accurate learned optical flow.
    Uses pretrained weights (neuflow_mixed.pth by default).
    """

    def __init__(
        self,
        device: str = "cuda",
        weights: str = "mixed",  # "mixed", "sintel", or "things"
    ):
        """Initialize NeuFlow_v2.

        Args:
            device: Device for computation
            weights: Which pretrained weights to use
        """
        self.device = torch.device(device)
        self.weights = weights
        self._model = None
        self._initialized_size = None

    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return

        # Add NeuFlow_v2 to path
        import os
        neuflow_path = os.path.join(
            os.path.dirname(__file__), '../../external/NeuFlow_v2'
        )
        if neuflow_path not in sys.path:
            sys.path.insert(0, neuflow_path)

        from NeuFlow.neuflow import NeuFlow
        from NeuFlow.backbone_v7 import ConvBlock

        self._model = NeuFlow().to(self.device)

        # Load weights
        weight_file = os.path.join(neuflow_path, f'neuflow_{self.weights}.pth')
        checkpoint = torch.load(weight_file, map_location=self.device)
        self._model.load_state_dict(checkpoint['model'], strict=True)

        # Fuse conv and bn for faster inference
        def fuse_conv_and_bn(conv, bn):
            fusedconv = (
                torch.nn.Conv2d(
                    conv.in_channels,
                    conv.out_channels,
                    kernel_size=conv.kernel_size,
                    stride=conv.stride,
                    padding=conv.padding,
                    dilation=conv.dilation,
                    groups=conv.groups,
                    bias=True,
                )
                .requires_grad_(False)
                .to(conv.weight.device)
            )
            w_conv = conv.weight.clone().view(conv.out_channels, -1)
            w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
            fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
            b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
            b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
            fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
            return fusedconv

        for m in self._model.modules():
            if type(m) is ConvBlock:
                m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)
                m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)
                delattr(m, "norm1")
                delattr(m, "norm2")
                m.forward = m.forward_fuse

        self._model.eval()
        self._model.half()

    def _init_size(self, H: int, W: int):
        """Initialize model for specific image size."""
        # NeuFlow requires dimensions divisible by 16
        self._proc_H = ((H + 15) // 16) * 16
        self._proc_W = ((W + 15) // 16) * 16
        self._orig_H = H
        self._orig_W = W
        self._model.init_bhwd(1, self._proc_H, self._proc_W, str(self.device))
        self._initialized_size = (H, W)

    @property
    def name(self) -> str:
        return f"NeuFlow_v2_{self.weights}"

    def estimate(self, img0: np.ndarray, img1: np.ndarray) -> np.ndarray:
        """Estimate optical flow from img0 to img1.

        Args:
            img0: First image (H, W, 3) BGR uint8
            img1: Second image (H, W, 3) BGR uint8

        Returns:
            Optical flow (H, W, 2) float32, (dx, dy) in pixels
        """
        self._load_model()

        H, W = img0.shape[:2]

        # Initialize for this size if needed
        if self._initialized_size != (H, W):
            self._init_size(H, W)

        # Preprocess: resize to processing size
        import cv2
        if (H, W) != (self._proc_H, self._proc_W):
            img0_proc = cv2.resize(img0, (self._proc_W, self._proc_H))
            img1_proc = cv2.resize(img1, (self._proc_W, self._proc_H))
        else:
            img0_proc = img0
            img1_proc = img1

        # Convert to tensor (BGR -> keep BGR, NeuFlow uses BGR)
        img0_t = torch.from_numpy(img0_proc).permute(2, 0, 1).half()[None].to(self.device)
        img1_t = torch.from_numpy(img1_proc).permute(2, 0, 1).half()[None].to(self.device)

        # Run inference
        with torch.no_grad():
            flow = self._model(img0_t, img1_t)[-1][0]  # Last scale, first batch
            flow = flow.permute(1, 2, 0).cpu().numpy()  # (H, W, 2)

        # Scale flow back to original resolution if needed
        if (H, W) != (self._proc_H, self._proc_W):
            scale_x = W / self._proc_W
            scale_y = H / self._proc_H
            flow = cv2.resize(flow, (W, H))
            flow[..., 0] *= scale_x
            flow[..., 1] *= scale_y

        return flow.astype(np.float32)


class RAFTEstimator(BaseFlowEstimator):
    """RAFT optical flow estimator from torchvision.

    RAFT (Recurrent All-Pairs Field Transforms) is a widely-used
    learned optical flow method. Uses pretrained weights from torchvision.

    Available weights:
    - "large": RAFT-Large trained on FlyingChairs + FlyingThings3D
    - "small": RAFT-Small (faster, less accurate)
    """

    def __init__(
        self,
        device: str = "cuda",
        model_size: str = "large",  # "large" or "small"
    ):
        """Initialize RAFT.

        Args:
            device: Device for computation
            model_size: Model size ("large" or "small")
        """
        self.device = torch.device(device)
        self.model_size = model_size
        self._model = None

    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return

        from torchvision.models.optical_flow import (
            raft_large, raft_small,
            Raft_Large_Weights, Raft_Small_Weights
        )

        if self.model_size == "large":
            self._model = raft_large(weights=Raft_Large_Weights.DEFAULT)
        else:
            self._model = raft_small(weights=Raft_Small_Weights.DEFAULT)

        self._model = self._model.to(self.device)
        self._model.eval()

    @property
    def name(self) -> str:
        return f"RAFT_{self.model_size}"

    def estimate(self, img0: np.ndarray, img1: np.ndarray) -> np.ndarray:
        """Estimate optical flow from img0 to img1.

        Args:
            img0: First image (H, W, 3) BGR uint8
            img1: Second image (H, W, 3) BGR uint8

        Returns:
            Optical flow (H, W, 2) float32, (dx, dy) in pixels
        """
        self._load_model()
        import cv2

        H, W = img0.shape[:2]

        # RAFT requires dimensions divisible by 8
        proc_H = ((H + 7) // 8) * 8
        proc_W = ((W + 7) // 8) * 8

        # Resize if needed
        if (H, W) != (proc_H, proc_W):
            img0_proc = cv2.resize(img0, (proc_W, proc_H))
            img1_proc = cv2.resize(img1, (proc_W, proc_H))
        else:
            img0_proc = img0
            img1_proc = img1

        # Convert BGR to RGB and to tensor
        # RAFT expects RGB images normalized to [-1, 1]
        img0_rgb = cv2.cvtColor(img0_proc, cv2.COLOR_BGR2RGB)
        img1_rgb = cv2.cvtColor(img1_proc, cv2.COLOR_BGR2RGB)

        # Normalize: [0, 255] -> [0, 1] -> [-1, 1]
        img0_t = torch.from_numpy(img0_rgb).permute(2, 0, 1).float()[None].to(self.device)
        img1_t = torch.from_numpy(img1_rgb).permute(2, 0, 1).float()[None].to(self.device)
        img0_t = img0_t / 255.0 * 2.0 - 1.0  # [-1, 1]
        img1_t = img1_t / 255.0 * 2.0 - 1.0  # [-1, 1]

        # Run inference
        with torch.no_grad():
            # RAFT returns list of flow predictions at different iterations
            # Use the last (most refined) prediction
            flow_predictions = self._model(img0_t, img1_t)
            flow = flow_predictions[-1][0]  # Last iteration, first batch
            flow = flow.permute(1, 2, 0).cpu().numpy()  # (H, W, 2)

        # Scale flow back to original resolution if needed
        if (H, W) != (proc_H, proc_W):
            scale_x = W / proc_W
            scale_y = H / proc_H
            flow = cv2.resize(flow, (W, H))
            flow[..., 0] *= scale_x
            flow[..., 1] *= scale_y

        # DEBUG: Try negating flow to test convention
        # flow = -flow

        return flow.astype(np.float32)


class GMFlowEstimator(BaseFlowEstimator):
    """GMFlow optical flow estimator.

    GMFlow uses global matching instead of local correlation,
    which may preserve global flow structure better for motion field decomposition.

    Paper: "GMFlow: Learning Optical Flow via Global Matching" (CVPR 2022 Oral)
    https://github.com/haofeixu/gmflow

    Available weights:
    - "sintel": Best for general use
    - "things": Trained on FlyingThings3D
    - "kitti": Trained on KITTI
    - "chairs": Trained on FlyingChairs
    """

    def __init__(
        self,
        device: str = "cuda",
        weights: str = "sintel",  # "sintel", "things", "kitti", "chairs"
        with_refine: bool = True,  # Use refinement version
    ):
        """Initialize GMFlow.

        Args:
            device: Device for computation
            weights: Which pretrained weights to use
            with_refine: Whether to use the refinement version (more accurate)
        """
        self.device = torch.device(device)
        self.weights = weights
        self.with_refine = with_refine
        self._model = None

    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return

        import os

        # Add GMFlow to path
        gmflow_path = os.path.join(
            os.path.dirname(__file__), '../../external/GMFlow'
        )
        if gmflow_path not in sys.path:
            sys.path.insert(0, gmflow_path)

        from gmflow.gmflow import GMFlow

        # Model config based on whether using refinement
        if self.with_refine:
            self._model = GMFlow(
                num_scales=2,
                upsample_factor=4,
                feature_channels=128,
                attention_type='swin',
                num_transformer_layers=6,
                ffn_dim_expansion=4,
                num_head=1,
            )
            weight_file = os.path.join(
                gmflow_path, 'pretrained', f'gmflow_with_refine_{self.weights}-*.pth'
            )
        else:
            self._model = GMFlow(
                num_scales=1,
                upsample_factor=8,
                feature_channels=128,
                attention_type='swin',
                num_transformer_layers=6,
                ffn_dim_expansion=4,
                num_head=1,
            )
            weight_file = os.path.join(
                gmflow_path, 'pretrained', f'gmflow_{self.weights}-*.pth'
            )

        # Find actual weight file (with hash suffix)
        import glob
        weight_files = glob.glob(weight_file)
        if not weight_files:
            raise FileNotFoundError(f"GMFlow weights not found: {weight_file}")
        weight_file = weight_files[0]

        # Load weights
        checkpoint = torch.load(weight_file, map_location=self.device)
        self._model.load_state_dict(checkpoint['model'], strict=True)

        self._model = self._model.to(self.device)
        self._model.eval()

    @property
    def name(self) -> str:
        refine_str = "_refine" if self.with_refine else ""
        return f"GMFlow{refine_str}_{self.weights}"

    def estimate(self, img0: np.ndarray, img1: np.ndarray) -> np.ndarray:
        """Estimate optical flow from img0 to img1.

        Args:
            img0: First image (H, W, 3) BGR uint8
            img1: Second image (H, W, 3) BGR uint8

        Returns:
            Optical flow (H, W, 2) float32, (dx, dy) in pixels
        """
        self._load_model()
        import cv2

        H, W = img0.shape[:2]

        # GMFlow requires dimensions divisible by 32 (for swin transformer)
        proc_H = ((H + 31) // 32) * 32
        proc_W = ((W + 31) // 32) * 32

        # Resize if needed
        if (H, W) != (proc_H, proc_W):
            img0_proc = cv2.resize(img0, (proc_W, proc_H))
            img1_proc = cv2.resize(img1, (proc_W, proc_H))
        else:
            img0_proc = img0
            img1_proc = img1

        # Convert BGR to RGB and to tensor
        img0_rgb = cv2.cvtColor(img0_proc, cv2.COLOR_BGR2RGB)
        img1_rgb = cv2.cvtColor(img1_proc, cv2.COLOR_BGR2RGB)

        # GMFlow normalizes internally, expects [0, 255] float
        img0_t = torch.from_numpy(img0_rgb).permute(2, 0, 1).float()[None].to(self.device)
        img1_t = torch.from_numpy(img1_rgb).permute(2, 0, 1).float()[None].to(self.device)

        # Run inference
        with torch.no_grad():
            # GMFlow inference parameters
            if self.with_refine:
                # With refinement: 2 scales
                attn_splits_list = [2, 8]
                corr_radius_list = [-1, 4]
                prop_radius_list = [-1, 1]
            else:
                # Without refinement: 1 scale
                attn_splits_list = [2]
                corr_radius_list = [-1]
                prop_radius_list = [-1]

            results = self._model(
                img0_t, img1_t,
                attn_splits_list=attn_splits_list,
                corr_radius_list=corr_radius_list,
                prop_radius_list=prop_radius_list,
            )

            flow = results['flow_preds'][-1][0]  # Last prediction, first batch
            flow = flow.permute(1, 2, 0).cpu().numpy()  # (H, W, 2)

        # Scale flow back to original resolution if needed
        if (H, W) != (proc_H, proc_W):
            scale_x = W / proc_W
            scale_y = H / proc_H
            flow = cv2.resize(flow, (W, H))
            flow[..., 0] *= scale_x
            flow[..., 1] *= scale_y

        return flow.astype(np.float32)


class WAFTEstimator(BaseFlowEstimator):
    """WAFT optical flow estimator.

    WAFT (Warping-Alone Field Transforms) replaces cost volume with
    high-resolution warping, achieving better accuracy with lower memory cost.
    Ranks 1st on Spring, Sintel, and KITTI benchmarks.

    Paper: "WAFT: Warping-Alone Field Transforms for Optical Flow" (2025)
    https://github.com/princeton-vl/WAFT

    Available weights (backbone):
    - "twins": Twins backbone (default, best zero-shot generalization)
    - "dav2": DepthAnythingV2 backbone
    - "dinov3": DINOv3 backbone
    """

    def __init__(
        self,
        device: str = "cuda",
        backbone: str = "twins",  # "twins", "dav2", "dinov3"
        checkpoint: str = None,  # Path to checkpoint, auto-detect if None
    ):
        """Initialize WAFT.

        Args:
            device: Device for computation
            backbone: Feature encoder backbone
            checkpoint: Path to checkpoint file
        """
        self.device = torch.device(device)
        self.backbone = backbone
        self.checkpoint = checkpoint
        self._model = None
        self._wrapper = None

    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return

        import os
        import argparse

        # Add WAFT to path - must handle conflicts with RAFT/monst3r/batrack
        waft_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '../../external/WAFT'
        ))

        # Save original state
        original_cwd = os.getcwd()
        original_path = sys.path.copy()

        # Remove ALL conflicting paths (anything in external except WAFT)
        conflicting_paths = ['RAFT', 'monst3r', 'SEA-RAFT', 'batrack', 'external']

        # Save and remove conflicting modules
        conflicting_mods = ['utils', 'model', 'inference_tools']
        for mod_name in list(sys.modules.keys()):
            for conflict in conflicting_mods:
                if mod_name == conflict or mod_name.startswith(conflict + '.'):
                    del sys.modules[mod_name]

        # Build clean path - ONLY system paths plus WAFT
        clean_path = [waft_path]
        for p in original_path:
            # Only include system Python paths, not any project paths
            if 'site-packages' in p or 'python3' in p:
                clean_path.append(p)

        # Change to WAFT directory for relative imports
        os.chdir(waft_path)
        sys.path = clean_path

        try:
            from model import fetch_model
            from inference_tools import InferenceWrapper
            from utils.utils import load_ckpt
        except Exception as e:
            os.chdir(original_cwd)
            sys.path = original_path
            raise e

        # Create args namespace for model config (using waft-a2 with multiple backbone support)
        args = argparse.Namespace(
            algorithm='waft-a2',
            feature_encoder=self.backbone,  # "twins", "dav2", or "dinov3"
            iterative_module='vits',  # ViT-S for iterative refinement
            use_var=True,
            var_min=0,
            var_max=10,
            iters=5,
            image_size=[432, 960],
        )

        # Create model
        self._model = fetch_model(args)

        # Find checkpoint
        if self.checkpoint is None:
            # Auto-detect checkpoint - check multiple paths (a2 structure: waftv2-ckpts/backbone/)
            possible_dirs = [
                os.path.join(waft_path, 'ckpts', 'waftv2-ckpts', self.backbone),
                os.path.join(waft_path, 'ckpts', self.backbone),
                os.path.join(waft_path, 'ckpts'),
            ]
            import glob
            for ckpt_dir in possible_dirs:
                if os.path.exists(ckpt_dir):
                    ckpts = glob.glob(os.path.join(ckpt_dir, '*.pth'))
                    if ckpts:
                        # Prefer sintel or things weights for generalization
                        for pattern in ['sintel', 'things', 'chairs-things']:
                            for ckpt in ckpts:
                                if pattern in os.path.basename(ckpt):
                                    self.checkpoint = ckpt
                                    break
                            if self.checkpoint:
                                break
                        if not self.checkpoint:
                            self.checkpoint = ckpts[0]
                        break

        if self.checkpoint and os.path.exists(self.checkpoint):
            load_ckpt(self._model, self.checkpoint)
        else:
            print(f"[WAFT] Warning: No checkpoint found. Download from: "
                  f"https://drive.google.com/drive/folders/1joBWKGoH2RUdCgcge8Tz2osOHcQUX5m_")

        self._model = self._model.to(self.device)
        self._model.eval()

        # Create inference wrapper
        self._wrapper = InferenceWrapper(
            self._model,
            scale=0,
            train_size=args.image_size,
            pad_to_train_size=False,
            tiling=False
        )

        # Now restore original working directory and path
        os.chdir(original_cwd)
        sys.path = original_path

    @property
    def name(self) -> str:
        return f"WAFT_{self.backbone}"

    def estimate(self, img0: np.ndarray, img1: np.ndarray) -> np.ndarray:
        """Estimate optical flow from img0 to img1.

        Args:
            img0: First image (H, W, 3) BGR uint8
            img1: Second image (H, W, 3) BGR uint8

        Returns:
            Optical flow (H, W, 2) float32, (dx, dy) in pixels
        """
        self._load_model()
        import cv2

        H, W = img0.shape[:2]

        # Convert BGR to RGB
        img0_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        # Convert to tensor: WAFT expects [0, 255] RGB
        # Use float16 for xformers compatibility with newer GPUs
        img0_t = torch.from_numpy(img0_rgb).permute(2, 0, 1).half()[None].to(self.device)
        img1_t = torch.from_numpy(img1_rgb).permute(2, 0, 1).half()[None].to(self.device)

        # Run inference
        with torch.no_grad(), torch.amp.autocast('cuda'):
            output = self._wrapper.calc_flow(img0_t, img1_t)
            flow = output['flow'][-1][0]  # Last iteration, first batch
            flow = flow.float().permute(1, 2, 0).cpu().numpy()  # (H, W, 2)

        return flow.astype(np.float32)


def get_flow_estimator(name: str, device: str = "cuda") -> Optional[BaseFlowEstimator]:
    """Get flow estimator by name.

    Args:
        name: Flow estimator name
        device: Device for computation

    Available estimators:
        - "raft" or "raft_large": RAFT-Large (torchvision, best generalization)
        - "raft_small": RAFT-Small (faster)
        - "neuflow" or "neuflow_mixed": NeuFlow_v2 mixed weights
        - "neuflow_sintel": NeuFlow_v2 Sintel weights
        - "neuflow_things": NeuFlow_v2 FlyingThings weights
        - "gmflow" or "gmflow_sintel": GMFlow with refinement (sintel weights)
        - "gmflow_things": GMFlow with refinement (things weights)
        - "gmflow_norefine": GMFlow without refinement (faster)
        - "waft" or "waft_twins": WAFT with Twins backbone (SOTA, best accuracy)
        - "waft_dav2": WAFT with DepthAnythingV2 backbone
        - "waft_dinov3": WAFT with DINOv3 backbone

    Returns:
        Flow estimator instance or None if not found
    """
    name = name.lower()

    # RAFT (torchvision) - best generalization
    if name == "raft" or name == "raft_large":
        return RAFTEstimator(device=device, model_size="large")
    elif name == "raft_small":
        return RAFTEstimator(device=device, model_size="small")

    # NeuFlow_v2
    elif name == "neuflow" or name == "neuflow_mixed":
        return NeuFlowV2Estimator(device=device, weights="mixed")
    elif name == "neuflow_sintel":
        return NeuFlowV2Estimator(device=device, weights="sintel")
    elif name == "neuflow_things":
        return NeuFlowV2Estimator(device=device, weights="things")

    # GMFlow (global matching)
    elif name == "gmflow" or name == "gmflow_sintel":
        return GMFlowEstimator(device=device, weights="sintel", with_refine=True)
    elif name == "gmflow_things":
        return GMFlowEstimator(device=device, weights="things", with_refine=True)
    elif name == "gmflow_norefine":
        return GMFlowEstimator(device=device, weights="sintel", with_refine=False)

    # WAFT (state-of-the-art, warping-based)
    elif name == "waft" or name == "waft_twins":
        return WAFTEstimator(device=device, backbone="twins")
    elif name == "waft_dav2":
        return WAFTEstimator(device=device, backbone="dav2")
    elif name == "waft_dinov3":
        return WAFTEstimator(device=device, backbone="dinov3")

    else:
        return None
