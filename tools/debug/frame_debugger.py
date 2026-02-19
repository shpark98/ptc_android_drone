#!/usr/bin/env python3
"""Frame-level debugger for PR-Depth pipeline.

Provides detailed analysis of individual frames including:
- Forward/backward pose estimation
- DC score comparison
- Triangulated vs warped depth
- Fusion variance analysis

Usage:
    from tools.debug import FrameDebugger

    debugger = FrameDebugger(dataset, config)
    result = debugger.analyze_frame(frame_idx)
    debugger.visualize(result, save_path='debug_frame_123.png')
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt


@dataclass
class DebugResult:
    """Container for frame debug information."""
    frame_idx: int

    # Pose info
    R_forward: np.ndarray = None
    t_forward: np.ndarray = None
    R_backward: np.ndarray = None
    t_backward: np.ndarray = None
    used_backward: bool = False

    # DC scores
    dc_score_forward: float = -1.0
    dc_score_backward: float = -1.0

    # Depth maps
    z_tri_forward: np.ndarray = None
    z_tri_backward: np.ndarray = None
    z_warp_flow: np.ndarray = None
    z_warp_pose: np.ndarray = None
    z_refined: np.ndarray = None
    prev_depth_used: np.ndarray = None

    # GT depth (if available)
    z_gt: np.ndarray = None

    # Variance
    V_prior: np.ndarray = None
    V_post: np.ndarray = None

    # Accuracy metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Raw result dict from C++
    raw_result: Dict[str, Any] = field(default_factory=dict)


class FrameDebugger:
    """Debug analyzer for PR-Depth pipeline."""

    def __init__(self, dataset, config=None, device='cuda'):
        """Initialize debugger.

        Args:
            dataset: KITTI dataset instance with rgb_path, depth_path
            config: Optional DepthRefinementConfig (debug=True will be set)
            device: Device for depth estimation
        """
        self.dataset = dataset
        self.device = device

        # Import here to avoid circular imports
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'cpp' / 'build'))
        import pr_depth_cpp

        # Setup config with debug enabled
        if config is None:
            self.config = pr_depth_cpp.DepthRefinementConfig()
            # Use camera intrinsics from dataset
            fx, fy, cx, cy = dataset.get_intrinsics()
            self.config.fx = fx
            self.config.fy = fy
            self.config.cx = cx
            self.config.cy = cy
            # Get image size from first frame
            sample = dataset.get(0)
            self.config.H = sample['image'].shape[0]
            self.config.W = sample['image'].shape[1]
        else:
            self.config = config

        # Enable debug mode
        self.config.debug = True

        self.pipeline = pr_depth_cpp.DepthRefinement(self.config)
        self.depth_estimator = None

    def _init_depth_estimator(self):
        """Lazy initialization of depth estimator."""
        if self.depth_estimator is None:
            from src.estimators.depth import DepthAnythingEstimator
            self.depth_estimator = DepthAnythingEstimator(encoder='vitl')

    def analyze_frame(self, frame_idx: int, warmup_frames: int = 5) -> DebugResult:
        """Analyze a specific frame with debug output.

        Args:
            frame_idx: Target frame index to analyze
            warmup_frames: Number of frames to process before target for state warmup

        Returns:
            DebugResult with all debug information
        """
        self._init_depth_estimator()
        self.pipeline.reset()

        # Determine start index for warmup
        start_idx = max(0, frame_idx - warmup_frames)

        result = DebugResult(frame_idx=frame_idx)

        # Process frames up to and including target
        for idx in range(start_idx, frame_idx + 1):
            sample = self.dataset.get(idx)
            img = sample['image']

            # Get depth estimation
            inv_depth = self.depth_estimator.infer(img)

            # Get baseline from GT if available
            if idx > start_idx:
                baseline = self.dataset.get_baseline(idx)
            else:
                baseline = 0.5  # Default baseline

            # Convert to BGR for C++
            if img.shape[-1] == 3 and len(img.shape) == 3:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img

            # Run pipeline
            raw_result = self.pipeline.refine(img_bgr, inv_depth, baseline)

            # Store result for target frame
            if idx == frame_idx:
                result.raw_result = raw_result
                result.z_refined = raw_result['z_refined']

                # Extract debug info
                debug = raw_result.get('debug', {})

                result.dc_score_forward = debug.get('dc_score_forward', -1.0)
                result.dc_score_backward = debug.get('dc_score_backward', -1.0)
                result.used_backward = debug.get('used_backward', False)

                result.R_forward = debug.get('R_forward')
                result.t_forward = debug.get('t_forward')
                result.R_backward = debug.get('R_backward')
                result.t_backward = debug.get('t_backward')

                result.z_tri_forward = debug.get('z_tri_forward')
                result.z_tri_backward = debug.get('z_tri_backward')
                result.z_warp_flow = debug.get('z_warp_flow')
                result.z_warp_pose = debug.get('z_warp_pose')
                result.prev_depth_used = debug.get('prev_depth_used')
                result.V_prior = debug.get('V_prior')
                result.V_post = debug.get('V_post')

                # Get GT depth if available
                if sample.get('depth') is not None:
                    result.z_gt = sample['depth']

                # Compute accuracy metrics
                result.metrics = self._compute_metrics(result)

        return result

    def _compute_metrics(self, result: DebugResult) -> Dict[str, float]:
        """Compute depth accuracy metrics."""
        metrics = {}

        if result.z_gt is None:
            return metrics

        z_gt = result.z_gt
        mask = (z_gt > 0.1) & (z_gt < 80) & np.isfinite(z_gt)

        def compute_d125(z_pred, z_gt, mask):
            """Compute delta < 1.25 accuracy."""
            valid = mask & np.isfinite(z_pred) & (z_pred > 0)
            if valid.sum() < 100:
                return -1.0
            ratio = np.maximum(z_pred[valid] / z_gt[valid], z_gt[valid] / z_pred[valid])
            return 100.0 * (ratio < 1.25).mean()

        # Refined depth accuracy
        if result.z_refined is not None:
            metrics['ref_d125'] = compute_d125(result.z_refined, z_gt, mask)

        # Forward triangulation accuracy
        if result.z_tri_forward is not None:
            # Scale to match GT
            z_tri = result.z_tri_forward
            valid = mask & np.isfinite(z_tri) & (z_tri > 0)
            if valid.sum() > 100:
                scale = np.median(z_gt[valid] / z_tri[valid])
                metrics['tri_fwd_d125'] = compute_d125(z_tri * scale, z_gt, mask)
                metrics['tri_fwd_scale'] = scale

        # Backward triangulation accuracy
        if result.z_tri_backward is not None:
            z_tri = result.z_tri_backward
            valid = mask & np.isfinite(z_tri) & (z_tri > 0)
            if valid.sum() > 100:
                scale = np.median(z_gt[valid] / z_tri[valid])
                metrics['tri_bwd_d125'] = compute_d125(z_tri * scale, z_gt, mask)
                metrics['tri_bwd_scale'] = scale

        # Warped prev depth accuracy
        if result.z_warp_flow is not None:
            metrics['warp_flow_d125'] = compute_d125(result.z_warp_flow, z_gt, mask)

        if result.z_warp_pose is not None:
            metrics['warp_pose_d125'] = compute_d125(result.z_warp_pose, z_gt, mask)

        return metrics

    def visualize(self, result: DebugResult, save_path: Optional[str] = None,
                  show: bool = True) -> Optional[np.ndarray]:
        """Create visualization of debug result.

        Args:
            result: DebugResult from analyze_frame
            save_path: Optional path to save visualization
            show: Whether to display plot

        Returns:
            Visualization image if save_path is provided
        """
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))

        # Helper to display depth map
        def show_depth(ax, depth, title, vmax=None):
            if depth is None or depth.size == 0:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
                return

            valid = np.isfinite(depth) & (depth > 0)
            if valid.sum() < 10:
                ax.text(0.5, 0.5, 'No valid', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
                return

            if vmax is None:
                vmax = np.percentile(depth[valid], 95)

            im = ax.imshow(depth, vmin=0, vmax=vmax, cmap='turbo')
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Row 1: GT, Refined, Error
        show_depth(axes[0, 0], result.z_gt, 'GT Depth')
        show_depth(axes[0, 1], result.z_refined, f'Refined (d125={result.metrics.get("ref_d125", -1):.1f}%)')

        # Error map
        if result.z_gt is not None and result.z_refined is not None:
            error = np.abs(result.z_refined - result.z_gt)
            error[~np.isfinite(error)] = 0
            axes[0, 2].imshow(error, vmin=0, vmax=5, cmap='hot')
            axes[0, 2].set_title('Absolute Error')
        else:
            axes[0, 2].set_title('Error: N/A')

        # DC scores text
        dc_text = f"DC Forward: {result.dc_score_forward:.3f}\n"
        dc_text += f"DC Backward: {result.dc_score_backward:.3f}\n"
        dc_text += f"Used Backward: {result.used_backward}"
        axes[0, 3].text(0.1, 0.5, dc_text, fontsize=12, family='monospace',
                       transform=axes[0, 3].transAxes, va='center')
        axes[0, 3].axis('off')
        axes[0, 3].set_title('DC Scores')

        # Row 2: Triangulated depths
        show_depth(axes[1, 0], result.z_tri_forward,
                  f'Tri Forward (d125={result.metrics.get("tri_fwd_d125", -1):.1f}%)')
        show_depth(axes[1, 1], result.z_tri_backward,
                  f'Tri Backward (d125={result.metrics.get("tri_bwd_d125", -1):.1f}%)')
        show_depth(axes[1, 2], result.z_warp_flow,
                  f'Warp Flow (d125={result.metrics.get("warp_flow_d125", -1):.1f}%)')
        show_depth(axes[1, 3], result.z_warp_pose,
                  f'Warp Pose (d125={result.metrics.get("warp_pose_d125", -1):.1f}%)')

        # Row 3: Prev depth, Variance
        show_depth(axes[2, 0], result.prev_depth_used, 'Prev Depth Used')

        if result.V_prior is not None:
            axes[2, 1].imshow(np.log10(result.V_prior + 1e-6), cmap='viridis')
            axes[2, 1].set_title('log10(V_prior)')
        else:
            axes[2, 1].set_title('V_prior: N/A')

        if result.V_post is not None:
            axes[2, 2].imshow(np.log10(result.V_post + 1e-6), cmap='viridis')
            axes[2, 2].set_title('log10(V_post)')
        else:
            axes[2, 2].set_title('V_post: N/A')

        # Metrics summary
        metrics_text = "Metrics:\n"
        for k, v in sorted(result.metrics.items()):
            metrics_text += f"  {k}: {v:.3f}\n"
        axes[2, 3].text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
                       transform=axes[2, 3].transAxes, va='center')
        axes[2, 3].axis('off')
        axes[2, 3].set_title(f'Frame {result.frame_idx}')

        plt.suptitle(f'PR-Depth Debug: Frame {result.frame_idx}', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved debug visualization to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return None

    def analyze_frames(self, frame_indices: List[int], output_dir: str = None) -> List[DebugResult]:
        """Analyze multiple frames and optionally save visualizations.

        Args:
            frame_indices: List of frame indices to analyze
            output_dir: Optional directory to save visualizations

        Returns:
            List of DebugResult objects
        """
        results = []

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        for idx in frame_indices:
            print(f"Analyzing frame {idx}...")
            result = self.analyze_frame(idx)
            results.append(result)

            if output_dir:
                save_path = Path(output_dir) / f'debug_frame_{idx:04d}.png'
                self.visualize(result, save_path=str(save_path), show=False)

        return results

    def print_summary(self, result: DebugResult):
        """Print summary of debug result."""
        print(f"\n{'='*60}")
        print(f"Frame {result.frame_idx} Debug Summary")
        print(f"{'='*60}")

        print(f"\nDC Scores:")
        print(f"  Forward:  {result.dc_score_forward:.4f}")
        print(f"  Backward: {result.dc_score_backward:.4f}")
        print(f"  Used Backward: {result.used_backward}")

        if result.t_forward is not None:
            print(f"\nPose (Forward):")
            print(f"  t = [{result.t_forward[0]:.4f}, {result.t_forward[1]:.4f}, {result.t_forward[2]:.4f}]")

        if result.t_backward is not None:
            print(f"\nPose (Backward):")
            print(f"  t = [{result.t_backward[0]:.4f}, {result.t_backward[1]:.4f}, {result.t_backward[2]:.4f}]")

        print(f"\nAccuracy Metrics:")
        for k, v in sorted(result.metrics.items()):
            print(f"  {k}: {v:.2f}")

        # Variance statistics
        if result.V_prior is not None:
            valid = np.isfinite(result.V_prior) & (result.V_prior > 0)
            print(f"\nV_prior stats:")
            print(f"  median: {np.median(result.V_prior[valid]):.4f}")
            print(f"  mean:   {np.mean(result.V_prior[valid]):.4f}")

        if result.V_post is not None:
            valid = np.isfinite(result.V_post) & (result.V_post > 0)
            print(f"\nV_post stats:")
            print(f"  median: {np.median(result.V_post[valid]):.4f}")
            print(f"  mean:   {np.mean(result.V_post[valid]):.4f}")

        print(f"{'='*60}\n")


if __name__ == '__main__':
    # Example usage
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from configs import get_dataset_paths
    from dataloader import KITTIEigenSplit

    # Load dataset
    paths = get_dataset_paths('kitti')
    dataset = KITTIEigenSplit(
        rgb_path=paths['rgb_path'],
        depth_path=paths['depth_path'],
        date='2011_10_03',
        drive='0027'
    )

    # Create debugger
    debugger = FrameDebugger(dataset)

    # Analyze specific frame
    result = debugger.analyze_frame(202)
    debugger.print_summary(result)
    debugger.visualize(result, save_path='/tmp/debug_frame_202.png')
