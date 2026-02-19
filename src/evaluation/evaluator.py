"""Main Evaluator class for running pose estimation evaluation."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Type, Callable
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

from .base import (
    BaseMethodRunner,
    BaseDataset,
    PoseResult,
    FrameMetrics,
    EvalSummary,
)
from .metrics import compute_depth_metrics, compute_pose_error, compute_ate, compute_rpe, compute_pose_auc, compute_tae


# Method display configuration
METHOD_COLORS = {
    'pr_depth': '#E74C3C',    # Red
    'madpose': '#3498DB',     # Blue
    'batrack': '#2ECC71',     # Green
    'monst3r': '#9B59B6',     # Purple
    'gt': '#2C3E50',          # Dark gray
}

METHOD_NAMES = {
    'pr_depth': 'PR-Depth',
    'madpose': 'MADPose',
    'batrack': 'BaTrack',
    'monst3r': 'MonST3R',
}


class Evaluator:
    """Main evaluation orchestrator.

    Handles the evaluation loop, metrics computation, and result aggregation.
    Supports real-time logging of results to CSV/NPZ and trajectory visualization.

    Example:
        >>> from src.evaluation import Evaluator
        >>> from src.evaluation.runners import PRDepthRunner
        >>> from dataloader import KITTIEigenSplit
        >>>
        >>> dataset = KITTIEigenSplit(...)
        >>> runner = PRDepthRunner(device='cuda')
        >>> evaluator = Evaluator(runner, dataset)
        >>> summary = evaluator.run(max_frames=100, output_dir='results/exp1')
    """

    def __init__(
        self,
        runner: BaseMethodRunner,
        dataset: BaseDataset,
        depth_estimator=None,
        results_base_dir: Optional[str] = None,
        use_gt_flow: bool = False,
        flow_estimator=None,
        odom_noise: float = 0.0,
        use_gps_baseline: bool = False,
        use_gt_pose_fallback: bool = False,
    ):
        """Initialize evaluator.

        Args:
            runner: Method runner instance
            dataset: Dataset instance
            depth_estimator: Optional depth estimator for methods that need it
            results_base_dir: Base directory for results (to load baselines for dashboard)
            use_gt_flow: Use GT optical flow from dataset (for ablation studies)
            flow_estimator: Optional external flow estimator (e.g., NeuFlow_v2)
            odom_noise: Relative noise level for odometry baseline (0.0 = no noise, 0.1 = ±10%)
            use_gps_baseline: Use GPS baseline instead of wheel odometry (Wheel dataset only)
            use_gt_pose_fallback: Use GT pose when rotation exceeds threshold
        """
        self.runner = runner
        self.dataset = dataset
        self.depth_estimator = depth_estimator
        self.results_base_dir = Path(results_base_dir) if results_base_dir else None
        self.use_gt_flow = use_gt_flow
        self.flow_estimator = flow_estimator
        self.odom_noise = odom_noise
        self.use_gps_baseline = use_gps_baseline
        self.use_gt_pose_fallback = use_gt_pose_fallback

        # Results storage
        self.frame_metrics: List[FrameMetrics] = []
        self.gt_positions: List[np.ndarray] = []
        self.est_positions: List[np.ndarray] = []
        self.gt_poses: List[tuple] = []  # [(R, t), ...]
        self.est_poses: List[tuple] = []
        self.debug_extras: List[dict] = []  # PR-Depth debug info

        # Trajectory accumulators (4x4 matrices)
        self._T_gt_global = np.eye(4)
        self._T_est_global = np.eye(4)

        # Output paths (set during run)
        self._output_dir: Optional[Path] = None
        self._csv_path: Optional[Path] = None
        self._npz_path: Optional[Path] = None
        self._depths_path: Optional[Path] = None
        self._traj_img_path: Optional[Path] = None
        self._dashboard_path: Optional[Path] = None
        self._dataset_id: Optional[str] = None

        # Depth storage for saving
        self._z_tri_list: List[np.ndarray] = []
        self._z_refined_list: List[np.ndarray] = []
        self._frame_indices: List[int] = []
        self._baselines_list: List[float] = []  # For TAE computation

        # Cached baseline trajectories for dashboard
        self._baseline_trajectories: Dict[str, Dict] = {}

    def run(
        self,
        start_frame: int = 0,
        max_frames: Optional[int] = None,
        verbose: bool = True,
        compute_depth_metrics_flag: bool = True,
        output_dir: Optional[str] = None,
        save_trajectory_img: bool = True,
        trajectory_update_interval: int = 10,
        dataset_id: Optional[str] = None,
        save_depths: bool = False,
    ) -> EvalSummary:
        """Run full evaluation with real-time logging.

        Args:
            start_frame: Frame index to start evaluation from (default: 0)
            max_frames: Maximum frames to process (None = all)
            verbose: Show progress bar
            compute_depth_metrics_flag: Whether to compute depth metrics
            output_dir: Directory for real-time output (CSV, NPZ, trajectory image)
            save_trajectory_img: Whether to save trajectory visualization
            trajectory_update_interval: Update trajectory image every N frames
            dataset_id: Dataset identifier for loading baselines (e.g., 'kitti_2011_10_03_0027')
            save_depths: Whether to save depth maps (z_tri, z_refined) to npz

        Returns:
            EvalSummary with aggregated results
        """
        # Initialize
        self._reset()
        end_frame = len(self.dataset) if max_frames is None else min(start_frame + max_frames, len(self.dataset))

        H, W = self.dataset.get_image_size()
        fx, fy, cx, cy = self.dataset.get_intrinsics()

        # Initialize runner
        self.runner.initialize(H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy)

        # Initialize positions
        self.gt_positions = [np.zeros(3)]
        self.est_positions = [np.zeros(3)]

        # Store dataset_id
        self._dataset_id = dataset_id

        # Load baseline trajectories for dashboard comparison
        if dataset_id and self.results_base_dir:
            self._baseline_trajectories = self._load_baseline_trajectories(dataset_id)
            if self._baseline_trajectories:
                print(f"Loaded {len(self._baseline_trajectories)} baseline trajectories for comparison")

        # Setup output directory for real-time logging
        if output_dir is not None:
            self._output_dir = Path(output_dir)
            self._output_dir.mkdir(parents=True, exist_ok=True)
            self._csv_path = self._output_dir / 'frame_metrics.csv'
            self._npz_path = self._output_dir / 'trajectory.npz'
            self._depths_path = self._output_dir / 'depths.npz' if save_depths else None
            self._traj_img_path = self._output_dir / 'trajectory.png' if save_trajectory_img else None
            self._dashboard_path = self._output_dir / 'dashboard.png' if save_trajectory_img else None

            # Initialize CSV with header
            self._init_csv()

        # Previous frame data
        prev_img = None
        prev_depth = None
        prev_gt_depth = None

        # Get previous frame data (no warm-up, just load prev frame for optical flow)
        if start_frame > 0:
            data = self.dataset.get(start_frame - 1)
            if data is not None:
                prev_img = data['image_og']
                prev_gt_depth = data.get('depth_og')
                if self.depth_estimator is not None:
                    prev_depth = self.depth_estimator.infer(prev_img)

        start_time = time.time()
        total_frames = end_frame - start_frame
        iterator = range(start_frame, end_frame)
        if verbose:
            iterator = tqdm(iterator, desc=f"Evaluating {self.runner.name}")

        for i in iterator:
            frame_metrics = self._process_frame(
                i, prev_img, prev_depth, compute_depth_metrics_flag,
                prev_gt_depth=prev_gt_depth, intrinsics=(fx, fy, cx, cy)
            )
            if frame_metrics is not None:
                self.frame_metrics.append(frame_metrics)

                # Real-time logging
                if self._output_dir is not None:
                    self._append_csv(frame_metrics)
                    self._save_trajectory_npz()

                    # Update trajectory image and dashboard periodically
                    if save_trajectory_img and (len(self.frame_metrics) % trajectory_update_interval == 0):
                        self._save_trajectory_image(i, end_frame)
                        self._save_dashboard(i, end_frame)

            # Update previous frame
            data = self.dataset.get(i)
            if data is not None:
                prev_img = data['image_og']
                prev_gt_depth = data.get('depth_og')
                if self.depth_estimator is not None:
                    prev_depth = self.depth_estimator.infer(prev_img)

        elapsed = time.time() - start_time

        # Finalize runner (for SLAM methods)
        final_result = self.runner.finalize()

        # Final trajectory image and dashboard
        if self._output_dir is not None and save_trajectory_img:
            self._save_trajectory_image(end_frame, end_frame, final=True)
            self._save_dashboard(end_frame, end_frame, final=True)

        # Compute summary
        summary = self._compute_summary(total_frames, elapsed)

        # Save debug extras to JSON
        if self._output_dir is not None and self.debug_extras:
            import json
            debug_path = self._output_dir / 'debug_extras.json'
            with open(debug_path, 'w') as f:
                json.dump(self.debug_extras, f, indent=2)

        # Save depths if requested
        if self._depths_path is not None and self._z_tri_list:
            self._save_depths()

        return summary

    def _init_csv(self):
        """Initialize CSV file with header."""
        if self._csv_path is None:
            return

        # Write header based on FrameMetrics fields
        # tri_* = triangulation depth, ref_* = reference depth
        # wp_* = 3D warped prior, wf_* = flow warped prior
        # iter0_* = iteration 0 (forward), iter1_* = iteration 1 (backward)
        # (MADPose: ref = UniDepth, PR-Depth: ref = refined depth)
        header = [
            'frame_idx', 'baseline', 'success', 'num_inliers',
            'rot_error', 'trans_error',
            'tri_MAE', 'tri_RMSE', 'tri_AbsRel', 'tri_d105', 'tri_d115', 'tri_d125',
            'ref_MAE', 'ref_RMSE', 'ref_AbsRel', 'ref_d105', 'ref_d115', 'ref_d125',
            'wp_MAE', 'wp_RMSE', 'wp_AbsRel', 'wp_d105', 'wp_d115', 'wp_d125',
            'wf_MAE', 'wf_RMSE', 'wf_AbsRel', 'wf_d105', 'wf_d115', 'wf_d125',
            # Per-iteration metrics
            'iter0_tri_AbsRel', 'iter0_tri_d125', 'iter0_fused_sparse_AbsRel', 'iter0_fused_sparse_d125',
            'iter0_ref_AbsRel', 'iter0_ref_d125',
            'iter1_tri_AbsRel', 'iter1_tri_d125', 'iter1_fused_sparse_AbsRel', 'iter1_fused_sparse_d125',
            'iter1_ref_AbsRel', 'iter1_ref_d125',
        ]
        with open(self._csv_path, 'w') as f:
            f.write(','.join(header) + '\n')

    def _append_csv(self, metrics: FrameMetrics):
        """Append single frame metrics to CSV."""
        if self._csv_path is None:
            return

        row = metrics.to_dict()
        # Ensure consistent column order
        columns = [
            'frame_idx', 'baseline', 'success', 'num_inliers',
            'rot_error', 'trans_error',
            'tri_MAE', 'tri_RMSE', 'tri_AbsRel', 'tri_d105', 'tri_d115', 'tri_d125',
            'ref_MAE', 'ref_RMSE', 'ref_AbsRel', 'ref_d105', 'ref_d115', 'ref_d125',
            'wp_MAE', 'wp_RMSE', 'wp_AbsRel', 'wp_d105', 'wp_d115', 'wp_d125',
            'wf_MAE', 'wf_RMSE', 'wf_AbsRel', 'wf_d105', 'wf_d115', 'wf_d125',
            # Per-iteration metrics
            'iter0_tri_AbsRel', 'iter0_tri_d125', 'iter0_fused_sparse_AbsRel', 'iter0_fused_sparse_d125',
            'iter0_ref_AbsRel', 'iter0_ref_d125',
            'iter1_tri_AbsRel', 'iter1_tri_d125', 'iter1_fused_sparse_AbsRel', 'iter1_fused_sparse_d125',
            'iter1_ref_AbsRel', 'iter1_ref_d125',
        ]
        values = [str(row.get(col, '')) for col in columns]

        with open(self._csv_path, 'a') as f:
            f.write(','.join(values) + '\n')

    def _save_trajectory_npz(self):
        """Save current trajectory to NPZ."""
        if self._npz_path is None:
            return

        # Extract R, t from poses
        gt_Rs = np.array([R for R, t in self.gt_poses]) if self.gt_poses else np.array([])
        gt_ts = np.array([t for R, t in self.gt_poses]) if self.gt_poses else np.array([])
        est_Rs = np.array([R for R, t in self.est_poses]) if self.est_poses else np.array([])
        est_ts = np.array([t for R, t in self.est_poses]) if self.est_poses else np.array([])

        np.savez(
            self._npz_path,
            gt_positions=np.array(self.gt_positions),
            est_positions=np.array(self.est_positions),
            gt_Rs=gt_Rs,
            gt_ts=gt_ts,
            est_Rs=est_Rs,
            est_ts=est_ts,
            baselines=np.array(self._baselines_list) if self._baselines_list else np.array([]),
        )

    def _save_depths(self):
        """Save depth maps (z_tri, z_refined) to NPZ."""
        if self._depths_path is None or not self._z_tri_list:
            return

        # Stack depths - handle varying shapes by checking first valid depth
        z_tri_valid = [z for z in self._z_tri_list if z is not None and len(z) > 0]
        z_refined_valid = [z for z in self._z_refined_list if z is not None and len(z) > 0]

        data = {
            'frame_indices': np.array(self._frame_indices),
        }

        if z_tri_valid:
            # All depths should have same shape
            data['z_tri'] = np.stack(z_tri_valid, axis=0)
            print(f"Saved z_tri: {data['z_tri'].shape}")

        if z_refined_valid:
            data['z_refined'] = np.stack(z_refined_valid, axis=0)
            print(f"Saved z_refined: {data['z_refined'].shape}")

        np.savez_compressed(self._depths_path, **data)
        print(f"Saved depths to: {self._depths_path}")

    def _save_trajectory_image(self, current_frame: int, total_frames: int, final: bool = False):
        """Save trajectory visualization image."""
        if self._traj_img_path is None or len(self.gt_positions) < 2:
            return

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        gt_arr = np.array(self.gt_positions)
        est_arr = np.array(self.est_positions)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Bird's eye view (X-Z)
        ax = axes[0]
        ax.plot(gt_arr[:, 0], gt_arr[:, 2], 'b-', label='Ground Truth', linewidth=2, alpha=0.8)
        ax.plot(est_arr[:, 0], est_arr[:, 2], 'r-', label=self.runner.name, linewidth=2, alpha=0.8)
        ax.scatter([gt_arr[0, 0]], [gt_arr[0, 2]], c='green', s=100, marker='o', zorder=5, label='Start')
        ax.scatter([gt_arr[-1, 0]], [gt_arr[-1, 2]], c='blue', s=80, marker='x', zorder=5)
        ax.scatter([est_arr[-1, 0]], [est_arr[-1, 2]], c='red', s=80, marker='x', zorder=5)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title("Bird's Eye View (X-Z)")
        ax.legend(loc='best')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)

        # Side view (Z-Y)
        ax = axes[1]
        ax.plot(gt_arr[:, 2], gt_arr[:, 1], 'b-', label='Ground Truth', linewidth=2, alpha=0.8)
        ax.plot(est_arr[:, 2], est_arr[:, 1], 'r-', label=self.runner.name, linewidth=2, alpha=0.8)
        ax.set_xlabel('Z (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Side View (Z-Y)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Title with progress
        status = "FINAL" if final else f"Frame {current_frame}/{total_frames}"
        plt.suptitle(f'{self.runner.name} Trajectory - {status}', fontsize=13, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self._traj_img_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

    def _process_frame(
        self,
        idx: int,
        prev_img: Optional[np.ndarray],
        prev_depth: Optional[np.ndarray],
        compute_depth: bool,
        prev_gt_depth: Optional[np.ndarray] = None,
        intrinsics: Optional[tuple] = None,
    ) -> Optional[FrameMetrics]:
        """Process a single frame."""
        data = self.dataset.get(idx)
        if data is None:
            return None

        img = data['image_og']
        gt_depth = data.get('depth_og')

        # Compute depth if estimator available
        curr_depth = None
        if self.depth_estimator is not None:
            curr_depth = self.depth_estimator.infer(img)

        # Skip first frame (no previous)
        if prev_img is None:
            return None

        # Get GT
        if self.use_gps_baseline and hasattr(self.dataset, 'get_gps_baseline'):
            baseline = self.dataset.get_gps_baseline(idx)
        else:
            baseline = self.dataset.get_baseline(idx)
        R_gt, t_gt = self.dataset.get_relative_pose(idx)

        # Add noise to odometry baseline if specified
        if self.odom_noise > 0:
            noise_factor = 1.0 + np.random.uniform(-self.odom_noise, self.odom_noise)
            baseline = baseline * noise_factor

        # Get external flow (GT flow or from flow estimator)
        external_flow = None
        if self.use_gt_flow:
            gt_flow = data.get('flow')
            if gt_flow is not None:
                external_flow = gt_flow
        elif self.flow_estimator is not None:
            # Use external flow estimator (e.g., NeuFlow_v2)
            # Note: prev_img is from frame idx-1, img is from frame idx
            # This matches the GT flow convention (flow from idx-1 to idx)
            external_flow = self.flow_estimator.estimate(prev_img, img)

        # Run method
        # Pass GT pose if GT pose fallback mode is enabled OR use_gt_R mode is enabled
        # use_gt_R needs gt_R for translation-only estimation
        needs_gt_R = self.use_gt_pose_fallback or getattr(self.runner, 'use_gt_R', False)
        gt_R_for_runner = R_gt if needs_gt_R else None
        gt_t_for_runner = t_gt if self.use_gt_pose_fallback else None

        result = self.runner.process_frame(
            img_curr=img,
            img_prev=prev_img,
            depth_curr=curr_depth,
            depth_prev=prev_depth,
            baseline=baseline,
            external_flow=external_flow,
            gt_R=gt_R_for_runner,
            gt_t=gt_t_for_runner,
        )

        # Create frame metrics
        metrics = FrameMetrics(
            frame_idx=idx,
            baseline=baseline,
            success=result.success,
            num_inliers=result.num_inliers,
        )

        # Pose errors - compute even if success=False (we have R, t estimates)
        pose_err = compute_pose_error(result.R, result.t, R_gt, t_gt)
        metrics.rot_error = pose_err['rot_error']
        metrics.trans_error = pose_err['trans_error']

        # For metric methods, check if t magnitude is reasonable
        # If t is too large (> 3x GT baseline), likely bad estimation - use R only
        t_est = result.t
        if self.runner.is_metric:
            t_norm = np.linalg.norm(t_est)
            if t_norm > baseline * 3.0:
                # Translation too large - use R only (t = 0)
                t_est = np.zeros(3)
                metrics.success = False

        # Update GT trajectory always
        T_gt = np.eye(4)
        T_gt[:3, :3] = R_gt
        T_gt[:3, 3] = t_gt
        self._T_gt_global = self._T_gt_global @ np.linalg.inv(T_gt)
        self.gt_positions.append(self._T_gt_global[:3, 3].copy())
        self.gt_poses.append((R_gt, t_gt))

        # Update estimated trajectory
        if metrics.success:
            # Success: accumulate estimated pose
            T_est = np.eye(4)
            T_est[:3, :3] = result.R
            T_est[:3, 3] = t_est
            self._T_est_global = self._T_est_global @ np.linalg.inv(T_est)
        # else: keep _T_est_global unchanged (skip bad frame)

        # Always append current position (same length as GT)
        self.est_positions.append(self._T_est_global[:3, 3].copy())
        self.est_poses.append((result.R, t_est))

        # Store debug extras for PR-Depth (excluding large arrays)
        if result.extra:
            debug_info = {
                'frame_idx': idx,
                'dc_score_forward': result.extra.get('dc_score_forward', -1),
                'dc_score_backward': result.extra.get('dc_score_backward', -1),
                'depth_consistency_score': result.extra.get('depth_consistency_score', -1),
                'depth_consistency_rejected': result.extra.get('depth_consistency_rejected', False),
                'used_backward': result.extra.get('used_backward', False),
                'metric_scale_forward': result.extra.get('metric_scale_forward', -1),
                'metric_scale_backward': result.extra.get('metric_scale_backward', -1),
                'tri_disabled': result.extra.get('tri_disabled', False),
                'baseline_used': result.extra.get('baseline', baseline),
                'dc_score_tri': result.extra.get('dc_score_tri', -1),
                'dc_score_warp': result.extra.get('dc_score_warp', -1),
                'dc_tri_pass': result.extra.get('dc_tri_pass', False),
                'dc_warp_pass': result.extra.get('dc_warp_pass', False),
                'fusion_case': result.extra.get('fusion_case', 0),
            }
            self.debug_extras.append(debug_info)

        # Depth metrics - compute regardless of success (pose may fail but depth still valid)
        if compute_depth and gt_depth is not None:
            self._compute_frame_depth_metrics(metrics, result, gt_depth, curr_depth)

        # GT warp error: warp prev GT depth using GT pose, compare with current GT depth
        # High error = GT pose is unreliable for this frame pair
        if prev_gt_depth is not None and gt_depth is not None and intrinsics is not None:
            gt_warp_err = self._compute_gt_warp_error(
                prev_gt_depth, gt_depth, R_gt, t_gt, intrinsics
            )
            if gt_warp_err is not None:
                metrics.gt_warp_absrel = gt_warp_err

        # Always store baselines for TAE computation
        self._baselines_list.append(baseline)

        # Store depths for saving (only when save_depths=True)
        if self._depths_path is not None:
            z_tri = result.extra.get('z_tri')
            z_refined = result.extra.get('z_refined')
            self._frame_indices.append(idx)
            self._z_tri_list.append(z_tri if z_tri is not None else np.array([]))
            self._z_refined_list.append(z_refined if z_refined is not None else np.array([]))

        return metrics

    def _update_trajectory(
        self,
        R_est: np.ndarray,
        t_est: np.ndarray,
        R_gt: np.ndarray,
        t_gt: np.ndarray,
    ):
        """Update trajectory accumulators."""
        # GT trajectory - build 4x4 matrix and multiply
        T_gt = np.eye(4)
        T_gt[:3, :3] = R_gt
        T_gt[:3, 3] = t_gt
        self._T_gt_global = self._T_gt_global @ np.linalg.inv(T_gt)
        self.gt_positions.append(self._T_gt_global[:3, 3].copy())

        # Estimated trajectory - invert pose (T_1to0 -> T_0to1)
        # Methods return T_1to0 convention, need T_0to1 for trajectory accumulation
        T_est = np.eye(4)
        T_est[:3, :3] = R_est
        T_est[:3, 3] = t_est
        self._T_est_global = self._T_est_global @ np.linalg.inv(T_est)
        self.est_positions.append(self._T_est_global[:3, 3].copy())

    @staticmethod
    def _compute_gt_warp_error(
        prev_gt_depth: np.ndarray,
        curr_gt_depth: np.ndarray,
        R_gt: np.ndarray,
        t_gt: np.ndarray,
        intrinsics: tuple,
    ) -> Optional[float]:
        """Warp prev GT depth to current frame using GT pose, compute AbsRel vs current GT depth.

        Convention: p_curr = R_gt @ p_prev + t_gt

        Returns AbsRel error, or None if insufficient valid pixels.
        """
        fx, fy, cx, cy = intrinsics
        H, W = prev_gt_depth.shape[:2]

        # Valid prev pixels
        valid_prev = (prev_gt_depth > 0.1) & np.isfinite(prev_gt_depth)
        if valid_prev.sum() < 100:
            return None

        # Pixel coordinates
        yy, xx = np.where(valid_prev)
        z_prev = prev_gt_depth[yy, xx].astype(np.float64)

        # Backproject to 3D
        X = (xx - cx) * z_prev / fx
        Y = (yy - cy) * z_prev / fy
        pts3d = np.stack([X, Y, z_prev], axis=1)  # (N, 3)

        # Transform: p_curr = R @ p_prev + t
        pts_curr = (R_gt.astype(np.float64) @ pts3d.T).T + t_gt.astype(np.float64)

        X_new = pts_curr[:, 0]
        Y_new = pts_curr[:, 1]
        Z_new = pts_curr[:, 2]

        # Project to current image
        valid_z = Z_new > 0.1
        u_new = np.round((X_new * fx) / (Z_new + 1e-8) + cx).astype(int)
        v_new = np.round((Y_new * fy) / (Z_new + 1e-8) + cy).astype(int)

        # Bounds check
        in_bounds = valid_z & (u_new >= 0) & (u_new < W) & (v_new >= 0) & (v_new < H)

        if in_bounds.sum() < 100:
            return None

        # Z-buffer: keep closest depth per pixel
        z_warped = np.full((H, W), np.inf, dtype=np.float64)
        u_valid = u_new[in_bounds]
        v_valid = v_new[in_bounds]
        z_valid = Z_new[in_bounds]

        # Sort by depth descending so closer points overwrite farther ones
        order = np.argsort(-z_valid)
        z_warped[v_valid[order], u_valid[order]] = z_valid[order]

        # Compare with current GT
        compare = (z_warped < np.inf) & (curr_gt_depth > 0.1) & np.isfinite(curr_gt_depth)
        n_compare = compare.sum()
        if n_compare < 100:
            return None

        abs_rel = float(np.mean(
            np.abs(z_warped[compare] - curr_gt_depth[compare].astype(np.float64))
            / curr_gt_depth[compare].astype(np.float64)
        ))

        return abs_rel

    def _compute_frame_depth_metrics(
        self,
        metrics: FrameMetrics,
        result: PoseResult,
        gt_depth: np.ndarray,
        inv_depth: Optional[np.ndarray],
    ):
        """Compute depth metrics for a frame."""
        # Triangulation depth
        z_tri = result.extra.get('z_tri')
        if z_tri is not None:
            tri_metrics = compute_depth_metrics(z_tri, gt_depth)
            if tri_metrics:
                metrics.tri_MAE = tri_metrics['MAE']
                metrics.tri_RMSE = tri_metrics['RMSE']
                metrics.tri_AbsRel = tri_metrics['AbsRel']
                metrics.tri_d105 = tri_metrics['d105']
                metrics.tri_d115 = tri_metrics['d115']
                metrics.tri_d125 = tri_metrics['d125']

        # Refined depth (PR-Depth)
        z_refined = result.extra.get('z_refined')
        if z_refined is not None:
            ref_metrics = compute_depth_metrics(z_refined, gt_depth)
            if ref_metrics:
                metrics.ref_MAE = ref_metrics['MAE']
                metrics.ref_RMSE = ref_metrics['RMSE']
                metrics.ref_AbsRel = ref_metrics['AbsRel']
                metrics.ref_d105 = ref_metrics['d105']
                metrics.ref_d115 = ref_metrics['d115']
                metrics.ref_d125 = ref_metrics['d125']

        # UniDepth metric depth (MADPose) - stored as ref_* for unified headers
        z_unidepth = result.extra.get('z_unidepth')
        if z_unidepth is not None and metrics.ref_MAE is None:
            # Only use UniDepth as ref if z_refined was not already set
            ref_metrics = compute_depth_metrics(z_unidepth, gt_depth)
            if ref_metrics:
                metrics.ref_MAE = ref_metrics['MAE']
                metrics.ref_RMSE = ref_metrics['RMSE']
                metrics.ref_AbsRel = ref_metrics['AbsRel']
                metrics.ref_d105 = ref_metrics['d105']
                metrics.ref_d115 = ref_metrics['d115']
                metrics.ref_d125 = ref_metrics['d125']

        # 3D warped prior (z_warp_pose)
        z_warp_pose = result.extra.get('z_warp_pose')
        if z_warp_pose is not None:
            wp_metrics = compute_depth_metrics(z_warp_pose, gt_depth)
            if wp_metrics:
                metrics.wp_MAE = wp_metrics['MAE']
                metrics.wp_RMSE = wp_metrics['RMSE']
                metrics.wp_AbsRel = wp_metrics['AbsRel']
                metrics.wp_d105 = wp_metrics['d105']
                metrics.wp_d115 = wp_metrics['d115']
                metrics.wp_d125 = wp_metrics['d125']

        # Flow warped prior (z_warp_flow)
        z_warp_flow = result.extra.get('z_warp_flow')
        if z_warp_flow is not None:
            wf_metrics = compute_depth_metrics(z_warp_flow, gt_depth)
            if wf_metrics:
                metrics.wf_MAE = wf_metrics['MAE']
                metrics.wf_RMSE = wf_metrics['RMSE']
                metrics.wf_AbsRel = wf_metrics['AbsRel']
                metrics.wf_d105 = wf_metrics['d105']
                metrics.wf_d115 = wf_metrics['d115']
                metrics.wf_d125 = wf_metrics['d125']

        # Relative depth (scaled monocular)
        if inv_depth is not None:
            mask = (gt_depth > 0) & (gt_depth < 80) & (inv_depth > 0.01)
            if mask.sum() > 100:
                scale = np.median(gt_depth[mask] * inv_depth[mask])
                depth_rel = scale / (inv_depth + 1e-8)
                rel_metrics = compute_depth_metrics(depth_rel, gt_depth)
                if rel_metrics:
                    metrics.rel_MAE = rel_metrics['MAE']
                    metrics.rel_RMSE = rel_metrics['RMSE']
                    metrics.rel_AbsRel = rel_metrics['AbsRel']
                    metrics.rel_d105 = rel_metrics['d105']
                    metrics.rel_d115 = rel_metrics['d115']
                    metrics.rel_d125 = rel_metrics['d125']

        # Per-iteration depth metrics (PR-Depth only)
        iteration_info = result.extra.get('iteration_info')
        if iteration_info is not None and len(iteration_info) > 0:
            for iter_data in iteration_info:
                iter_idx = iter_data.get('iter', 0)
                prefix = f'iter{iter_idx}'

                # Triangulation depth for this iteration
                z_tri_iter = iter_data.get('z_tri')
                if z_tri_iter is not None and len(z_tri_iter) > 0:
                    iter_tri_metrics = compute_depth_metrics(z_tri_iter, gt_depth)
                    if iter_tri_metrics:
                        setattr(metrics, f'{prefix}_tri_AbsRel', iter_tri_metrics['AbsRel'])
                        setattr(metrics, f'{prefix}_tri_d125', iter_tri_metrics['d125'])

                # Sparse fusion depth for this iteration (before solve_metric_from_rel)
                z_fused_sparse_iter = iter_data.get('z_fused_sparse')
                if z_fused_sparse_iter is not None and len(z_fused_sparse_iter) > 0:
                    iter_fused_sparse_metrics = compute_depth_metrics(z_fused_sparse_iter, gt_depth)
                    if iter_fused_sparse_metrics:
                        setattr(metrics, f'{prefix}_fused_sparse_AbsRel', iter_fused_sparse_metrics['AbsRel'])
                        setattr(metrics, f'{prefix}_fused_sparse_d125', iter_fused_sparse_metrics['d125'])

                # Refined depth for this iteration (after solve_metric_from_rel)
                z_ref_iter = iter_data.get('z_refined')
                if z_ref_iter is not None and len(z_ref_iter) > 0:
                    iter_ref_metrics = compute_depth_metrics(z_ref_iter, gt_depth)
                    if iter_ref_metrics:
                        setattr(metrics, f'{prefix}_ref_AbsRel', iter_ref_metrics['AbsRel'])
                        setattr(metrics, f'{prefix}_ref_d125', iter_ref_metrics['d125'])

    def _compute_summary(self, total_frames: int, elapsed: float) -> EvalSummary:
        """Compute summary statistics."""
        df = self.to_dataframe()

        summary = EvalSummary(
            method_name=self.runner.name,
            dataset_name=getattr(self.dataset, 'dataset_name', 'unknown'),
            total_frames=total_frames,
            success_frames=int(df['success'].sum()) if 'success' in df.columns else 0,
            elapsed_time=elapsed,
            fps=total_frames / elapsed if elapsed > 0 else 0,
        )

        # Pose metrics
        if 'rot_error' in df.columns:
            valid = df['rot_error'].dropna()
            if len(valid) > 0:
                summary.rot_error_mean = float(valid.mean())
                summary.rot_error_median = float(valid.median())

        if 'trans_error' in df.columns:
            valid = df['trans_error'].dropna()
            if len(valid) > 0:
                summary.trans_error_mean = float(valid.mean())
                summary.trans_error_median = float(valid.median())

        # ATE & RPE (computed on full trajectory after accumulation)
        if len(self.gt_positions) > 1 and len(self.est_positions) > 1:
            gt_pos = np.array(self.gt_positions)
            est_pos = np.array(self.est_positions)

            # Extract rotations from accumulated poses
            gt_rots = [np.eye(3)] + [R for R, t in self.gt_poses]
            est_rots = [np.eye(3)] + [R for R, t in self.est_poses]

            # ATE (with Sim3 alignment)
            ate_metrics = compute_ate(gt_pos, est_pos, align=True)
            if ate_metrics:
                summary.ATE_RMSE = ate_metrics['ATE_RMSE']
                summary.ATE_mean = ate_metrics['ATE_mean']
                summary.final_drift = ate_metrics['final_drift']
                summary.trajectory_length = ate_metrics['trajectory_length']

            # RPE (translation and rotation)
            rpe_metrics = compute_rpe(gt_pos, est_pos, gt_rots, est_rots, delta=1)
            if rpe_metrics:
                summary.RPE_trans_RMSE = rpe_metrics.get('RPE_trans_RMSE')
                summary.RPE_trans_mean = rpe_metrics.get('RPE_trans_mean')
                summary.RPE_rot_RMSE = rpe_metrics.get('RPE_rot_RMSE')
                summary.RPE_rot_mean = rpe_metrics.get('RPE_rot_mean')

        # Pose AUC (MADPose style)
        if 'rot_error' in df.columns and 'trans_error' in df.columns:
            rot_errors = df['rot_error'].dropna().tolist()
            trans_errors = df['trans_error'].dropna().tolist()
            if rot_errors and trans_errors:
                auc_metrics = compute_pose_auc(rot_errors, trans_errors, thresholds=[5, 10, 20])
                summary.AUC_5 = auc_metrics.get('AUC@5')
                summary.AUC_10 = auc_metrics.get('AUC@10')
                summary.AUC_20 = auc_metrics.get('AUC@20')

        # Depth metrics
        for prefix, attr_prefix in [('tri', 'tri'), ('ref', 'ref')]:
            col = f'{prefix}_d125'
            if col in df.columns:
                valid = df[col].dropna()
                if len(valid) > 0:
                    setattr(summary, f'{attr_prefix}_d125_mean', float(valid.mean()))

            col = f'{prefix}_MAE'
            if col in df.columns:
                valid = df[col].dropna()
                if len(valid) > 0:
                    setattr(summary, f'{attr_prefix}_MAE_mean', float(valid.mean()))

        # TAE (Temporal Alignment Error) - Video Depth Anything metric
        if len(self._z_refined_list) > 1 and len(self.est_poses) > 0:
            # Filter valid depth maps
            valid_depths = [z for z in self._z_refined_list if z is not None and z.size > 0]
            if len(valid_depths) > 1 and len(self._baselines_list) > 0:
                # Get camera intrinsics
                fx, fy, cx, cy = self.dataset.get_intrinsics()

                # Use estimated poses for PR-Depth (pose estimation method)
                # For depth-only methods, this would use GT poses
                tae_metrics = compute_tae(
                    depths=valid_depths,
                    poses=self.est_poses,
                    baselines=self._baselines_list,
                    fx=fx, fy=fy, cx=cx, cy=cy
                )
                if tae_metrics:
                    summary.TAE = tae_metrics.get('TAE')
                    summary.TAE_forward = tae_metrics.get('TAE_forward')
                    summary.TAE_backward = tae_metrics.get('TAE_backward')

        return summary

    def to_dataframe(self) -> pd.DataFrame:
        """Convert frame metrics to DataFrame."""
        return pd.DataFrame([m.to_dict() for m in self.frame_metrics])

    def _reset(self):
        """Reset all state."""
        self.frame_metrics = []
        self.gt_positions = []
        self.est_positions = []
        self.gt_poses = []
        self.est_poses = []
        self.debug_extras = []
        self._T_gt_global = np.eye(4)
        self._T_est_global = np.eye(4)
        self._output_dir = None
        self._csv_path = None
        self._npz_path = None
        self._depths_path = None
        self._traj_img_path = None
        self._dashboard_path = None
        self._dataset_id = None
        self._baseline_trajectories = {}
        self._z_tri_list = []
        self._z_refined_list = []
        self._frame_indices = []
        self._baselines_list = []
        self.runner.reset()

    def get_trajectories(self) -> Dict[str, np.ndarray]:
        """Get GT and estimated trajectories."""
        return {
            'gt_positions': np.array(self.gt_positions),
            'est_positions': np.array(self.est_positions),
        }

    def get_poses(self) -> Dict[str, List[tuple]]:
        """Get GT and estimated relative poses."""
        return {
            'gt_poses': self.gt_poses,
            'est_poses': self.est_poses,
        }

    def _load_baseline_trajectories(self, dataset_id: str) -> Dict[str, Dict]:
        """Load all existing baseline trajectories for comparison.

        Args:
            dataset_id: Dataset identifier (e.g., 'kitti_2011_10_03_0027')

        Returns:
            Dict of method_name -> {'positions': array, 'metrics': dict}
        """
        baselines = {}

        if self.results_base_dir is None:
            return baselines

        # Load from baselines folder
        baselines_dir = self.results_base_dir / dataset_id / 'baselines'
        if baselines_dir.exists():
            for method_dir in baselines_dir.iterdir():
                if method_dir.is_dir():
                    traj_file = method_dir / 'trajectory.npz'
                    summary_file = method_dir / 'summary.json'

                    if traj_file.exists():
                        try:
                            data = np.load(traj_file)
                            baselines[method_dir.name] = {
                                'positions': data['est_positions'],
                                'metrics': {}
                            }

                            # Load metrics if available
                            if summary_file.exists():
                                with open(summary_file) as f:
                                    summary = json.load(f)
                                baselines[method_dir.name]['metrics'] = {
                                    'ATE_RMSE': summary.get('ATE_RMSE'),
                                    'rot_error_mean': summary.get('rot_error_mean'),
                                    'fps': summary.get('fps'),
                                }
                        except Exception as e:
                            print(f"Warning: Could not load {method_dir.name}: {e}")

        # Load from pr_depth folder (previous experiments)
        pr_depth_dir = self.results_base_dir / dataset_id / 'pr_depth'
        if pr_depth_dir.exists():
            for exp_dir in pr_depth_dir.iterdir():
                if exp_dir.is_dir() and exp_dir.name != 'latest' and not exp_dir.is_symlink():
                    traj_file = exp_dir / 'trajectory.npz'
                    summary_file = exp_dir / 'summary.json'

                    if traj_file.exists():
                        try:
                            data = np.load(traj_file)
                            # Use experiment name as key with pr_depth prefix
                            key = f"pr_depth:{exp_dir.name}"
                            baselines[key] = {
                                'positions': data['est_positions'],
                                'metrics': {}
                            }

                            if summary_file.exists():
                                with open(summary_file) as f:
                                    summary = json.load(f)
                                baselines[key]['metrics'] = {
                                    'ATE_RMSE': summary.get('ATE_RMSE'),
                                    'rot_error_mean': summary.get('rot_error_mean'),
                                    'fps': summary.get('fps'),
                                }
                        except Exception:
                            pass

        return baselines

    def _save_dashboard(self, current_frame: int, total_frames: int, final: bool = False):
        """Save dashboard image with all methods compared.

        Layout 2: Vertical Split
        - Left: Main XZ trajectory (tall)
        - Right top: Side view (Z-Y)
        - Right middle: Depth accuracy plot
        - Right bottom: Metrics table + Progress
        """
        if self._dashboard_path is None or len(self.gt_positions) < 2:
            return

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        gt_arr = np.array(self.gt_positions)
        est_arr = np.array(self.est_positions)

        # Create figure with fixed size - use tight_layout to prevent expansion
        fig = plt.figure(figsize=(20, 12), constrained_layout=False)
        gs = GridSpec(4, 2, figure=fig, width_ratios=[1.2, 1], height_ratios=[1, 1, 1, 0.5],
                      hspace=0.35, wspace=0.25)

        color_current = METHOD_COLORS.get('pr_depth', '#E74C3C')

        # === Main XZ trajectory (left, spans all rows except progress) ===
        ax_main = fig.add_subplot(gs[0:3, 0])

        # Plot GT
        ax_main.plot(gt_arr[:, 0], gt_arr[:, 2], 'k-',
                     label='Ground Truth', linewidth=2.5, alpha=0.9)

        # Plot current method (PR-Depth)
        ax_main.plot(est_arr[:, 0], est_arr[:, 2], '-',
                     color=color_current, label=f'{self.runner.name} (current)',
                     linewidth=2.5, alpha=0.9)

        # Plot baselines
        for method, data in self._baseline_trajectories.items():
            pos = data['positions']
            if len(pos) > 0:
                base_method = method.split(':')[0] if ':' in method else method
                color = METHOD_COLORS.get(base_method, '#95A5A6')
                display_name = METHOD_NAMES.get(base_method, method)

                if method.startswith('pr_depth:'):
                    exp_name = method.split(':')[1]
                    display_name = f"PR-Depth ({exp_name})"
                    color = '#F5B7B1'
                    alpha = 0.5
                    linewidth = 1.5
                else:
                    alpha = 0.8
                    linewidth = 2

                ax_main.plot(pos[:, 0], pos[:, 2], '--',
                            color=color, label=display_name,
                            linewidth=linewidth, alpha=alpha)

        # Start/end markers
        ax_main.scatter([gt_arr[0, 0]], [gt_arr[0, 2]],
                        c='green', s=200, marker='o', zorder=10, label='Start')
        ax_main.scatter([gt_arr[-1, 0]], [gt_arr[-1, 2]],
                        c='black', s=150, marker='x', zorder=10, linewidths=3)
        ax_main.scatter([est_arr[-1, 0]], [est_arr[-1, 2]],
                        c=color_current, s=150, marker='x', zorder=10, linewidths=3)

        ax_main.set_xlabel('X (m)', fontsize=12)
        ax_main.set_ylabel('Z (m)', fontsize=12)
        ax_main.set_title("Bird's Eye View (X-Z)", fontsize=14, fontweight='bold')
        ax_main.legend(loc='upper left', fontsize=9, ncol=2)
        ax_main.set_aspect('equal', adjustable='datalim')
        # Don't use equal aspect - let matplotlib auto-scale to fit the subplot area
        # This prevents the figure from expanding when trajectory is very long in one direction
        ax_main.grid(True, alpha=0.3)

        # === Side view (right top) ===
        ax_side = fig.add_subplot(gs[0, 1])
        ax_side.plot(gt_arr[:, 2], gt_arr[:, 1], 'k-', label='GT', linewidth=2)
        ax_side.plot(est_arr[:, 2], est_arr[:, 1], '-', color=color_current,
                     label=self.runner.name, linewidth=2)

        for method, data in self._baseline_trajectories.items():
            if not method.startswith('pr_depth:'):
                pos = data['positions']
                if len(pos) > 0:
                    base_method = method.split(':')[0] if ':' in method else method
                    color = METHOD_COLORS.get(base_method, '#95A5A6')
                    ax_side.plot(pos[:, 2], pos[:, 1], '--', color=color, linewidth=1.5, alpha=0.7)

        ax_side.set_xlabel('Z (m)', fontsize=11)
        ax_side.set_ylabel('Y (m)', fontsize=11)
        ax_side.set_title('Side View (Z-Y)', fontsize=12, fontweight='bold')
        ax_side.legend(loc='best', fontsize=8)
        ax_side.grid(True, alpha=0.3)

        # === Depth accuracy plot (right middle) - AbsRel ===
        ax_depth = fig.add_subplot(gs[1, 1])

        # Extract AbsRel metrics from frame_metrics
        if self.frame_metrics:
            frames = [m.frame_idx for m in self.frame_metrics]
            tri_absrel = [m.tri_AbsRel if m.tri_AbsRel is not None else np.nan for m in self.frame_metrics]
            ref_absrel = [m.ref_AbsRel if m.ref_AbsRel is not None else np.nan for m in self.frame_metrics]

            # Plot AbsRel (lower is better)
            if any(not np.isnan(v) for v in tri_absrel):
                ax_depth.plot(frames, tri_absrel, '-', color='#3498DB', label='Triangulated', linewidth=1.5, alpha=0.8)
            if any(not np.isnan(v) for v in ref_absrel):
                ax_depth.plot(frames, ref_absrel, '-', color='#E74C3C', label='Refined', linewidth=1.5, alpha=0.8)

            # Calculate and show mean values
            tri_mean = np.nanmean(tri_absrel) if any(not np.isnan(v) for v in tri_absrel) else None
            ref_mean = np.nanmean(ref_absrel) if any(not np.isnan(v) for v in ref_absrel) else None

            if tri_mean is not None:
                ax_depth.axhline(y=tri_mean, color='#3498DB', linestyle='--', alpha=0.5)
                ax_depth.text(frames[-1], tri_mean, f' {tri_mean:.3f}', va='center', fontsize=9, color='#3498DB')
            if ref_mean is not None:
                ax_depth.axhline(y=ref_mean, color='#E74C3C', linestyle='--', alpha=0.5)
                ax_depth.text(frames[-1], ref_mean, f' {ref_mean:.3f}', va='center', fontsize=9, color='#E74C3C')

            # Set reasonable y-limits based on data
            all_vals = [v for v in tri_absrel + ref_absrel if not np.isnan(v)]
            if all_vals:
                ymax = min(np.percentile(all_vals, 95) * 1.2, 1.0)
                ax_depth.set_ylim(0, ymax)

            ax_depth.legend(loc='upper right', fontsize=8)
        else:
            ax_depth.text(0.5, 0.5, 'No depth data yet', ha='center', va='center', fontsize=11, transform=ax_depth.transAxes)

        ax_depth.set_xlabel('Frame', fontsize=11)
        ax_depth.set_ylabel('AbsRel (↓)', fontsize=11)
        ax_depth.set_title('Depth Estimation - AbsRel', fontsize=12, fontweight='bold')
        ax_depth.grid(True, alpha=0.3)

        # === Metrics table (right bottom) ===
        ax_table = fig.add_subplot(gs[2, 1])
        ax_table.axis('off')

        # Compute current metrics
        current_ate = None
        current_rot = None
        current_trans = None
        if len(gt_arr) > 10:
            ate_metrics = compute_ate(gt_arr, est_arr)
            if ate_metrics:
                current_ate = ate_metrics['ATE_RMSE']
        if self.frame_metrics:
            rot_errors = [m.rot_error for m in self.frame_metrics if m.rot_error is not None]
            trans_errors = [m.trans_error for m in self.frame_metrics if m.trans_error is not None]
            if rot_errors:
                current_rot = np.mean(rot_errors)
            if trans_errors:
                current_trans = np.mean(trans_errors)

        # Build table data - two rows: Pose metrics and Depth metrics
        # Row 1: Pose metrics header
        headers = ['Method', 'ATE(m)', 'Rot(°)', 'Trans(°)', 'δ1.05', 'δ1.15', 'δ1.25']

        # Current method depth accuracy (δ thresholds)
        d105_val, d115_val, d125_val = None, None, None
        if self.frame_metrics:
            # Use refined if available, otherwise triangulated
            ref_d105 = [m.ref_d105 for m in self.frame_metrics if m.ref_d105 is not None]
            ref_d115 = [m.ref_d115 for m in self.frame_metrics if m.ref_d115 is not None]
            ref_d125 = [m.ref_d125 for m in self.frame_metrics if m.ref_d125 is not None]

            if ref_d105:
                d105_val = np.mean(ref_d105)
            if ref_d115:
                d115_val = np.mean(ref_d115)
            if ref_d125:
                d125_val = np.mean(ref_d125)

            # Fallback to triangulated
            if d105_val is None:
                tri_d105 = [m.tri_d105 for m in self.frame_metrics if m.tri_d105 is not None]
                if tri_d105:
                    d105_val = np.mean(tri_d105)
            if d115_val is None:
                tri_d115 = [m.tri_d115 for m in self.frame_metrics if m.tri_d115 is not None]
                if tri_d115:
                    d115_val = np.mean(tri_d115)
            if d125_val is None:
                tri_d125 = [m.tri_d125 for m in self.frame_metrics if m.tri_d125 is not None]
                if tri_d125:
                    d125_val = np.mean(tri_d125)

        table_data = []
        current_row = [
            self.runner.name,
            f'{current_ate:.2f}' if current_ate else '-',
            f'{current_rot:.2f}' if current_rot else '-',
            f'{current_trans:.2f}' if current_trans else '-',
            f'{d105_val:.1f}' if d105_val else '-',
            f'{d115_val:.1f}' if d115_val else '-',
            f'{d125_val:.1f}' if d125_val else '-',
        ]
        table_data.append(current_row)

        # Baselines
        for method, data in self._baseline_trajectories.items():
            if method.startswith('pr_depth:'):
                continue

            base_method = method.split(':')[0] if ':' in method else method
            display_name = METHOD_NAMES.get(base_method, method)
            metrics = data.get('metrics', {})

            ate = metrics.get('ATE_RMSE')
            rot = metrics.get('rot_error_mean')
            trans = metrics.get('trans_error_mean')

            row = [
                display_name,
                f'{ate:.2f}' if ate else '-',
                f'{rot:.2f}' if rot else '-',
                f'{trans:.2f}' if trans else '-',
                '-', '-', '-'  # No depth metrics for baselines
            ]
            table_data.append(row)

        # Create table
        table = ax_table.table(
            cellText=table_data,
            colLabels=headers,
            loc='center',
            cellLoc='center',
            colWidths=[0.18, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)

        # Color header
        for j, header in enumerate(headers):
            table[(0, j)].set_facecolor('#3498DB')
            table[(0, j)].set_text_props(color='white', fontweight='bold')

        # Color current method row
        for j in range(len(headers)):
            table[(1, j)].set_facecolor('#FADBD8')

        ax_table.set_title('Metrics Comparison', fontsize=12, fontweight='bold', pad=20)

        # === Progress bar (bottom, spans both columns) ===
        ax_progress = fig.add_subplot(gs[3, :])
        ax_progress.axis('off')

        progress = current_frame / total_frames if total_frames > 0 else 0
        status_text = "COMPLETE" if final else f"Processing... {current_frame}/{total_frames} ({progress*100:.1f}%)"

        # Draw progress bar
        bar_width = 0.8
        bar_height = 0.4
        bar_x = 0.1
        bar_y = 0.3

        ax_progress.add_patch(plt.Rectangle((bar_x, bar_y), bar_width, bar_height,
                                             facecolor='#ECF0F1', edgecolor='#BDC3C7', linewidth=2))
        fill_color = '#27AE60' if final else '#3498DB'
        ax_progress.add_patch(plt.Rectangle((bar_x, bar_y), bar_width * progress, bar_height,
                                             facecolor=fill_color, edgecolor='none'))

        ax_progress.text(0.5, 0.85, status_text, ha='center', va='center',
                         fontsize=14, fontweight='bold', transform=ax_progress.transAxes)

        ax_progress.set_xlim(0, 1)
        ax_progress.set_ylim(0, 1)

        # Main title
        title = f"PR-Depth Evaluation Dashboard - {self._dataset_id or 'Unknown Dataset'}"
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

        # Use fixed figure size - don't use bbox_inches='tight' which can expand the figure
        plt.savefig(self._dashboard_path, dpi=120,
                    facecolor='white', edgecolor='none')
        plt.close(fig)
