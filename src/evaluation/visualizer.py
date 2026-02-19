"""Visualization utilities for evaluation results."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend by default
import matplotlib.pyplot as plt


class Visualizer:
    """Visualization tools for evaluation results.

    Provides consistent plotting styles for:
    - Trajectory plots (bird's eye view, side view)
    - Method comparison plots
    - Error bar charts

    Example:
        >>> viz = Visualizer()
        >>> viz.plot_trajectory(gt_pos, est_pos, 'PR-Depth', save_path)
        >>> viz.plot_comparison(all_results, gt_pos, save_path)
    """

    # Method colors
    COLORS = {
        'pr_depth': '#E74C3C',    # Red
        'PR-Depth': '#E74C3C',
        'madpose': '#3498DB',     # Blue
        'MADPose': '#3498DB',
        'batrack': '#2ECC71',     # Green
        'BaTrack': '#2ECC71',
        'monst3r': '#9B59B6',     # Purple
        'MonST3R': '#9B59B6',
        'default': '#95A5A6',     # Gray
    }

    # Display names
    DISPLAY_NAMES = {
        'pr_depth': 'PR-Depth',
        'madpose': 'MADPose',
        'batrack': 'BaTrack',
        'monst3r': 'MonST3R',
    }

    def __init__(self, style: str = 'default', dpi: int = 150):
        """Initialize visualizer.

        Args:
            style: Matplotlib style ('default', 'seaborn', etc.)
            dpi: Output DPI for saved figures
        """
        self.dpi = dpi
        if style != 'default':
            plt.style.use(style)

    def get_color(self, method: str) -> str:
        """Get color for method."""
        return self.COLORS.get(method, self.COLORS.get(method.lower(), self.COLORS['default']))

    def get_display_name(self, method: str) -> str:
        """Get display name for method."""
        return self.DISPLAY_NAMES.get(method.lower(), method)

    def plot_trajectory(
        self,
        gt_positions: np.ndarray,
        est_positions: np.ndarray,
        method_name: str,
        save_path: Path,
        metrics: Optional[Dict] = None,
        timing_ms: Optional[float] = None,
        title: Optional[str] = None,
    ):
        """Plot single method trajectory comparison.

        Args:
            gt_positions: (N, 3) ground truth positions
            est_positions: (N, 3) estimated positions
            method_name: Method name for display
            save_path: Path to save figure
            metrics: Optional dict with 'ate', 'rpe_trans', 'rpe_rot'
            timing_ms: Optional average time per frame (ms)
            title: Optional custom title
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        color = self.get_color(method_name)
        display_name = self.get_display_name(method_name)

        # Bird's eye view (X-Z)
        ax = axes[0]
        ax.plot(gt_positions[:, 0], gt_positions[:, 2], 'b-',
                label='Ground Truth', linewidth=2, alpha=0.8)
        ax.plot(est_positions[:, 0], est_positions[:, 2], '--',
                color=color, label=display_name, linewidth=2, alpha=0.8)
        ax.scatter([gt_positions[0, 0]], [gt_positions[0, 2]],
                   c='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter([gt_positions[-1, 0]], [gt_positions[-1, 2]],
                   c='blue', s=100, marker='x', label='GT End', zorder=5)
        ax.scatter([est_positions[-1, 0]], [est_positions[-1, 2]],
                   c='red', s=100, marker='x', label='Est End', zorder=5)
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Z (m)', fontsize=11)
        ax.set_title("Bird's Eye View (X-Z)", fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.axis('equal')
        ax.grid(True, alpha=0.3)

        # Side view (Z-Y)
        ax = axes[1]
        ax.plot(gt_positions[:, 2], gt_positions[:, 1], 'b-',
                label='Ground Truth', linewidth=2, alpha=0.8)
        ax.plot(est_positions[:, 2], est_positions[:, 1], '--',
                color=color, label=display_name, linewidth=2, alpha=0.8)
        ax.set_xlabel('Z (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_title('Side View (Z-Y)', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Main title
        if title is None:
            title = f'{display_name} Trajectory'
            if timing_ms is not None:
                title += f' | {timing_ms:.1f} ms/frame'
            if metrics:
                if 'ATE_RMSE' in metrics:
                    title += f' | ATE: {metrics["ATE_RMSE"]:.2f}m'
                elif 'ate' in metrics:
                    title += f' | ATE: {metrics["ate"]:.2f}m'

        plt.suptitle(title, fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def plot_comparison(
        self,
        results: Dict[str, Dict],
        gt_positions: np.ndarray,
        save_path: Path,
        title: str = "Method Comparison",
    ):
        """Plot comparison of multiple methods.

        Args:
            results: Dict of method_name -> {
                'est_positions': (N, 3),
                'timing_ms': float,
                'metrics': {'ate': float, ...}
            }
            gt_positions: (N, 3) ground truth positions
            save_path: Path to save figure
            title: Plot title
        """
        methods = list(results.keys())
        if not methods:
            return

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.25)

        # Trajectory plot (all methods overlaid)
        ax_traj = fig.add_subplot(gs[0, :])
        ax_traj.plot(gt_positions[:, 0], gt_positions[:, 2], 'k-',
                     label='GT', linewidth=2.5, alpha=0.8)

        for method in methods:
            res = results[method]
            est_pos = res.get('est_positions')
            if est_pos is not None:
                color = self.get_color(method)
                display_name = self.get_display_name(method)
                ax_traj.plot(est_pos[:, 0], est_pos[:, 2], '--',
                            color=color, label=display_name, linewidth=2)

        ax_traj.scatter([gt_positions[0, 0]], [gt_positions[0, 2]],
                        c='green', s=150, marker='o', label='Start', zorder=10)
        ax_traj.set_xlabel('X (m)', fontsize=12)
        ax_traj.set_ylabel('Z (m)', fontsize=12)
        ax_traj.set_title(title, fontsize=14, fontweight='bold')
        ax_traj.legend(loc='best', fontsize=10)
        ax_traj.axis('equal')
        ax_traj.grid(True, alpha=0.3)

        # Timing bar chart
        ax_time = fig.add_subplot(gs[1, 0])
        timing_data = []
        for method in methods:
            timing = results[method].get('timing_ms', 0)
            timing_data.append((self.get_display_name(method), timing, self.get_color(method)))

        timing_data.sort(key=lambda x: x[1])
        names, times, colors = zip(*timing_data) if timing_data else ([], [], [])

        bars = ax_time.barh(names, times, color=colors)
        for bar, t in zip(bars, times):
            ax_time.text(t + 1, bar.get_y() + bar.get_height()/2,
                        f'{t:.1f} ms', va='center', fontsize=10)

        ax_time.set_xlabel('Time per Frame (ms)', fontsize=11)
        ax_time.set_title('Execution Time', fontsize=12, fontweight='bold')
        if times:
            ax_time.set_xlim(0, max(times) * 1.3)

        # Error bar chart
        ax_err = fig.add_subplot(gs[1, 1])

        x = np.arange(len(methods))
        width = 0.35

        ate_vals = []
        rot_vals = []
        for method in methods:
            metrics = results[method].get('metrics', {})
            ate = metrics.get('ATE_RMSE', metrics.get('ate', 0))
            rot = metrics.get('rot_error_mean', metrics.get('rpe_rot', 0))
            ate_vals.append(ate if ate else 0)
            rot_vals.append(rot if rot else 0)

        bars1 = ax_err.bar(x - width/2, ate_vals, width, label='ATE (m)', color='#3498DB')
        bars2 = ax_err.bar(x + width/2, rot_vals, width, label='Rot Error (°)', color='#E74C3C')

        ax_err.set_ylabel('Error', fontsize=11)
        ax_err.set_title('Trajectory Errors', fontsize=12, fontweight='bold')
        ax_err.set_xticks(x)
        ax_err.set_xticklabels([self.get_display_name(m) for m in methods], fontsize=10)
        ax_err.legend(fontsize=9)
        ax_err.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars1:
            if bar.get_height() > 0:
                ax_err.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            if bar.get_height() > 0:
                ax_err.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def plot_depth_comparison(
        self,
        gt_depth: np.ndarray,
        tri_depth: np.ndarray,
        ref_depth: np.ndarray,
        save_path: Path,
        title: str = "Depth Comparison",
        vmax: float = 80.0,
    ):
        """Plot depth map comparison.

        Args:
            gt_depth: Ground truth depth (H, W)
            tri_depth: Triangulated depth (H, W)
            ref_depth: Refined depth (H, W)
            save_path: Path to save figure
            title: Plot title
            vmax: Maximum depth for colorbar
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # GT depth
        im0 = axes[0].imshow(gt_depth, cmap='turbo', vmin=0, vmax=vmax)
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Triangulated depth
        im1 = axes[1].imshow(tri_depth, cmap='turbo', vmin=0, vmax=vmax)
        axes[1].set_title('Triangulated')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Refined depth
        im2 = axes[2].imshow(ref_depth, cmap='turbo', vmin=0, vmax=vmax)
        axes[2].set_title('Refined')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def plot_metrics_over_time(
        self,
        frame_indices: List[int],
        metrics_dict: Dict[str, List[float]],
        save_path: Path,
        title: str = "Metrics Over Time",
    ):
        """Plot metrics evolution over frames.

        Args:
            frame_indices: List of frame indices
            metrics_dict: Dict of metric_name -> list of values
            save_path: Path to save figure
            title: Plot title
        """
        n_metrics = len(metrics_dict)
        if n_metrics == 0:
            return

        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 3 * n_metrics), sharex=True)
        if n_metrics == 1:
            axes = [axes]

        colors = plt.cm.tab10(np.linspace(0, 1, n_metrics))

        for i, (name, values) in enumerate(metrics_dict.items()):
            ax = axes[i]
            ax.plot(frame_indices[:len(values)], values, color=colors[i], linewidth=1.5)
            ax.set_ylabel(name, fontsize=11)
            ax.grid(True, alpha=0.3)

            # Add mean line
            mean_val = np.nanmean(values)
            ax.axhline(y=mean_val, color=colors[i], linestyle='--', alpha=0.5)
            ax.text(frame_indices[-1], mean_val, f' mean={mean_val:.2f}',
                   va='center', fontsize=9)

        axes[-1].set_xlabel('Frame', fontsize=11)
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()


def enable_interactive():
    """Enable interactive matplotlib backend."""
    matplotlib.use('TkAgg')
    plt.ion()


def disable_interactive():
    """Disable interactive matplotlib backend."""
    plt.ioff()
    matplotlib.use('Agg')
