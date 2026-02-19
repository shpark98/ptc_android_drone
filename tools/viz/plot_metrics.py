#!/usr/bin/env python3
"""
Depth/Pose metrics plotting utility.
Usage: python tools/viz/plot_metrics.py --data results/metrics.npz --output results/metrics.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse


def plot_depth_metrics(metrics, ax=None, title='Depth Metrics'):
    """Plot depth metrics (MAE, RMSE, delta thresholds)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    frames = np.arange(len(metrics['mae']))

    ax.plot(frames, metrics['mae'], 'b-', label='MAE', linewidth=2)
    ax.plot(frames, metrics['rmse'], 'r-', label='RMSE', linewidth=2)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Error (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_pose_errors(rot_err, trans_err, ax=None, title='Pose Errors'):
    """Plot rotation and translation errors."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    frames = np.arange(len(rot_err))

    ax2 = ax.twinx()
    l1 = ax.plot(frames, rot_err, 'b-', label='Rotation (deg)', linewidth=2)
    l2 = ax2.plot(frames, trans_err, 'r-', label='Translation (deg)', linewidth=2)

    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Rotation Error (deg)', fontsize=12, color='b')
    ax2.set_ylabel('Translation Error (deg)', fontsize=12, color='r')
    ax.set_title(title, fontsize=14, fontweight='bold')

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right')
    ax.grid(True, alpha=0.3)

    return ax


def plot_trajectory_2d(poses_gt, poses_est, ax=None, title='Trajectory'):
    """Plot 2D trajectory comparison (X-Z plane)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    gt_x = [p[0, 3] for p in poses_gt]
    gt_z = [p[2, 3] for p in poses_gt]
    est_x = [p[0, 3] for p in poses_est]
    est_z = [p[2, 3] for p in poses_est]

    ax.plot(gt_x, gt_z, 'b-', label='Ground Truth', linewidth=2)
    ax.plot(est_x, est_z, 'r-', label='Estimated', linewidth=2)
    ax.plot(gt_x[0], gt_z[0], 'go', markersize=10, label='Start')
    ax.plot(gt_x[-1], gt_z[-1], 'ro', markersize=10, label='End')

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Z (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    return ax


def plot_delta_thresholds(d105, d115, d125, ax=None, title='Delta Thresholds'):
    """Plot delta accuracy thresholds."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    frames = np.arange(len(d105))

    ax.plot(frames, d105, 'g-', label='δ<1.05', linewidth=2)
    ax.plot(frames, d115, 'b-', label='δ<1.15', linewidth=2)
    ax.plot(frames, d125, 'r-', label='δ<1.25', linewidth=2)

    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    return ax


def main():
    parser = argparse.ArgumentParser(description='Plot depth/pose metrics')
    parser.add_argument('--data', required=True, help='Path to metrics .npz file')
    parser.add_argument('--output', default='results/metrics.png')
    parser.add_argument('--type', choices=['depth', 'pose', 'trajectory', 'all'], default='all')
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)

    if args.type == 'all':
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        if 'mae' in data:
            plot_depth_metrics({'mae': data['mae'], 'rmse': data['rmse']}, axes[0, 0])
        if 'd105' in data:
            plot_delta_thresholds(data['d105'], data['d115'], data['d125'], axes[0, 1])
        if 'rot_err' in data:
            plot_pose_errors(data['rot_err'], data['trans_err'], axes[1, 0])
        if 'poses_gt' in data:
            plot_trajectory_2d(data['poses_gt'], data['poses_est'], axes[1, 1])

        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        if args.type == 'depth':
            plot_depth_metrics({'mae': data['mae'], 'rmse': data['rmse']}, ax)
        elif args.type == 'pose':
            plot_pose_errors(data['rot_err'], data['trans_err'], ax)
        elif args.type == 'trajectory':
            plot_trajectory_2d(data['poses_gt'], data['poses_est'], ax)

    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
