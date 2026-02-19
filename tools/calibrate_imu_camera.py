#!/usr/bin/env python3
"""IMU-Camera extrinsic calibration using LiDAR ground truth.

Finds the optimal yaw offset between IMU and camera by:
1. Testing various yaw offsets (0-360 degrees)
2. Computing 3D warp using IMU rotation + GPS baseline
3. Comparing warped depth with LiDAR ground truth
4. Selecting the yaw offset with highest accuracy

Usage:
    python tools/calibrate_imu_camera.py -w outdoor --max-frames 200
    python tools/calibrate_imu_camera.py -w indoor --max-frames 100
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import cv2
from tqdm import tqdm

from dataloader import WheelLoader, WheelEvalWrapper
from src.evaluation.metrics import compute_depth_metrics

# Wheel dataset name mapping
WHEEL_DATASETS = {
    'indoor': '25_10_20_14_50',
    'outdoor': '25_10_20_14_30',
    'forest': '25_11_04_16_00',
}


def warp_depth_3d(depth_prev: np.ndarray, R: np.ndarray, t: np.ndarray,
                  fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """3D warp previous depth to current frame using pose (R, t).

    Args:
        depth_prev: Previous frame depth (H, W)
        R: Rotation matrix (3, 3) from prev to curr
        t: Translation vector (3,) from prev to curr
        fx, fy, cx, cy: Camera intrinsics

    Returns:
        Warped depth in current frame (H, W)
    """
    H, W = depth_prev.shape

    # Create pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Valid depth mask
    valid = depth_prev > 0

    # Back-project to 3D (prev frame)
    Z = depth_prev
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    # Stack points (3, H*W)
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=0)

    # Transform to current frame: P_curr = R @ P_prev + t
    points_curr = R @ points + t.reshape(3, 1)

    # Project to current image
    X_curr = points_curr[0].reshape(H, W)
    Y_curr = points_curr[1].reshape(H, W)
    Z_curr = points_curr[2].reshape(H, W)

    # Compute pixel coordinates in current frame
    u_curr = (fx * X_curr / Z_curr + cx).astype(np.float32)
    v_curr = (fy * Y_curr / Z_curr + cy).astype(np.float32)

    # Forward warp (scatter)
    warped = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            if not valid[i, j]:
                continue
            if Z_curr[i, j] <= 0:
                continue

            ui = int(round(u_curr[i, j]))
            vi = int(round(v_curr[i, j]))

            if 0 <= ui < W and 0 <= vi < H:
                # Keep closer depth (handle occlusion)
                if warped[vi, ui] == 0 or Z_curr[i, j] < warped[vi, ui]:
                    warped[vi, ui] = Z_curr[i, j]

    return warped


def compute_warp_accuracy(warped: np.ndarray, gt: np.ndarray,
                          threshold: float = 1.25) -> dict:
    """Compute accuracy between warped depth and ground truth.

    Args:
        warped: Warped depth (H, W)
        gt: Ground truth depth (H, W)
        threshold: Accuracy threshold (delta < threshold)

    Returns:
        Dictionary with accuracy metrics
    """
    # Valid mask: both warped and GT have depth
    valid = (warped > 0) & (gt > 0)

    if valid.sum() < 100:
        return {'accuracy': 0.0, 'valid_pixels': 0, 'mae': float('inf')}

    w = warped[valid]
    g = gt[valid]

    # Compute delta
    delta = np.maximum(w / g, g / w)

    # Accuracy: percentage of pixels with delta < threshold
    accuracy = (delta < threshold).mean() * 100

    # MAE
    mae = np.abs(w - g).mean()

    return {
        'accuracy': accuracy,
        'valid_pixels': valid.sum(),
        'mae': mae,
    }


def test_yaw_offset(dataset: WheelEvalWrapper, yaw_offset_deg: float,
                    fx: float, fy: float, cx: float, cy: float,
                    max_frames: int = 100, min_baseline: float = 0.05) -> dict:
    """Test a specific yaw offset and return accuracy metrics.

    Args:
        dataset: WheelEvalWrapper with specified yaw offset
        yaw_offset_deg: Yaw offset in degrees
        fx, fy, cx, cy: Camera intrinsics
        max_frames: Maximum frames to process
        min_baseline: Minimum baseline to consider

    Returns:
        Dictionary with accuracy metrics
    """
    # Create dataset with this yaw offset
    loader = dataset._loader
    test_dataset = WheelEvalWrapper(
        loader,
        imu_yaw_offset=yaw_offset_deg,
        use_imu_rotation=True,
        use_gps_forward_only=True  # Use GPS baseline as forward motion
    )

    accuracies = []
    z_ratios = []

    prev_data = None

    n_frames = min(len(test_dataset), max_frames)

    for idx in range(n_frames):
        data = test_dataset.get(idx)
        if data is None:
            continue

        if idx == 0:
            prev_data = data
            continue

        # Get pose
        R, t = test_dataset.get_relative_pose(idx)
        baseline = np.linalg.norm(t)

        if baseline < min_baseline:
            prev_data = data
            continue

        # Get depths
        depth_prev = prev_data['depth']
        depth_curr = data['depth']  # LiDAR GT

        # Skip if no valid depth
        if (depth_prev > 0).sum() < 1000 or (depth_curr > 0).sum() < 1000:
            prev_data = data
            continue

        # 3D warp
        warped = warp_depth_3d(depth_prev, R, t, fx, fy, cx, cy)

        # Compute accuracy
        metrics = compute_warp_accuracy(warped, depth_curr)

        if metrics['valid_pixels'] >= 1000:
            accuracies.append(metrics['accuracy'])

            # Compute Z-ratio (forward motion ratio)
            t_norm = t / (np.linalg.norm(t) + 1e-10)
            z_ratios.append(abs(t_norm[2]))

        prev_data = data

    if len(accuracies) == 0:
        return {
            'yaw_offset': yaw_offset_deg,
            'accuracy_mean': 0.0,
            'accuracy_std': 0.0,
            'z_ratio_mean': 0.0,
            'n_valid': 0,
        }

    return {
        'yaw_offset': yaw_offset_deg,
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'z_ratio_mean': np.mean(z_ratios),
        'n_valid': len(accuracies),
    }


def main():
    parser = argparse.ArgumentParser(description='IMU-Camera calibration')

    parser.add_argument('--wheel-name', '-w', type=str, required=True,
                        choices=['indoor', 'outdoor', 'forest',
                                 '25_10_20_14_30', '25_10_20_14_50', '25_11_04_16_00'],
                        help='Wheel dataset name')
    parser.add_argument('--max-frames', type=int, default=200,
                        help='Maximum frames to process')
    parser.add_argument('--yaw-step', type=float, default=5.0,
                        help='Yaw offset step in degrees')
    parser.add_argument('--min-baseline', type=float, default=0.05,
                        help='Minimum baseline to consider')

    args = parser.parse_args()

    # Map friendly names
    wheel_name = WHEEL_DATASETS.get(args.wheel_name, args.wheel_name)

    print(f"\n{'='*60}")
    print(f"IMU-Camera Calibration for: {wheel_name}")
    print(f"{'='*60}")

    # Load dataset
    loader = WheelLoader(name=wheel_name)
    dataset = WheelEvalWrapper(loader)

    print(f"Loaded {len(dataset)} frames")

    # Get intrinsics
    fx, fy, cx, cy = dataset.get_intrinsics()
    print(f"Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # Test yaw offsets
    yaw_offsets = np.arange(0, 360, args.yaw_step)
    results = []

    print(f"\nTesting {len(yaw_offsets)} yaw offsets...")
    print(f"{'Yaw':>8} {'Accuracy':>10} {'Std':>8} {'Z-ratio':>8} {'N':>6}")
    print("-" * 50)

    for yaw in tqdm(yaw_offsets, desc="Calibrating"):
        result = test_yaw_offset(
            dataset, yaw, fx, fy, cx, cy,
            max_frames=args.max_frames,
            min_baseline=args.min_baseline
        )
        results.append(result)

        # Print progress
        tqdm.write(f"{yaw:>8.1f}° {result['accuracy_mean']:>9.2f}% "
                   f"{result['accuracy_std']:>7.2f} {result['z_ratio_mean']:>7.3f} "
                   f"{result['n_valid']:>6}")

    # Find best yaw offset
    best_idx = np.argmax([r['accuracy_mean'] for r in results])
    best_result = results[best_idx]

    print(f"\n{'='*60}")
    print(f"CALIBRATION RESULT")
    print(f"{'='*60}")
    print(f"Best yaw offset: {best_result['yaw_offset']:.1f}°")
    print(f"Accuracy: {best_result['accuracy_mean']:.2f}% ± {best_result['accuracy_std']:.2f}%")
    print(f"Z-ratio: {best_result['z_ratio_mean']:.3f}")
    print(f"Valid frames: {best_result['n_valid']}")

    # Also find best with Z-ratio > 0.9 (forward motion constraint)
    forward_results = [r for r in results if r['z_ratio_mean'] > 0.9]
    if forward_results:
        best_forward_idx = np.argmax([r['accuracy_mean'] for r in forward_results])
        best_forward = forward_results[best_forward_idx]
        print(f"\nBest with forward motion (Z-ratio > 0.9):")
        print(f"  Yaw offset: {best_forward['yaw_offset']:.1f}°")
        print(f"  Accuracy: {best_forward['accuracy_mean']:.2f}%")

    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    output_path = PROJECT_ROOT / "results" / f"calibration_{wheel_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_path}")

    # Plot results
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        yaws = [r['yaw_offset'] for r in results]
        accs = [r['accuracy_mean'] for r in results]
        zrats = [r['z_ratio_mean'] for r in results]

        ax1.plot(yaws, accs, 'b-', linewidth=2)
        ax1.axvline(best_result['yaw_offset'], color='r', linestyle='--',
                    label=f"Best: {best_result['yaw_offset']:.1f}°")
        ax1.set_xlabel('Yaw Offset (degrees)')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('3D Warp Accuracy vs Yaw Offset')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(yaws, zrats, 'g-', linewidth=2)
        ax2.axhline(0.9, color='orange', linestyle='--', label='Z-ratio = 0.9')
        ax2.set_xlabel('Yaw Offset (degrees)')
        ax2.set_ylabel('Z-ratio (forward motion)')
        ax2.set_title('Forward Motion Ratio vs Yaw Offset')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = PROJECT_ROOT / "results" / f"calibration_{wheel_name}.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Saved plot to: {plot_path}")
        plt.close()

    except ImportError:
        print("(matplotlib not available, skipping plot)")


if __name__ == '__main__':
    main()
