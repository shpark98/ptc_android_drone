#!/usr/bin/env python3
"""
Unified pose estimation comparison: PR-Depth vs MADPose

Uses the new unified estimator interfaces from src/estimators/.

Usage:
    # Compare methods on a sequence
    python compare.py --exp-name test --date 2011_10_03 --drive 0027 --frames 0 100

    # Use specific depth models
    python compare.py --exp-name full_test --pr-depth-only
    python compare.py --exp-name full_test --madpose-only
"""
import argparse
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'cpp' / 'build'))

from configs import get_dataset_paths
from dataloader.dataset.kitti import KITTIEigenSplit
from src.estimators import (
    DepthAnythingEstimator,
    DISFlowEstimator,
    PRDepthEstimator,
    MADPoseEstimator,
)
from src.evaluation import compute_depth_metrics, compute_pose_error, compute_ate
from src.evaluation.metrics import integrate_trajectory
from tools.eval.result_saver import Experiment


def parse_args():
    parser = argparse.ArgumentParser(description='Compare pose estimation methods')

    # Required
    parser.add_argument('--exp-name', required=True, help='Experiment name')

    # Dataset
    parser.add_argument('--date', default='2011_09_26', help='KITTI date')
    parser.add_argument('--drive', default='0001', help='KITTI drive')
    parser.add_argument('--frames', type=int, nargs=2, default=[1, 100],
                        help='Frame range [start, end)')

    # Methods
    parser.add_argument('--pr-depth-only', action='store_true', help='Only run PR-Depth')
    parser.add_argument('--madpose-only', action='store_true', help='Only run MADPose')

    # PR-Depth options
    parser.add_argument('--iterative', action='store_true', help='Enable iterative refinement')
    parser.add_argument('--dc-threshold', type=float, default=0.65,
                        help='Depth consistency threshold')

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"Pose Estimation Comparison: {args.exp_name}")
    print(f"{'='*60}")
    print(f"Date: {args.date}, Drive: {args.drive}")
    print(f"Frames: {args.frames[0]} to {args.frames[1]}")
    print()

    # Create experiment
    exp = Experiment(args.exp_name, description=f"Compare PR-Depth vs MADPose on {args.drive}")
    exp.set_config({
        'date': args.date,
        'drive': args.drive,
        'frame_range': args.frames,
        'iterative': args.iterative,
        'dc_threshold': args.dc_threshold,
    })

    # Load dataset
    paths = get_dataset_paths('kitti')
    dataset = KITTIEigenSplit(
        rgb_path=paths['rgb_path'],
        date=args.date,
        drive=args.drive,
        depth_path=paths['depth_path'],
        data_type='pointcloud'
    )

    fx, fy, cx, cy = dataset.get_intrinsics()
    H, W = dataset.get_image_size()
    K = dataset.get_K_matrix()

    print(f"Image size: {H}x{W}")
    print(f"Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    print(f"Total frames: {len(dataset)}")
    print()

    # Initialize estimators
    print("Initializing estimators...")

    # Flow (shared)
    flow_est = DISFlowEstimator(preset='medium')
    print(f"  Flow: {flow_est.name}")

    # Depth (DepthAnything for PR-Depth)
    depth_est = DepthAnythingEstimator(input_size=518, encoder='vitl')
    print(f"  Depth: {depth_est.name}")

    # Determine which methods to run
    run_pr_depth = not args.madpose_only
    run_madpose = not args.pr_depth_only

    # PR-Depth estimator
    pr_depth_est = None
    if run_pr_depth:
        pr_depth_est = PRDepthEstimator(
            H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy,
            iterative=args.iterative,
            iterative_iters=1,
        )
        print(f"  Pose (PR-Depth): {pr_depth_est.name}")

    # MADPose estimator (note: should use UniDepth, but using DA for now)
    madpose_est = None
    if run_madpose:
        madpose_est = MADPoseEstimator(K=K, grid_stride=12)
        print(f"  Pose (MADPose): {madpose_est.name}")
        print("  WARNING: MADPose should ideally use UniDepth for metric depth")

    print()

    # Storage for results
    results = {
        'pr_depth': {'rot': [], 'trans': [], 'R_list': [], 't_list': [], 'baselines': []},
        'madpose': {'rot': [], 'trans': [], 'R_list': [], 't_list': [], 'baselines': []},
    }

    # Ground truth storage
    gt_R_list = []
    gt_t_list = []
    gt_baselines = []

    # Run comparison
    frame_start, frame_end = args.frames
    frame_end = min(frame_end, len(dataset))

    prev_img = None
    prev_depth = None

    print(f"Processing frames {frame_start} to {frame_end}...")

    for idx in tqdm(range(frame_start, frame_end)):
        data = dataset.get(idx)
        if data is None:
            continue

        img = data['image_og']
        gt_depth = data['depth_og']

        # Estimate depth
        inv_depth = depth_est.infer(img)

        # Convert to metric depth for MADPose (rough approximation)
        # This is a simplification - proper approach needs UniDepth
        metric_depth = 1.0 / (inv_depth + 1e-8)
        metric_depth = np.clip(metric_depth, 0.1, 100.0)

        if prev_img is None:
            prev_img = img
            prev_depth = inv_depth
            prev_metric_depth = metric_depth
            continue

        # Compute flow
        flow = flow_est.compute(prev_img, img)

        # Get GT pose
        R_gt, t_gt = dataset.get_relative_pose(idx)
        baseline = dataset.get_baseline(idx)

        gt_R_list.append(R_gt)
        gt_t_list.append(t_gt)
        gt_baselines.append(baseline)

        # PR-Depth estimation
        if run_pr_depth:
            result = pr_depth_est.estimate(
                prev_img, img,
                prev_depth, inv_depth,
                flow, baseline
            )

            if result.success:
                errors = compute_pose_error(result.R, result.t, R_gt, t_gt)
                results['pr_depth']['rot'].append(errors['rot_error'])
                results['pr_depth']['trans'].append(errors['trans_error'])
                results['pr_depth']['R_list'].append(result.R)
                results['pr_depth']['t_list'].append(result.t)
                results['pr_depth']['baselines'].append(baseline)

        # MADPose estimation
        if run_madpose:
            result = madpose_est.estimate(
                prev_img, img,
                prev_metric_depth, metric_depth,
                flow, baseline=None  # MADPose doesn't need baseline
            )

            if result.success:
                errors = compute_pose_error(result.R, result.t, R_gt, t_gt)
                results['madpose']['rot'].append(errors['rot_error'])
                results['madpose']['trans'].append(errors['trans_error'])
                results['madpose']['R_list'].append(result.R)
                results['madpose']['t_list'].append(result.t)
                results['madpose']['baselines'].append(baseline)

        # Update previous
        prev_img = img
        prev_depth = inv_depth
        prev_metric_depth = metric_depth

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    print(f"\n{'Method':<15} {'Rot Err (deg)':<20} {'Trans Err (deg)':<20} {'Frames':<10}")
    print("-" * 65)

    for method in ['pr_depth', 'madpose']:
        if not results[method]['rot']:
            continue

        rot_mean = np.mean(results[method]['rot'])
        rot_std = np.std(results[method]['rot'])
        trans_mean = np.mean(results[method]['trans'])
        trans_std = np.std(results[method]['trans'])
        n_frames = len(results[method]['rot'])

        print(f"{method:<15} {rot_mean:.4f} +/- {rot_std:.4f}    {trans_mean:.4f} +/- {trans_std:.4f}    {n_frames}")

        # Save to experiment
        exp.save_pose_metrics(
            method=method,
            sequence=args.drive,
            rot_errors=results[method]['rot'],
            trans_errors=results[method]['trans'],
        )

    # Compute trajectories and ATE
    print("\n--- Trajectory Metrics ---")

    gt_positions, _ = integrate_trajectory(gt_R_list, gt_t_list)

    for method in ['pr_depth', 'madpose']:
        if not results[method]['R_list']:
            continue

        est_positions, _ = integrate_trajectory(
            results[method]['R_list'],
            results[method]['t_list'],
        )

        ate = compute_ate(gt_positions, est_positions)
        if ate:
            print(f"\n{method}:")
            print(f"  ATE RMSE: {ate['ATE_RMSE']:.4f} m")
            print(f"  Final Drift: {ate['final_drift']:.4f} m")
            print(f"  Trajectory Length: {ate['trajectory_length']:.2f} m")

        # Save trajectory plot
        exp.save_trajectory(
            method=method,
            sequence=args.drive,
            gt_positions=gt_positions,
            est_positions=est_positions,
            title=f"{method} vs GT - {args.drive}"
        )

    # Save summary
    exp.save_summary()

    print(f"\nResults saved to: results/experiments/{exp.full_name}/")


if __name__ == '__main__':
    main()
