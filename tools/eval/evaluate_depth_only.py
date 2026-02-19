#!/usr/bin/env python3
"""Depth-only evaluation script for comparing depth estimation methods.

Compares depth estimation quality (without pose) for:
- PR-Depth (uses estimated poses for TAE)
- UniDepth (uses GT poses for TAE)
- Video Depth Anything (uses GT poses for TAE)
- Depth Anything v2 (uses GT poses for TAE)

Metrics:
- AbsRel, RMSE, MAE
- Delta thresholds (d105, d115, d125)
- TAE (Temporal Alignment Error) from Video Depth Anything

Usage:
    # Evaluate PR-Depth experiment
    python tools/eval/evaluate_depth_only.py --method pr_depth \
        --depths results/kitti_2011_10_03_0027/pr_depth/latest/depths.npz

    # Evaluate VDA
    python tools/eval/evaluate_depth_only.py --method vda \
        --depths results/kitti_2011_10_03_0027/baselines/vda_stream/depths.npz

    # Evaluate UniDepth
    python tools/eval/evaluate_depth_only.py --method unidepth \
        --depths results/kitti_2011_10_03_0027/baselines/unidepth/depths.npz

    # Compare multiple methods
    python tools/eval/evaluate_depth_only.py --compare \
        --pr-depth results/.../pr_depth/latest/depths.npz \
        --vda results/.../baselines/vda_stream/depths.npz \
        --unidepth results/.../baselines/unidepth/depths.npz
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import get_dataset_paths
from dataloader import KITTIEigenSplit
from src.evaluation.metrics import compute_depth_metrics, compute_tae


def load_depths(depths_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load depth maps from npz file.

    Returns:
        depths: (N, H, W) depth maps
        frame_indices: (N,) frame indices
    """
    data = np.load(str(depths_path))

    # Try different keys
    if 'z_refined' in data:
        depths = data['z_refined']
    elif 'z_tri' in data:
        depths = data['z_tri']
    elif 'depths' in data:
        depths = data['depths']
    else:
        raise ValueError(f"No depth data found in {depths_path}. Keys: {list(data.keys())}")

    frame_indices = data.get('frame_indices', np.arange(len(depths)))

    return depths, frame_indices


def load_poses(trajectory_path: Path, use_estimated: bool = True) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    """Load poses and baselines from trajectory.npz file.

    Args:
        trajectory_path: Path to trajectory.npz
        use_estimated: If True, use estimated poses; if False, use GT poses

    Returns:
        poses: List of (R, t) tuples for relative poses
        baselines: List of baseline values
    """
    data = np.load(str(trajectory_path))

    # Load poses
    if use_estimated:
        Rs = data.get('est_Rs', np.array([]))
        ts = data.get('est_ts', np.array([]))
    else:
        Rs = data.get('gt_Rs', np.array([]))
        ts = data.get('gt_ts', np.array([]))

    poses = []
    if len(Rs) > 0 and len(ts) > 0:
        poses = [(Rs[i], ts[i]) for i in range(len(Rs))]

    # Load baselines
    baselines = list(data.get('baselines', np.array([])))

    return poses, baselines


def evaluate_depths(
    depths: np.ndarray,
    frame_indices: np.ndarray,
    dataset,
    poses: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    baselines: Optional[List[float]] = None,
    use_gt_pose_for_tae: bool = True,
    max_depth: float = 80.0,
) -> Dict[str, float]:
    """Evaluate depth maps against GT.

    Args:
        depths: (N, H, W) predicted depth maps
        frame_indices: (N,) frame indices
        dataset: Dataset with GT depth and poses
        poses: Optional list of (R, t) for TAE (if None and use_gt_pose_for_tae, use GT)
        baselines: Optional list of baselines for TAE
        use_gt_pose_for_tae: Use GT poses for TAE computation
        max_depth: Maximum depth for metrics

    Returns:
        Dictionary with aggregated metrics
    """
    all_metrics = []
    gt_poses = []
    gt_baselines = []

    fx, fy, cx, cy = dataset.get_intrinsics()

    for i, frame_idx in enumerate(tqdm(frame_indices, desc="Evaluating")):
        frame_idx = int(frame_idx)

        # Get GT depth
        sample = dataset.get(frame_idx)
        if sample is None:
            continue

        gt_depth = sample.get('depth_og')
        if gt_depth is None:
            continue

        pred_depth = depths[i]

        # Resize if needed
        if pred_depth.shape != gt_depth.shape:
            import cv2
            pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)

        # Compute depth metrics
        metrics = compute_depth_metrics(pred_depth, gt_depth, max_depth=max_depth)
        if metrics:
            metrics['frame_idx'] = frame_idx
            all_metrics.append(metrics)

        # Collect GT poses for TAE
        if frame_idx > 0:
            R_gt, t_gt = dataset.get_relative_pose(frame_idx)
            baseline_gt = dataset.get_baseline(frame_idx)
            gt_poses.append((R_gt, t_gt))
            gt_baselines.append(baseline_gt)

    if not all_metrics:
        return {}

    # Aggregate metrics
    df = pd.DataFrame(all_metrics)
    result = {
        'AbsRel': float(df['AbsRel'].mean()),
        'RMSE': float(df['RMSE'].mean()),
        'MAE': float(df['MAE'].mean()),
        'd105': float(df['d105'].mean()),
        'd115': float(df['d115'].mean()),
        'd125': float(df['d125'].mean()),
        'num_frames': len(df),
    }

    # Compute TAE
    if len(depths) > 1:
        # Determine which poses to use
        if use_gt_pose_for_tae:
            tae_poses = gt_poses
            tae_baselines = gt_baselines
        else:
            tae_poses = poses if poses else gt_poses
            tae_baselines = baselines if baselines else gt_baselines

        if tae_poses and tae_baselines:
            # Filter valid depths
            valid_depths = [depths[i] for i in range(len(depths)) if depths[i] is not None and depths[i].size > 0]

            tae_result = compute_tae(
                depths=valid_depths,
                poses=tae_poses,
                baselines=tae_baselines,
                fx=fx, fy=fy, cx=cx, cy=cy
            )
            if tae_result:
                result['TAE'] = tae_result['TAE']
                result['TAE_forward'] = tae_result['TAE_forward']
                result['TAE_backward'] = tae_result['TAE_backward']

    return result


def main():
    parser = argparse.ArgumentParser(description='Depth-only evaluation')

    # Single method evaluation
    parser.add_argument('--method', type=str, default=None,
                        choices=['pr_depth', 'vda', 'vda_stream', 'vda_offline', 'unidepth', 'dav2'],
                        help='Method to evaluate')
    parser.add_argument('--depths', type=str, default=None,
                        help='Path to depths.npz file')
    parser.add_argument('--trajectory', type=str, default=None,
                        help='Path to trajectory.npz file (for PR-Depth poses)')

    # Comparison mode
    parser.add_argument('--compare', action='store_true',
                        help='Compare multiple methods')
    parser.add_argument('--pr-depth', type=str, default=None,
                        help='PR-Depth depths.npz path')
    parser.add_argument('--pr-depth-traj', type=str, default=None,
                        help='PR-Depth trajectory.npz path')
    parser.add_argument('--vda', type=str, default=None,
                        help='VDA depths.npz path')
    parser.add_argument('--unidepth', type=str, default=None,
                        help='UniDepth depths.npz path')
    parser.add_argument('--dav2', type=str, default=None,
                        help='Depth Anything v2 depths.npz path')

    # Dataset options
    parser.add_argument('--dataset', '-d', type=str, default='kitti',
                        choices=['kitti'],
                        help='Dataset type')
    parser.add_argument('--date', type=str, default='2011_10_03',
                        help='KITTI date')
    parser.add_argument('--drive', type=str, default='0027',
                        help='KITTI drive')
    parser.add_argument('--max-depth', type=float, default=80.0,
                        help='Maximum depth for metrics')

    # Output
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output CSV path')

    args = parser.parse_args()

    # Load dataset
    print(f"\nLoading KITTI {args.date}/{args.drive}")
    paths = get_dataset_paths('kitti')
    dataset = KITTIEigenSplit(
        rgb_path=paths['rgb_path'],
        depth_path=paths['depth_path'],
        date=args.date,
        drive=args.drive,
    )
    print(f"Loaded {len(dataset)} frames")

    results = {}

    if args.compare:
        # Compare multiple methods
        methods_to_eval = []

        if args.pr_depth:
            methods_to_eval.append(('PR-Depth', args.pr_depth, args.pr_depth_traj, False))
        if args.vda:
            methods_to_eval.append(('VDA', args.vda, None, True))
        if args.unidepth:
            methods_to_eval.append(('UniDepth', args.unidepth, None, True))
        if args.dav2:
            methods_to_eval.append(('DAv2', args.dav2, None, True))

        for method_name, depths_path, traj_path, use_gt_pose in methods_to_eval:
            print(f"\n{'='*60}")
            print(f"Evaluating: {method_name}")
            print(f"{'='*60}")

            depths, frame_indices = load_depths(Path(depths_path))
            print(f"Loaded {len(depths)} depth maps")

            # Load poses if available
            poses = None
            baselines = None
            if traj_path:
                # For PR-Depth, use estimated poses; for others, use GT poses
                poses, baselines = load_poses(Path(traj_path), use_estimated=not use_gt_pose)

            result = evaluate_depths(
                depths=depths,
                frame_indices=frame_indices,
                dataset=dataset,
                poses=poses,
                baselines=baselines,
                use_gt_pose_for_tae=use_gt_pose,
                max_depth=args.max_depth,
            )
            results[method_name] = result

    elif args.method and args.depths:
        # Single method evaluation
        print(f"\n{'='*60}")
        print(f"Evaluating: {args.method}")
        print(f"{'='*60}")

        depths, frame_indices = load_depths(Path(args.depths))
        print(f"Loaded {len(depths)} depth maps")

        # Determine if this method uses GT poses for TAE
        use_gt_pose = args.method not in ['pr_depth']

        # Load poses if available
        poses = None
        baselines = None
        if args.trajectory:
            # For PR-Depth, use estimated poses; for others, use GT poses
            poses, baselines = load_poses(Path(args.trajectory), use_estimated=not use_gt_pose)

        result = evaluate_depths(
            depths=depths,
            frame_indices=frame_indices,
            dataset=dataset,
            poses=poses,
            baselines=baselines,
            use_gt_pose_for_tae=use_gt_pose,
            max_depth=args.max_depth,
        )
        results[args.method] = result

    else:
        parser.error("Either --method with --depths, or --compare with method paths required")

    # Print results
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")

    # Create comparison table
    metrics_order = ['AbsRel', 'd125', 'RMSE', 'MAE', 'TAE']
    header = f"{'Method':<15}"
    for m in metrics_order:
        header += f"{m:>10}"
    print(header)
    print("-" * len(header))

    for method_name, result in results.items():
        row = f"{method_name:<15}"
        for m in metrics_order:
            if m in result:
                if m in ['AbsRel', 'TAE']:
                    row += f"{result[m]:>10.4f}"
                elif m in ['d125', 'd115', 'd105']:
                    row += f"{result[m]:>10.1f}"
                else:
                    row += f"{result[m]:>10.3f}"
            else:
                row += f"{'N/A':>10}"
        print(row)

    # Save to CSV if requested
    if args.output:
        df = pd.DataFrame(results).T
        df.to_csv(args.output)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
