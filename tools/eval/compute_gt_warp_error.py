#!/usr/bin/env python3
"""Compute per-frame GT warp AbsRel for a dataset.

Warps previous GT depth to current frame using GT pose, compares with current GT depth.
High gt_warp_absrel → GT pose is unreliable for that frame pair.

Usage:
    # MS2 thermal
    python tools/eval/compute_gt_warp_error.py -d ms2 --ms2-timestamp 2021-08-13-21-18-04 --ms2-data-type thr

    # MS2 RGB
    python tools/eval/compute_gt_warp_error.py -d ms2 --ms2-timestamp 2021-08-13-21-18-04 --ms2-data-type rgb

    # KITTI
    python tools/eval/compute_gt_warp_error.py -d kitti --kitti-seq 09

    # Append to existing frame_metrics.csv
    python tools/eval/compute_gt_warp_error.py -d ms2 --ms2-timestamp 2021-08-13-21-18-04 --ms2-data-type thr \
        --append-to results/ms2__2021-08-13-21-18-04_thr/pr_depth/iter0/frame_metrics.csv

Output: CSV with columns [frame_idx, baseline, gt_warp_absrel]
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def compute_gt_warp_absrel(
    prev_depth: np.ndarray,
    curr_depth: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
) -> float:
    """Warp prev GT depth to current frame using GT pose, return AbsRel.

    Convention: p_curr = R_gt @ p_prev + t_gt

    Returns NaN if insufficient valid pixels.
    """
    H, W = prev_depth.shape[:2]

    valid_prev = (prev_depth > 0.1) & np.isfinite(prev_depth)
    if valid_prev.sum() < 100:
        return float('nan')

    yy, xx = np.where(valid_prev)
    z_prev = prev_depth[yy, xx].astype(np.float64)

    # Backproject to 3D
    X = (xx - cx) * z_prev / fx
    Y = (yy - cy) * z_prev / fy
    pts3d = np.stack([X, Y, z_prev], axis=1)  # (N, 3)

    # Transform
    pts_curr = (R_gt.astype(np.float64) @ pts3d.T).T + t_gt.astype(np.float64)
    Z_new = pts_curr[:, 2]

    # Project
    valid_z = Z_new > 0.1
    u_new = np.round((pts_curr[:, 0] * fx) / (Z_new + 1e-8) + cx).astype(int)
    v_new = np.round((pts_curr[:, 1] * fy) / (Z_new + 1e-8) + cy).astype(int)
    in_bounds = valid_z & (u_new >= 0) & (u_new < W) & (v_new >= 0) & (v_new < H)

    if in_bounds.sum() < 100:
        return float('nan')

    # Z-buffer (closest wins)
    z_warped = np.full((H, W), np.inf, dtype=np.float64)
    u_v = u_new[in_bounds]
    v_v = v_new[in_bounds]
    z_v = Z_new[in_bounds]
    order = np.argsort(-z_v)
    z_warped[v_v[order], u_v[order]] = z_v[order]

    # Compare
    compare = (z_warped < np.inf) & (curr_depth > 0.1) & np.isfinite(curr_depth)
    if compare.sum() < 100:
        return float('nan')

    abs_rel = float(np.mean(
        np.abs(z_warped[compare] - curr_depth[compare].astype(np.float64))
        / curr_depth[compare].astype(np.float64)
    ))
    return abs_rel


def load_dataset(args):
    """Load dataset based on arguments."""
    if args.dataset == 'ms2':
        from dataloader.dataset.ms2 import MS2Loader
        from configs import get_dataset_paths
        paths = get_dataset_paths('ms2')
        dataset = MS2Loader(
            dataset_path=paths['dataset_path'],
            timestamp=args.ms2_timestamp,
            data_type=args.ms2_data_type,
        )
        dataset_id = f"ms2__{args.ms2_timestamp}_{args.ms2_data_type}"
    elif args.dataset == 'kitti':
        from dataloader.dataset.kitti import KITTILoader
        from configs import get_dataset_paths
        paths = get_dataset_paths('kitti')
        dataset = KITTILoader(
            dataset_path=paths['dataset_path'],
            sequence=args.kitti_seq,
        )
        dataset_id = f"kitti_{args.kitti_seq}"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    return dataset, dataset_id


def main():
    parser = argparse.ArgumentParser(description="Compute per-frame GT warp AbsRel")
    parser.add_argument('-d', '--dataset', required=True, choices=['ms2', 'kitti'])
    parser.add_argument('--ms2-timestamp', type=str, default='2021-08-13-21-18-04')
    parser.add_argument('--ms2-data-type', type=str, default='thr', choices=['thr', 'rgb'])
    parser.add_argument('--kitti-seq', type=str, default='09')
    parser.add_argument('--max-frames', type=int, default=None, help='Max frames to process')
    parser.add_argument('--start-frame', type=int, default=0, help='Start frame index')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output CSV path')
    parser.add_argument('--append-to', type=str, default=None,
                        help='Append gt_warp_absrel column to existing frame_metrics.csv')
    args = parser.parse_args()

    dataset, dataset_id = load_dataset(args)
    fx, fy, cx, cy = dataset.get_intrinsics()
    n_frames = len(dataset)

    if args.max_frames:
        n_frames = min(n_frames, args.start_frame + args.max_frames)

    print(f"Dataset: {dataset_id}")
    print(f"Frames: {args.start_frame} to {n_frames-1} ({n_frames - args.start_frame} frames)")
    print(f"Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    results = []
    prev_gt_depth = None

    # Load initial prev frame if starting from > 0
    if args.start_frame > 0:
        data = dataset.get(args.start_frame - 1)
        if data is not None:
            prev_gt_depth = data.get('depth_og')

    for idx in tqdm(range(args.start_frame, n_frames), desc="Computing GT warp error"):
        data = dataset.get(idx)
        if data is None:
            results.append({'frame_idx': idx, 'baseline': 0.0, 'gt_warp_absrel': float('nan')})
            continue

        gt_depth = data.get('depth_og')
        baseline = dataset.get_baseline(idx)
        R_gt, t_gt = dataset.get_relative_pose(idx)

        gt_warp = float('nan')
        if prev_gt_depth is not None and gt_depth is not None and idx > 0:
            gt_warp = compute_gt_warp_absrel(
                prev_gt_depth, gt_depth, R_gt, t_gt, fx, fy, cx, cy
            )

        results.append({
            'frame_idx': idx,
            'baseline': baseline,
            'gt_warp_absrel': gt_warp,
        })

        prev_gt_depth = gt_depth

    df = pd.DataFrame(results)

    # Print summary
    valid = df['gt_warp_absrel'].dropna()
    print(f"\n{'='*60}")
    print(f"GT Warp AbsRel Summary ({len(valid)} valid frames)")
    print(f"{'='*60}")
    print(f"Mean:   {valid.mean():.4f}")
    print(f"Median: {valid.median():.4f}")
    print(f"Std:    {valid.std():.4f}")
    print(f"P90:    {valid.quantile(0.90):.4f}")
    print(f"P95:    {valid.quantile(0.95):.4f}")
    print(f"P99:    {valid.quantile(0.99):.4f}")
    print(f"Max:    {valid.max():.4f}")

    # Distribution
    thresholds = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
    print(f"\nFrames above threshold:")
    for thr in thresholds:
        n_above = (valid > thr).sum()
        pct = 100 * n_above / len(valid) if len(valid) > 0 else 0
        print(f"  > {thr:.2f}: {n_above:5d} ({pct:.1f}%)")

    # Output
    if args.append_to:
        # Append column to existing CSV
        existing = pd.read_csv(args.append_to)
        # Merge on frame_idx
        merged = existing.merge(df[['frame_idx', 'gt_warp_absrel']], on='frame_idx', how='left',
                                suffixes=('_old', ''))
        # If gt_warp_absrel_old exists, drop it
        if 'gt_warp_absrel_old' in merged.columns:
            merged = merged.drop(columns=['gt_warp_absrel_old'])
        merged.to_csv(args.append_to, index=False)
        print(f"\nAppended gt_warp_absrel to: {args.append_to}")
    else:
        out_path = args.output or f"gt_warp_error_{dataset_id}.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved to: {out_path}")


if __name__ == '__main__':
    main()
