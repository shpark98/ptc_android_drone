#!/usr/bin/env python3
"""Compute summary table with TAE and depth metrics for all methods."""

import sys
sys.path.insert(0, '/home/arrl_server2/workspace/pr_depth')

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from configs import get_dataset_paths
from dataloader import KITTIEigenSplit


def get_eigen_crop(H, W):
    """Get Eigen crop boundaries for KITTI."""
    y_min = int(0.40810811 * H)
    y_max = int(0.99189189 * H)
    x_min = int(0.03594771 * W)
    x_max = int(0.96405229 * W)
    return y_min, y_max, x_min, x_max


def compute_depth_metrics(pred, gt, max_depth=80.0):
    """Compute depth metrics with Eigen crop."""
    H, W = gt.shape
    y_min, y_max, x_min, x_max = get_eigen_crop(H, W)

    pred_crop = pred[y_min:y_max, x_min:x_max]
    gt_crop = gt[y_min:y_max, x_min:x_max]

    # Cap pred to operating range
    pred_crop = np.clip(pred_crop, 0, max_depth)

    mask = (gt_crop > 0) & (gt_crop < max_depth) & np.isfinite(pred_crop) & (pred_crop > 0)

    if mask.sum() < 100:
        return None

    pred_valid = pred_crop[mask]
    gt_valid = gt_crop[mask]

    abs_rel = float(np.mean(np.abs(pred_valid - gt_valid) / gt_valid))
    ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    d105 = float(np.mean(ratio < 1.05) * 100)
    d115 = float(np.mean(ratio < 1.15) * 100)
    d125 = float(np.mean(ratio < 1.25) * 100)

    return {'AbsRel': abs_rel, 'd105': d105, 'd115': d115, 'd125': d125}


def compute_tae(depth1, depth2, R, t, fx, fy, cx, cy, device, max_depth=80.0):
    """Compute TAE for a frame pair."""
    H, W = depth1.shape

    d1 = torch.from_numpy(depth1.astype(np.float32)).to(device)
    d2 = torch.from_numpy(depth2.astype(np.float32)).to(device)
    R_t = torch.from_numpy(R.astype(np.float32)).to(device)
    t_vec = torch.from_numpy(t.astype(np.float32)).to(device)

    def warp_and_compare(src_depth, tgt_depth, R_mat, t_vec):
        xx, yy = torch.meshgrid(torch.arange(W, device=device),
                                 torch.arange(H, device=device), indexing='xy')
        xx, yy = xx.float(), yy.float()

        X = (xx - cx) * src_depth / fx
        Y = (yy - cy) * src_depth / fy
        Z = src_depth
        points3d = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)

        points3d_transformed = torch.matmul(points3d, R_mat.T) + t_vec
        X_new, Y_new, Z_new = points3d_transformed[:, 0], points3d_transformed[:, 1], points3d_transformed[:, 2]

        u_new = (X_new * fx) / (Z_new + 1e-8) + cx
        v_new = (Y_new * fy) / (Z_new + 1e-8) + cy

        u_int = torch.round(u_new).long()
        v_int = torch.round(v_new).long()

        valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H) & \
                (Z_new > 0.01) & (Z_new < max_depth)

        if valid.sum() == 0:
            return None

        depth_proj = torch.zeros((H, W), dtype=src_depth.dtype, device=device)
        depth_proj[v_int[valid], u_int[valid]] = Z_new[valid]

        compare_mask = (depth_proj > 0.01) & (depth_proj < max_depth) & \
                       (tgt_depth > 0.01) & (tgt_depth < max_depth)
        if compare_mask.sum() < 100:
            return None

        abs_rel = torch.mean(torch.abs(depth_proj[compare_mask] - tgt_depth[compare_mask]) / tgt_depth[compare_mask])
        return abs_rel.item()

    err_fwd = warp_and_compare(d1, d2, R_t, t_vec)
    R_inv = R_t.T
    t_inv = -torch.matmul(R_t.T, t_vec)
    err_bwd = warp_and_compare(d2, d1, R_inv, t_inv)

    if err_fwd is None and err_bwd is None:
        return None
    elif err_fwd is None:
        return err_bwd * 100
    elif err_bwd is None:
        return err_fwd * 100
    else:
        return (err_fwd + err_bwd) / 2 * 100


def process_sequence(seq_info, method_names, device):
    """Process a single sequence and return per-method metrics."""
    results_dir = seq_info['results_dir']
    date = seq_info['date']
    drive = seq_info['drive']

    paths = get_dataset_paths('kitti')
    dataset = KITTIEigenSplit(
        rgb_path=paths['rgb_path'],
        depth_path=paths['depth_path'],
        date=date,
        drive=drive,
    )
    fx, fy, cx, cy = dataset.get_intrinsics()

    # Load methods
    methods = {}
    for name in method_names:
        npz_path = results_dir / f"{name}_depths.npz"
        if npz_path.exists():
            data = np.load(str(npz_path), mmap_mode='r')
            methods[name] = {
                'depths': data['depths'],
                'frame_indices': np.array(data['frame_indices']),
            }

    results = {}
    for method_name, method_data in methods.items():
        depths = method_data['depths']
        frame_indices = method_data['frame_indices']

        # Depth metrics
        abs_rel_list, d105_list, d115_list, d125_list = [], [], [], []
        for i in range(len(depths)):
            try:
                gt_depth = dataset.get(i)['depth_og']
                if gt_depth is None:
                    continue
                gt_depth = np.array(gt_depth)
                pred_depth = np.array(depths[i])

                if gt_depth.shape != pred_depth.shape:
                    continue

                metrics = compute_depth_metrics(pred_depth, gt_depth)
                if metrics:
                    abs_rel_list.append(metrics['AbsRel'])
                    d105_list.append(metrics['d105'])
                    d115_list.append(metrics['d115'])
                    d125_list.append(metrics['d125'])
            except:
                continue

        # TAE
        tae_list = []
        for i in range(len(depths) - 1):
            idx_curr = int(frame_indices[i])
            idx_next = int(frame_indices[i + 1])

            if idx_next != idx_curr + 1:
                continue

            try:
                R_gt, t_gt = dataset.get_relative_pose(idx_next)
                baseline = dataset.get_baseline(idx_next)

                if baseline < 0.01:
                    continue

                d_curr = np.array(depths[i])
                d_next = np.array(depths[i + 1])

                tae = compute_tae(d_curr, d_next, R_gt, t_gt, fx, fy, cx, cy, device)
                if tae is not None:
                    tae_list.append(tae)
            except:
                continue

        results[method_name] = {
            'abs_rel_list': abs_rel_list,
            'd105_list': d105_list,
            'd115_list': d115_list,
            'd125_list': d125_list,
            'tae_list': tae_list,
            'num_frames': len(depths),
        }

    return results, len(dataset)


def main():
    RESULTS_ROOT = Path('/home/arrl_server2/workspace/pr_depth/results')
    METHOD_NAMES = ['pr_depth', 'pr_depth_wo_fusion', 'unidepth', 'vda']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Find all KITTI sequences with depth_comparison (all 4 methods required)
    sequences = []
    for d in sorted(RESULTS_ROOT.glob('kitti_*')):
        depth_comp_dir = d / 'depth_comparison'
        if depth_comp_dir.exists():
            # Check if all 4 methods exist
            has_all = all((depth_comp_dir / f"{m}_depths.npz").exists() for m in METHOD_NAMES)
            if has_all:
                parts = d.name.split('_')
                if len(parts) >= 5:
                    date = '_'.join(parts[1:4])
                    drive = parts[4]
                    sequences.append({
                        'name': d.name,
                        'date': date,
                        'drive': drive,
                        'results_dir': depth_comp_dir,
                    })

    print(f"Found {len(sequences)} complete sequences:")
    for seq in sequences:
        print(f"  - {seq['name']}")

    if not sequences:
        print("No complete sequences found!")
        return

    # Aggregate metrics across all sequences
    all_metrics = {m: {'abs_rel': [], 'd105': [], 'd115': [], 'd125': [], 'tae': []} for m in METHOD_NAMES}
    per_seq_results = []

    for seq in sequences:
        print(f"\n{'='*60}")
        print(f"Processing {seq['name']}...")
        print(f"{'='*60}")

        results, num_frames = process_sequence(seq, METHOD_NAMES, device)

        seq_summary = {'sequence': seq['name'], 'frames': num_frames}

        for method_name in METHOD_NAMES:
            if method_name in results:
                r = results[method_name]
                if r['abs_rel_list']:
                    all_metrics[method_name]['abs_rel'].extend(r['abs_rel_list'])
                    all_metrics[method_name]['d105'].extend(r['d105_list'])
                    all_metrics[method_name]['d115'].extend(r['d115_list'])
                    all_metrics[method_name]['d125'].extend(r['d125_list'])
                    seq_summary[f'{method_name}_AbsRel'] = np.mean(r['abs_rel_list'])
                    seq_summary[f'{method_name}_d125'] = np.mean(r['d125_list'])
                if r['tae_list']:
                    all_metrics[method_name]['tae'].extend(r['tae_list'])
                    seq_summary[f'{method_name}_TAE'] = np.mean(r['tae_list'])

        per_seq_results.append(seq_summary)

    # Print per-sequence summary
    print("\n" + "=" * 100)
    print("Per-Sequence Results")
    print("=" * 100)

    for seq_res in per_seq_results:
        print(f"\n{seq_res['sequence']} ({seq_res['frames']} frames):")
        print(f"  {'Method':<25} {'AbsRel':>10} {'δ<1.25':>10} {'TAE':>10}")
        print(f"  {'-'*55}")
        for m in METHOD_NAMES:
            abs_rel = seq_res.get(f'{m}_AbsRel', None)
            d125 = seq_res.get(f'{m}_d125', None)
            tae = seq_res.get(f'{m}_TAE', None)
            abs_rel_s = f"{abs_rel:.4f}" if abs_rel else "N/A"
            d125_s = f"{d125:.2f}%" if d125 else "N/A"
            tae_s = f"{tae:.2f}%" if tae else "N/A"
            print(f"  {m:<25} {abs_rel_s:>10} {d125_s:>10} {tae_s:>10}")

    # Print aggregate summary
    print("\n" + "=" * 100)
    print(f"AGGREGATE Summary ({len(sequences)} sequences, Eigen Crop, max_depth=80m)")
    print("=" * 100)
    print(f"\n{'Method':<25} {'AbsRel':>10} {'δ<1.05':>10} {'δ<1.15':>10} {'δ<1.25':>10} {'TAE':>10}")
    print("-" * 85)

    for method_name in METHOD_NAMES:
        m = all_metrics[method_name]
        abs_rel = f"{np.mean(m['abs_rel']):.4f}" if m['abs_rel'] else "N/A"
        d105 = f"{np.mean(m['d105']):.2f}%" if m['d105'] else "N/A"
        d115 = f"{np.mean(m['d115']):.2f}%" if m['d115'] else "N/A"
        d125 = f"{np.mean(m['d125']):.2f}%" if m['d125'] else "N/A"
        tae = f"{np.mean(m['tae']):.2f}%" if m['tae'] else "N/A"
        print(f"{method_name:<25} {abs_rel:>10} {d105:>10} {d115:>10} {d125:>10} {tae:>10}")

    print("=" * 85)

    # Total frames
    total_frames = sum(len(all_metrics[METHOD_NAMES[0]]['abs_rel']) for _ in [1])
    print(f"\nTotal frames evaluated: {len(all_metrics[METHOD_NAMES[0]]['abs_rel'])}")


if __name__ == '__main__':
    main()
