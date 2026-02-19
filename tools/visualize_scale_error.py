#!/usr/bin/env python3
"""Visualize scale error between estimated depth and GT LiDAR.

Shows per-pixel scale ratio (z_pred / z_gt) with large colored points.
- Red: overestimation (z_pred > z_gt)
- White: accurate (z_pred ≈ z_gt)
- Blue: underestimation (z_pred < z_gt)

Usage:
    python tools/visualize_scale_error.py \
        --exp-dir results/kitti/pr_depth/2011_10_03_0027 \
        --date 2011_10_03 --drive 0027 \
        --start 190 --end 220 \
        --point-size 7 \
        --output figures/rotation_analysis
"""

import sys
from pathlib import Path
import numpy as np
import argparse

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from configs import get_dataset_paths
from dataloader import KITTIEigenSplit


def depth_to_colormap(depth, vmin=0, vmax=80):
    """Convert depth to Spectral_r colormap."""
    depth_normalized = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
    cmap = plt.cm.Spectral_r
    depth_colored = (cmap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)
    return depth_colored


def draw_scale_error_map(rgb, z_pred, z_gt, point_size=7, scale_min=0.5, scale_max=1.5):
    """Draw scale error map with large colored points at GT LiDAR locations.

    Args:
        rgb: RGB image (H, W, 3)
        z_pred: Predicted depth (H, W)
        z_gt: GT LiDAR depth (H, W), 0 = invalid
        point_size: Radius of points in pixels
        scale_min: Min scale ratio for colormap
        scale_max: Max scale ratio for colormap

    Returns:
        img: Visualization image with scale error overlay
        median_scale: Median scale ratio
        n_valid: Number of valid GT points
    """
    valid = (z_gt > 0) & (z_pred > 0) & np.isfinite(z_pred) & np.isfinite(z_gt)
    n_valid = valid.sum()

    if n_valid == 0:
        return rgb.copy(), np.nan, 0

    # Compute scale ratio at valid pixels
    scale_ratio = z_pred[valid] / z_gt[valid]
    median_scale = np.median(scale_ratio)

    # Colormap: RdBu_r centered at 1.0
    # Red (>1) = overestimation, Blue (<1) = underestimation, White (=1) = accurate
    norm = TwoSlopeNorm(vmin=scale_min, vcenter=1.0, vmax=scale_max)
    cmap = plt.cm.RdBu_r
    colors = cmap(norm(scale_ratio))[:, :3]  # RGB, 0-1

    # Draw points on image
    img = rgb.copy()
    v_coords, u_coords = np.where(valid)

    for i, (v, u) in enumerate(zip(v_coords, u_coords)):
        color = (colors[i] * 255).astype(np.uint8)
        # OpenCV uses BGR
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))
        cv2.circle(img, (u, v), point_size, color_bgr, -1)

    return img, median_scale, n_valid


def create_colorbar(scale_min=0.5, scale_max=1.5, height=30, width=300):
    """Create a horizontal colorbar image."""
    norm = TwoSlopeNorm(vmin=scale_min, vcenter=1.0, vmax=scale_max)
    cmap = plt.cm.RdBu_r

    # Create gradient
    gradient = np.linspace(scale_min, scale_max, width)
    gradient = np.tile(gradient, (height, 1))

    # Apply colormap
    colors = cmap(norm(gradient))[:, :, :3]
    colorbar = (colors * 255).astype(np.uint8)
    colorbar = cv2.cvtColor(colorbar, cv2.COLOR_RGB2BGR)

    return colorbar


def main():
    parser = argparse.ArgumentParser(description='Visualize scale error between depth estimation and GT')
    parser.add_argument('--exp-dir', type=str, required=True,
                        help='Path to experiment directory (containing depths.npz)')
    parser.add_argument('--date', type=str, default='2011_10_03', help='KITTI date')
    parser.add_argument('--drive', type=str, default='0027', help='KITTI drive')
    parser.add_argument('--start', type=int, default=0, help='Start frame index')
    parser.add_argument('--end', type=int, default=None, help='End frame index (exclusive)')
    parser.add_argument('--point-size', type=int, default=7, help='LiDAR point size in pixels')
    parser.add_argument('--scale-min', type=float, default=0.5, help='Min scale ratio for colormap')
    parser.add_argument('--scale-max', type=float, default=1.5, help='Max scale ratio for colormap')
    parser.add_argument('--output', type=str, default=None, help='Output directory for images')
    parser.add_argument('--save-video', action='store_true', help='Save as video instead of images')
    parser.add_argument('--depth-key', type=str, default='z_refined',
                        choices=['z_refined', 'z_tri'], help='Which depth to use')
    args = parser.parse_args()

    # Load dataset
    paths = get_dataset_paths('kitti')
    dataset = KITTIEigenSplit(
        rgb_path=paths['rgb_path'],
        depth_path=paths['depth_path'],
        date=args.date,
        drive=args.drive,
    )
    print(f"Loaded KITTI {args.date}_{args.drive}: {len(dataset)} frames")

    # Load depth predictions
    exp_dir = Path(args.exp_dir)
    if exp_dir.is_symlink():
        exp_dir = exp_dir.resolve()

    depths_path = exp_dir / 'depths.npz'
    if not depths_path.exists():
        print(f"Error: depths.npz not found in {exp_dir}")
        print("Run evaluate.py with --save-depths flag first")
        sys.exit(1)

    print(f"Loading depths from: {depths_path}")
    data = np.load(str(depths_path))

    # Get depths
    if args.depth_key in data:
        depths = data[args.depth_key]
    elif 'z_refined' in data:
        depths = data['z_refined']
    elif 'z_tri' in data:
        depths = data['z_tri']
    else:
        print(f"Error: No depth data found in {depths_path}")
        print(f"Available keys: {list(data.keys())}")
        sys.exit(1)

    frame_indices = data.get('frame_indices', np.arange(len(depths)))
    print(f"Loaded {len(depths)} depth maps, frame indices: {frame_indices[0]}-{frame_indices[-1]}")

    # Determine frame range
    start_idx = args.start
    end_idx = args.end if args.end is not None else len(depths)
    end_idx = min(end_idx, len(depths))

    print(f"Processing frames {start_idx} to {end_idx-1}")

    # Setup output
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = exp_dir / 'scale_error_analysis'
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create colorbar
    colorbar = create_colorbar(args.scale_min, args.scale_max, height=30, width=300)

    # Collect scale ratios for summary plot
    frame_numbers = []
    median_scales = []

    # Video writer
    video_writer = None

    for i in range(start_idx, end_idx):
        frame_idx = int(frame_indices[i])

        # Get data
        sample = dataset.get(frame_idx)
        if sample is None:
            print(f"Warning: Frame {frame_idx} not found in dataset")
            continue

        rgb = sample['image_og']
        gt_depth = sample.get('depth_og')
        pred_depth = depths[i]

        if gt_depth is None:
            print(f"Warning: No GT depth for frame {frame_idx}")
            continue

        # Resize pred_depth if necessary
        if pred_depth.shape != gt_depth.shape:
            pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]),
                                    interpolation=cv2.INTER_LINEAR)

        # Create visualizations
        # 1. Scale error map
        scale_map, median_scale, n_valid = draw_scale_error_map(
            rgb, pred_depth, gt_depth,
            point_size=args.point_size,
            scale_min=args.scale_min,
            scale_max=args.scale_max
        )

        # 2. Depth colormap
        depth_colored = depth_to_colormap(pred_depth, vmin=0, vmax=80)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)

        # 3. GT depth with large points (for comparison)
        gt_vis = rgb.copy()
        gt_valid = gt_depth > 0
        gt_colored_full = depth_to_colormap(gt_depth, vmin=0, vmax=80)
        gt_colored_full = cv2.cvtColor(gt_colored_full, cv2.COLOR_RGB2BGR)
        v_coords, u_coords = np.where(gt_valid)
        for v, u in zip(v_coords, u_coords):
            color = gt_colored_full[v, u].tolist()
            cv2.circle(gt_vis, (u, v), args.point_size, color, -1)

        # Store for summary
        frame_numbers.append(frame_idx)
        median_scales.append(median_scale)

        # Combine images
        h, w = rgb.shape[:2]

        # Row: RGB | Depth | Scale Error
        combined = np.hstack([rgb, depth_colored, scale_map])

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, 'RGB', (10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, f'{args.depth_key}', (w + 10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, 'Scale Error', (2*w + 10, 30), font, 0.8, (255, 255, 255), 2)

        # Add info bar at bottom
        info_bar = np.zeros((60, combined.shape[1], 3), dtype=np.uint8)
        info_text = f"Frame: {frame_idx}  |  Median Scale: {median_scale:.3f}  |  Valid pts: {n_valid}"
        cv2.putText(info_bar, info_text, (10, 25), font, 0.7, (255, 255, 255), 2)

        # Add colorbar
        cb_x = combined.shape[1] - 320
        info_bar[15:45, cb_x:cb_x+300] = colorbar
        cv2.putText(info_bar, f'{args.scale_min}', (cb_x - 40, 35), font, 0.5, (255, 255, 255), 1)
        cv2.putText(info_bar, f'{args.scale_max}', (cb_x + 305, 35), font, 0.5, (255, 255, 255), 1)
        cv2.putText(info_bar, '1.0', (cb_x + 135, 55), font, 0.5, (200, 200, 200), 1)

        combined = np.vstack([combined, info_bar])

        # Save or show
        if args.save_video:
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_path = output_dir / 'scale_error.mp4'
                video_writer = cv2.VideoWriter(str(video_path), fourcc, 10,
                                               (combined.shape[1], combined.shape[0]))
            video_writer.write(combined)
        else:
            # Save individual frame
            frame_path = output_dir / f'frame_{frame_idx:04d}.png'
            cv2.imwrite(str(frame_path), combined)

        if (i - start_idx) % 10 == 0:
            print(f"  Frame {frame_idx}: median_scale={median_scale:.3f}, valid={n_valid}")

    if video_writer is not None:
        video_writer.release()
        print(f"\nSaved video: {output_dir / 'scale_error.mp4'}")

    # Save summary plot
    if len(median_scales) > 0:
        plt.figure(figsize=(12, 4))
        plt.plot(frame_numbers, median_scales, 'b-', linewidth=1.5, marker='o', markersize=3)
        plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Ideal (1.0)')
        plt.xlabel('Frame Index')
        plt.ylabel('Median Scale Ratio (z_pred / z_gt)')
        plt.title(f'Scale Ratio over Time - KITTI {args.date}_{args.drive}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plot_path = output_dir / 'scale_ratio_over_time.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot: {plot_path}")

    print(f"\nOutput directory: {output_dir}")
    print(f"Total frames processed: {len(median_scales)}")
    if len(median_scales) > 0:
        print(f"Scale ratio range: {min(median_scales):.3f} - {max(median_scales):.3f}")


if __name__ == '__main__':
    main()
