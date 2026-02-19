#!/usr/bin/env python3
"""Visualize optical flow comparison: DIS vs RAFT vs NeuFlow vs GT."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'cpp' / 'build'))

import numpy as np
import cv2
import flow_vis
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from configs import get_dataset_paths
from dataloader import TartanairLoader


def compute_dis_flow(img0: np.ndarray, img1: np.ndarray) -> np.ndarray:
    """Compute DIS optical flow."""
    # Convert to grayscale
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # Create DIS flow
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(gray0, gray1, None)
    return flow


def main():
    # Load TartanAir dataset
    paths = get_dataset_paths('tartanair')
    dataset = TartanairLoader(
        dataset_path=paths['dataset_path'],
        scene='seasonsforest',
        level='Easy',
        num=1,
    )
    print(f"Loaded {len(dataset)} frames")

    # Get flow estimators
    from src.evaluation.flow_estimators import get_flow_estimator
    raft_estimator = get_flow_estimator('raft', device='cuda')
    neuflow_estimator = get_flow_estimator('neuflow', device='cuda')

    # Select frames to visualize (with good motion)
    frame_indices = [50, 100, 150]  # Different frames

    for frame_idx in frame_indices:
        print(f"\nProcessing frame {frame_idx}...")

        # Get consecutive frames
        data0 = dataset.get(frame_idx - 1)
        data1 = dataset.get(frame_idx)

        if data0 is None or data1 is None:
            print(f"  Skipping frame {frame_idx}")
            continue

        img0 = data0['image_og']
        img1 = data1['image_og']
        gt_flow = data1.get('flow')

        # Compute flows
        print("  Computing DIS flow...")
        dis_flow = compute_dis_flow(img0, img1)

        print("  Computing RAFT flow...")
        raft_flow = raft_estimator.estimate(img0, img1)

        print("  Computing NeuFlow flow...")
        neuflow_flow = neuflow_estimator.estimate(img0, img1)

        # Visualize using flow_vis
        dis_vis = flow_vis.flow_to_color(dis_flow)
        raft_vis = flow_vis.flow_to_color(raft_flow)
        neuflow_vis = flow_vis.flow_to_color(neuflow_flow)

        if gt_flow is not None:
            gt_vis = flow_vis.flow_to_color(gt_flow)
        else:
            gt_vis = np.zeros_like(dis_vis)

        # Compute flow magnitude statistics
        def flow_stats(flow, name):
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            return f"{name}: mean={mag.mean():.2f}, max={mag.max():.2f}"

        print(f"  {flow_stats(dis_flow, 'DIS')}")
        print(f"  {flow_stats(raft_flow, 'RAFT')}")
        print(f"  {flow_stats(neuflow_flow, 'NeuFlow')}")
        if gt_flow is not None:
            print(f"  {flow_stats(gt_flow, 'GT')}")

        # Create visualization
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 4, figure=fig, hspace=0.15, wspace=0.05)

        # Row 1: Images and GT
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
        ax.set_title('Frame t-1', fontsize=14)
        ax.axis('off')

        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        ax.set_title('Frame t', fontsize=14)
        ax.axis('off')

        ax = fig.add_subplot(gs[0, 2])
        ax.imshow(gt_vis)
        ax.set_title('GT Flow', fontsize=14, fontweight='bold', color='green')
        ax.axis('off')

        # Placeholder for colorbar reference
        ax = fig.add_subplot(gs[0, 3])
        ax.axis('off')
        ax.text(0.5, 0.5, 'Flow Color Wheel\n(Hue = Direction\nSaturation = Magnitude)',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)

        # Row 2: Different flow methods
        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(dis_vis)
        ax.set_title('DIS Flow (Classical)', fontsize=14, fontweight='bold', color='blue')
        ax.axis('off')

        ax = fig.add_subplot(gs[1, 1])
        ax.imshow(raft_vis)
        ax.set_title('RAFT Flow (Learned)', fontsize=14, fontweight='bold', color='orange')
        ax.axis('off')

        ax = fig.add_subplot(gs[1, 2])
        ax.imshow(neuflow_vis)
        ax.set_title('NeuFlow_v2 (Learned)', fontsize=14, fontweight='bold', color='purple')
        ax.axis('off')

        # Error maps (if GT available)
        if gt_flow is not None:
            ax = fig.add_subplot(gs[1, 3])

            # Compute EPE (End Point Error) for each method
            def compute_epe(pred, gt):
                diff = pred - gt
                return np.sqrt(diff[..., 0]**2 + diff[..., 1]**2)

            dis_epe = compute_epe(dis_flow, gt_flow)
            raft_epe = compute_epe(raft_flow, gt_flow)
            neuflow_epe = compute_epe(neuflow_flow, gt_flow)

            # Show comparison bar chart
            methods = ['DIS', 'RAFT', 'NeuFlow']
            mean_epes = [dis_epe.mean(), raft_epe.mean(), neuflow_epe.mean()]
            colors = ['blue', 'orange', 'purple']

            bars = ax.bar(methods, mean_epes, color=colors, alpha=0.7)
            ax.set_ylabel('Mean EPE (pixels)', fontsize=12)
            ax.set_title('Flow Error (EPE)', fontsize=14, fontweight='bold')

            # Add values on bars
            for bar, val in zip(bars, mean_epes):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')

            ax.set_ylim(0, max(mean_epes) * 1.3)

        # Main title
        fig.suptitle(f'Optical Flow Comparison - TartanAir seasonsforest Frame {frame_idx}',
                    fontsize=16, fontweight='bold', y=0.98)

        # Save
        output_path = PROJECT_ROOT / 'results' / f'flow_comparison_frame{frame_idx}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {output_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
