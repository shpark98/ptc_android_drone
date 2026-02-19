#!/usr/bin/env python3
"""Visualize VDA depth results as HTML for SSH environment."""

import sys
from pathlib import Path
import numpy as np
import base64
from io import BytesIO

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from configs import get_dataset_paths
from dataloader import KITTIEigenSplit


def depth_to_colormap(depth, vmin=None, vmax=None):
    """Convert depth to colored image."""
    if vmin is None:
        vmin = np.percentile(depth[depth > 0], 5) if (depth > 0).any() else 0
    if vmax is None:
        vmax = np.percentile(depth[depth > 0], 95) if (depth > 0).any() else 1

    depth_normalized = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
    depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return depth_colored


def img_to_base64(img):
    """Convert numpy image to base64 string."""
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


def main():
    # Paths
    stream_path = PROJECT_ROOT / "results/kitti_2011_10_03_0027/baselines/vda_stream/vitl_metric_streaming_depths.npz"
    offline_path = PROJECT_ROOT / "results/kitti_2011_10_03_0027/baselines/vda_offline/vitl_metric_offline_depths.npz"

    # Load dataset for RGB and GT
    paths = get_dataset_paths('kitti')
    dataset = KITTIEigenSplit(
        rgb_path=paths['rgb_path'],
        depth_path=paths['depth_path'],
        date='2011_10_03',
        drive='0027',
    )

    # Sample frames
    sample_indices = [0, 50, 100, 200, 500, 1000, 2000, 3000]
    sample_indices = [i for i in sample_indices if i < len(dataset)]

    print(f"Loading depths (this may take a while due to 7GB files)...")

    # Load depths using memory mapping
    stream_data = np.load(str(stream_path), mmap_mode='r')
    offline_data = np.load(str(offline_path), mmap_mode='r')

    stream_depths = stream_data['depths']
    offline_depths = offline_data['depths']

    print(f"Stream depths shape: {stream_depths.shape}")
    print(f"Offline depths shape: {offline_depths.shape}")

    # Build HTML
    html_parts = ["""
<!DOCTYPE html>
<html>
<head>
    <title>VDA Depth Visualization</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
        .frame { margin-bottom: 40px; border: 1px solid #444; padding: 20px; }
        .frame h2 { margin-top: 0; }
        .row { display: flex; gap: 10px; margin-bottom: 10px; }
        .col { flex: 1; }
        .col img { width: 100%; height: auto; }
        .col p { margin: 5px 0; font-size: 12px; text-align: center; }
        .stats { background: #333; padding: 10px; border-radius: 5px; font-size: 12px; }
    </style>
</head>
<body>
    <h1>VDA Depth Comparison: Streaming vs Offline</h1>
    <p>Dataset: KITTI 2011_10_03_0027</p>
"""]

    for idx in sample_indices:
        print(f"Processing frame {idx}...")

        # Get data
        data = dataset.get(idx)
        if data is None:
            continue

        rgb = data['image_og']
        gt_depth = data.get('depth_og')

        stream_depth = stream_depths[idx]
        offline_depth = offline_depths[idx]

        # Compute stats
        if gt_depth is not None:
            valid_mask = gt_depth > 0

            # Stream metrics
            stream_mae = np.mean(np.abs(stream_depth[valid_mask] - gt_depth[valid_mask]))
            stream_rmse = np.sqrt(np.mean((stream_depth[valid_mask] - gt_depth[valid_mask])**2))

            # Offline metrics
            offline_mae = np.mean(np.abs(offline_depth[valid_mask] - gt_depth[valid_mask]))
            offline_rmse = np.sqrt(np.mean((offline_depth[valid_mask] - gt_depth[valid_mask])**2))

            stats_html = f"""
            <div class="stats">
                <b>Streaming:</b> MAE={stream_mae:.2f}m, RMSE={stream_rmse:.2f}m |
                <b>Offline:</b> MAE={offline_mae:.2f}m, RMSE={offline_rmse:.2f}m |
                <b>Depth range:</b> Stream [{stream_depth.min():.1f}, {stream_depth.max():.1f}],
                Offline [{offline_depth.min():.1f}, {offline_depth.max():.1f}],
                GT [{gt_depth[valid_mask].min():.1f}, {gt_depth[valid_mask].max():.1f}]
            </div>
            """
        else:
            stats_html = "<div class='stats'>No GT available</div>"

        # Use same colormap range for fair comparison
        vmin = 0
        vmax = 80  # KITTI max depth

        # Convert to colormaps
        rgb_b64 = img_to_base64(rgb)
        stream_colored = depth_to_colormap(stream_depth, vmin, vmax)
        offline_colored = depth_to_colormap(offline_depth, vmin, vmax)

        stream_b64 = img_to_base64(stream_colored)
        offline_b64 = img_to_base64(offline_colored)

        gt_b64 = ""
        if gt_depth is not None:
            gt_colored = depth_to_colormap(gt_depth, vmin, vmax)
            # Mark invalid pixels
            gt_colored[gt_depth == 0] = [0, 0, 0]
            gt_b64 = img_to_base64(gt_colored)

        html_parts.append(f"""
    <div class="frame">
        <h2>Frame {idx}</h2>
        {stats_html}
        <div class="row">
            <div class="col">
                <p>RGB</p>
                <img src="data:image/jpeg;base64,{rgb_b64}">
            </div>
            <div class="col">
                <p>GT Depth (LiDAR)</p>
                <img src="data:image/jpeg;base64,{gt_b64 if gt_b64 else ''}">
            </div>
        </div>
        <div class="row">
            <div class="col">
                <p>VDA Streaming</p>
                <img src="data:image/jpeg;base64,{stream_b64}">
            </div>
            <div class="col">
                <p>VDA Offline</p>
                <img src="data:image/jpeg;base64,{offline_b64}">
            </div>
        </div>
    </div>
""")

    html_parts.append("""
</body>
</html>
""")

    # Save HTML
    output_path = PROJECT_ROOT / "results/kitti_2011_10_03_0027/baselines/vda_comparison.html"
    with open(output_path, 'w') as f:
        f.write(''.join(html_parts))

    print(f"\nSaved to: {output_path}")
    print("Open in browser or use: python -m http.server 8000")


if __name__ == '__main__':
    main()
