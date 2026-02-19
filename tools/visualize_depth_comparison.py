#!/usr/bin/env python3
"""Compare depth estimations from different methods as video player in browser.

Compares PR-Depth (z_refined) with VDA (streaming/offline) and GT.

Usage:
    python tools/visualize_depth_comparison.py --pr-depth-dir results/kitti_2011_10_03_0027/pr_depth/latest

    # Custom VDA paths
    python tools/visualize_depth_comparison.py \
        --pr-depth-dir results/kitti_2011_10_03_0027/pr_depth/v1 \
        --vda-stream results/kitti_2011_10_03_0027/baselines/vda_stream/vitl_metric_streaming_depths.npz \
        --vda-offline results/kitti_2011_10_03_0027/baselines/vda_offline/vitl_metric_offline_depths.npz
"""

import sys
from pathlib import Path
import numpy as np
import base64
import json
import argparse

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from configs import get_dataset_paths
from dataloader import KITTIEigenSplit


def depth_to_colormap(depth, vmin, vmax):
    """Convert depth to Spectral_r colormap."""
    import matplotlib.pyplot as plt

    depth_normalized = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
    cmap = plt.cm.Spectral_r
    depth_colored = (cmap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
    return depth_colored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pr-depth-dir', type=str, required=True,
                        help='Path to PR-Depth results directory (containing depths.npz)')
    parser.add_argument('--vda-stream', type=str, default=None,
                        help='Path to VDA streaming depths.npz')
    parser.add_argument('--vda-offline', type=str, default=None,
                        help='Path to VDA offline depths.npz')
    parser.add_argument('--max-frames', type=int, default=500, help='Max frames to load')
    parser.add_argument('--port', type=int, default=8050)
    parser.add_argument('--date', type=str, default='2011_10_03', help='KITTI date')
    parser.add_argument('--drive', type=str, default='0027', help='KITTI drive')
    args = parser.parse_args()

    # Paths
    pr_depth_dir = Path(args.pr_depth_dir)
    if pr_depth_dir.is_symlink():
        pr_depth_dir = pr_depth_dir.resolve()

    pr_depths_path = pr_depth_dir / 'depths.npz'
    if not pr_depths_path.exists():
        print(f"Error: depths.npz not found in {pr_depth_dir}")
        print("Run evaluate.py with --save-depths flag first")
        sys.exit(1)

    # Default VDA paths
    dataset_id = f"kitti_{args.date}_{args.drive}"
    baselines_dir = PROJECT_ROOT / "results" / dataset_id / "baselines"

    vda_stream_path = Path(args.vda_stream) if args.vda_stream else \
        baselines_dir / "vda_stream" / "vitl_metric_streaming_depths.npz"
    vda_offline_path = Path(args.vda_offline) if args.vda_offline else \
        baselines_dir / "vda_offline" / "vitl_metric_offline_depths.npz"

    # Load dataset for RGB and GT
    paths = get_dataset_paths('kitti')
    dataset = KITTIEigenSplit(
        rgb_path=paths['rgb_path'],
        depth_path=paths['depth_path'],
        date=args.date,
        drive=args.drive,
    )

    print(f"Loading PR-Depth depths from: {pr_depths_path}")
    pr_data = np.load(str(pr_depths_path))
    pr_depths = pr_data.get('z_refined', pr_data.get('z_tri'))
    pr_frame_indices = pr_data.get('frame_indices', np.arange(len(pr_depths)))
    print(f"  Shape: {pr_depths.shape}, frame indices: {pr_frame_indices[0]}-{pr_frame_indices[-1]}")

    # Load VDA depths if available
    vda_stream_depths = None
    vda_offline_depths = None

    if vda_stream_path.exists():
        print(f"Loading VDA streaming depths from: {vda_stream_path}")
        stream_data = np.load(str(vda_stream_path), mmap_mode='r')
        vda_stream_depths = stream_data['depths']
        print(f"  Shape: {vda_stream_depths.shape}")
    else:
        print(f"VDA streaming depths not found: {vda_stream_path}")

    if vda_offline_path.exists():
        print(f"Loading VDA offline depths from: {vda_offline_path}")
        offline_data = np.load(str(vda_offline_path), mmap_mode='r')
        vda_offline_depths = offline_data['depths']
        print(f"  Shape: {vda_offline_depths.shape}")
    else:
        print(f"VDA offline depths not found: {vda_offline_path}")

    num_frames = min(args.max_frames, len(pr_depths))
    print(f"Processing {num_frames} frames...")

    # Fixed depth range for consistency
    vmin, vmax = 0, 80

    # Encode frames as base64 JPEGs
    frames_data = []
    for i in range(num_frames):
        if i % 50 == 0:
            print(f"  Frame {i}/{num_frames}")

        # Get frame index from PR-Depth
        frame_idx = int(pr_frame_indices[i])
        data = dataset.get(frame_idx)
        if data is None:
            continue

        rgb = data['image_og']
        gt_depth = data.get('depth_og')
        pr_depth = pr_depths[i]

        # Get VDA depths for same frame index
        vda_stream = vda_stream_depths[frame_idx] if vda_stream_depths is not None and frame_idx < len(vda_stream_depths) else None
        vda_offline = vda_offline_depths[frame_idx] if vda_offline_depths is not None and frame_idx < len(vda_offline_depths) else None

        # Apply colormap
        pr_colored = depth_to_colormap(pr_depth, vmin, vmax)

        # Resize for faster loading (half size)
        h, w = rgb.shape[:2]
        new_w, new_h = w // 2, h // 2
        rgb_small = cv2.resize(rgb, (new_w, new_h))
        pr_small = cv2.resize(pr_colored, (new_w, new_h))

        # GT depth
        if gt_depth is not None:
            gt_colored = depth_to_colormap(gt_depth, vmin, vmax)
            gt_colored[gt_depth == 0] = [0, 0, 0]
            gt_small = cv2.resize(gt_colored, (new_w, new_h))
        else:
            gt_small = np.zeros_like(rgb_small)

        # VDA depths
        if vda_stream is not None:
            stream_colored = depth_to_colormap(vda_stream, vmin, vmax)
            stream_small = cv2.resize(stream_colored, (new_w, new_h))
        else:
            stream_small = np.zeros_like(rgb_small)

        if vda_offline is not None:
            offline_colored = depth_to_colormap(vda_offline, vmin, vmax)
            offline_small = cv2.resize(offline_colored, (new_w, new_h))
        else:
            offline_small = np.zeros_like(rgb_small)

        # Stack: Row1: RGB | GT | PR-Depth
        #        Row2: VDA Stream | VDA Offline | (empty or diff)
        top_row = np.hstack([rgb_small, gt_small, pr_small])
        bottom_row = np.hstack([stream_small, offline_small, np.zeros_like(rgb_small)])
        combined = np.vstack([top_row, bottom_row])

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, 'RGB', (10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, 'GT LiDAR', (new_w + 10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, 'PR-Depth', (2 * new_w + 10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, 'VDA Stream', (10, new_h + 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, 'VDA Offline', (new_w + 10, new_h + 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, f'Frame {frame_idx}', (2 * new_w + 10, new_h + 30), font, 0.8, (255, 255, 255), 2)

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(buffer).decode('utf-8')
        frames_data.append(b64)

    print(f"Generating HTML...")

    # Generate HTML with video player
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Depth Comparison Video Player</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #1a1a1a;
            color: #fff;
        }}
        .container {{
            max-width: 1800px;
            margin: 0 auto;
        }}
        #player {{
            text-align: center;
        }}
        #frame-img {{
            max-width: 100%;
            border: 2px solid #444;
        }}
        .controls {{
            margin: 20px 0;
            display: flex;
            gap: 10px;
            align-items: center;
            justify-content: center;
        }}
        button {{
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            background: #333;
            color: #fff;
            border: 1px solid #555;
            border-radius: 5px;
        }}
        button:hover {{
            background: #444;
        }}
        #slider {{
            width: 400px;
        }}
        .info {{
            text-align: center;
            margin: 10px 0;
            font-size: 14px;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin: 10px 0;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Depth Comparison - KITTI {args.date}_{args.drive}</h1>
        <div class="legend">
            <div>Top: RGB | GT LiDAR | PR-Depth (refined)</div>
            <div>Bottom: VDA Streaming | VDA Offline</div>
        </div>
        <div class="info">Depth range: 0-80m (Spectral_r colormap)</div>
        <div class="info">PR-Depth source: {pr_depth_dir.name}</div>

        <div id="player">
            <img id="frame-img" src="">
        </div>

        <div class="controls">
            <button id="prev-btn">&lt; Prev</button>
            <button id="play-btn">Play</button>
            <button id="next-btn">Next &gt;</button>
            <input type="range" id="slider" min="0" max="{num_frames-1}" value="0">
            <span id="frame-info">Frame: 0 / {num_frames-1}</span>
        </div>

        <div class="controls">
            <label>Speed: </label>
            <select id="speed">
                <option value="100">10 FPS</option>
                <option value="66">15 FPS</option>
                <option value="33" selected>30 FPS</option>
                <option value="16">60 FPS</option>
            </select>
        </div>
    </div>

    <script>
        const frames = {json.dumps(frames_data)};
        let currentFrame = 0;
        let playing = false;
        let intervalId = null;

        const img = document.getElementById('frame-img');
        const slider = document.getElementById('slider');
        const frameInfo = document.getElementById('frame-info');
        const playBtn = document.getElementById('play-btn');
        const speedSelect = document.getElementById('speed');

        function showFrame(idx) {{
            currentFrame = idx;
            img.src = 'data:image/jpeg;base64,' + frames[idx];
            slider.value = idx;
            frameInfo.textContent = 'Frame: ' + idx + ' / ' + (frames.length - 1);
        }}

        function play() {{
            if (playing) {{
                clearInterval(intervalId);
                playBtn.textContent = 'Play';
                playing = false;
            }} else {{
                const speed = parseInt(speedSelect.value);
                intervalId = setInterval(() => {{
                    currentFrame = (currentFrame + 1) % frames.length;
                    showFrame(currentFrame);
                }}, speed);
                playBtn.textContent = 'Pause';
                playing = true;
            }}
        }}

        playBtn.onclick = play;
        document.getElementById('prev-btn').onclick = () => showFrame(Math.max(0, currentFrame - 1));
        document.getElementById('next-btn').onclick = () => showFrame(Math.min(frames.length - 1, currentFrame + 1));
        slider.oninput = () => showFrame(parseInt(slider.value));

        speedSelect.onchange = () => {{
            if (playing) {{
                play();
                play();
            }}
        }};

        // Keyboard controls
        document.onkeydown = (e) => {{
            if (e.key === ' ') {{ play(); e.preventDefault(); }}
            if (e.key === 'ArrowLeft') showFrame(Math.max(0, currentFrame - 1));
            if (e.key === 'ArrowRight') showFrame(Math.min(frames.length - 1, currentFrame + 1));
        }};

        // Show first frame
        showFrame(0);
    </script>
</body>
</html>"""

    output_path = pr_depth_dir / "depth_comparison.html"
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"\nSaved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"\nStarting server on port {args.port}...")

    import http.server
    import socketserver
    import os

    os.chdir(output_path.parent)

    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", args.port), handler) as httpd:
        print(f"Open: http://localhost:{args.port}/depth_comparison.html")
        print("Press Ctrl+C to stop")
        httpd.serve_forever()


if __name__ == '__main__':
    main()
