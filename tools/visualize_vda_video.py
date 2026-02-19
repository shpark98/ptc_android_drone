#!/usr/bin/env python3
"""Visualize VDA depth as video player in browser."""

import sys
from pathlib import Path
import numpy as np
import base64
from io import BytesIO
import json

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from configs import get_dataset_paths
from dataloader import KITTIEigenSplit


def depth_to_colormap(depth, vmin, vmax):
    """Convert depth to Spectral_r colormap."""
    import matplotlib.pyplot as plt

    depth_normalized = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
    # Use Spectral_r colormap
    cmap = plt.cm.Spectral_r
    depth_colored = (cmap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)
    # Convert RGB to BGR for cv2
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
    return depth_colored


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-frames', type=int, default=500, help='Max frames to load')
    parser.add_argument('--port', type=int, default=8050)
    args = parser.parse_args()

    # Paths
    stream_path = PROJECT_ROOT / "results/kitti_2011_10_03_0027/baselines/vda_stream/vitl_metric_streaming_depths.npz"
    offline_path = PROJECT_ROOT / "results/kitti_2011_10_03_0027/baselines/vda_offline/vitl_metric_offline_depths.npz"

    # Load dataset for RGB
    paths = get_dataset_paths('kitti')
    dataset = KITTIEigenSplit(
        rgb_path=paths['rgb_path'],
        depth_path=paths['depth_path'],
        date='2011_10_03',
        drive='0027',
    )

    print(f"Loading depths...")
    stream_data = np.load(str(stream_path), mmap_mode='r')
    offline_data = np.load(str(offline_path), mmap_mode='r')

    stream_depths = stream_data['depths']
    offline_depths = offline_data['depths']

    num_frames = min(args.max_frames, len(dataset), stream_depths.shape[0])
    print(f"Processing {num_frames} frames...")

    # Fixed depth range for consistency
    vmin, vmax = 0, 80

    # Encode frames as base64 JPEGs
    frames_data = []
    for idx in range(num_frames):
        if idx % 50 == 0:
            print(f"  Frame {idx}/{num_frames}")

        data = dataset.get(idx)
        if data is None:
            continue

        rgb = data['image_og']
        gt_depth = data.get('depth_og')
        stream_depth = stream_depths[idx]
        offline_depth = offline_depths[idx]

        # Apply colormap
        stream_colored = depth_to_colormap(stream_depth, vmin, vmax)
        offline_colored = depth_to_colormap(offline_depth, vmin, vmax)

        # Resize for faster loading (half size)
        h, w = rgb.shape[:2]
        new_w, new_h = w // 2, h // 2
        rgb_small = cv2.resize(rgb, (new_w, new_h))
        stream_small = cv2.resize(stream_colored, (new_w, new_h))
        offline_small = cv2.resize(offline_colored, (new_w, new_h))

        # GT depth
        if gt_depth is not None:
            gt_colored = depth_to_colormap(gt_depth, vmin, vmax)
            gt_colored[gt_depth == 0] = [0, 0, 0]  # Mark invalid as black
            gt_small = cv2.resize(gt_colored, (new_w, new_h))
        else:
            gt_small = np.zeros_like(rgb_small)

        # Stack: RGB | GT | Stream | Offline
        top_row = np.hstack([rgb_small, gt_small])
        bottom_row = np.hstack([stream_small, offline_small])
        combined = np.vstack([top_row, bottom_row])

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(buffer).decode('utf-8')
        frames_data.append(b64)

    print(f"Generating HTML...")

    # Generate HTML with video player
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>VDA Depth Video Player</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #1a1a1a;
            color: #fff;
        }}
        .container {{
            max-width: 1400px;
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
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>VDA Depth Comparison - KITTI 2011_10_03_0027</h1>
        <div class="legend">
            <div class="legend-item">Top-Left: RGB</div>
            <div class="legend-item">Top-Right: GT LiDAR</div>
            <div class="legend-item">Bottom-Left: VDA Streaming</div>
            <div class="legend-item">Bottom-Right: VDA Offline</div>
        </div>
        <div class="info">Depth range: 0-80m (Spectral_r colormap)</div>

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

    output_path = PROJECT_ROOT / "results/kitti_2011_10_03_0027/baselines/vda_video.html"
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
        print(f"Open: http://localhost:{args.port}/vda_video.html")
        print("Press Ctrl+C to stop")
        httpd.serve_forever()


if __name__ == '__main__':
    main()
