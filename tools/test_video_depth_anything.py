#!/usr/bin/env python3
"""Test script for Video Depth Anything integration.

This script tests:
1. Streaming mode (metric and relative depth)
2. Offline mode (metric and relative depth)
3. Comparison between modes

Usage:
    python tools/test_video_depth_anything.py [--video PATH] [--frames N]
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_estimator_basic(encoder: str = 'vitl'):
    """Test basic estimator functionality."""
    print("\n" + "="*60)
    print("Testing VideoDepthAnythingEstimator")
    print("="*60)

    from src.estimators.depth import VideoDepthAnythingEstimator

    # Create a dummy image
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    configs = [
        {'metric': True, 'streaming': True, 'name': 'Metric + Streaming'},
        {'metric': True, 'streaming': False, 'name': 'Metric + Offline'},
        {'metric': False, 'streaming': True, 'name': 'Relative + Streaming'},
        {'metric': False, 'streaming': False, 'name': 'Relative + Offline'},
    ]

    for cfg in configs:
        print(f"\n--- {cfg['name']} ---")
        try:
            estimator = VideoDepthAnythingEstimator(
                encoder=encoder,
                metric=cfg['metric'],
                streaming=cfg['streaming'],
            )
            print(f"  Name: {estimator.name}")
            print(f"  Is Metric: {estimator.is_metric}")

            # Run inference
            start = time.time()
            depth = estimator.infer(dummy_img)
            elapsed = time.time() - start

            print(f"  Input shape: {dummy_img.shape}")
            print(f"  Output shape: {depth.shape}")
            print(f"  Depth range: [{depth.min():.4f}, {depth.max():.4f}]")
            print(f"  Inference time: {elapsed*1000:.1f}ms")
            print(f"  [OK] {cfg['name']} works!")

            # Reset for next test
            estimator.reset()
            del estimator

        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            import traceback
            traceback.print_exc()


def test_streaming_vs_offline(video_path: str = None, num_frames: int = 10, encoder: str = 'vitl'):
    """Compare streaming vs offline modes on real video frames."""
    print("\n" + "="*60)
    print("Comparing Streaming vs Offline Modes")
    print("="*60)

    from src.estimators.depth import VideoDepthAnythingEstimator

    # Generate synthetic video frames or load from file
    if video_path and Path(video_path).exists():
        print(f"Loading video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        print(f"Loaded {len(frames)} frames")
    else:
        print(f"Generating {num_frames} synthetic frames")
        frames = []
        for i in range(num_frames):
            # Create gradual change for temporal consistency test
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.circle(frame, (320 + i*10, 240), 50, (255, 255, 255), -1)
            cv2.rectangle(frame, (100, 100), (300, 300), (128, 128, 128), -1)
            frames.append(frame)

    # Test streaming mode
    print("\n--- Streaming Mode (Metric) ---")
    stream_estimator = VideoDepthAnythingEstimator(
        encoder=encoder,
        metric=True,
        streaming=True,
    )

    stream_depths = []
    stream_times = []
    for i, frame in enumerate(frames):
        start = time.time()
        depth = stream_estimator.infer(frame)
        elapsed = time.time() - start
        stream_depths.append(depth)
        stream_times.append(elapsed)
        print(f"  Frame {i}: depth range [{depth.min():.2f}, {depth.max():.2f}], time {elapsed*1000:.1f}ms")

    print(f"  Average time: {np.mean(stream_times)*1000:.1f}ms")
    print(f"  Total time: {sum(stream_times)*1000:.1f}ms")

    # Test offline mode (needs buffering)
    print("\n--- Offline Mode (Metric) ---")
    offline_estimator = VideoDepthAnythingEstimator(
        encoder=encoder,
        metric=True,
        streaming=False,
    )

    offline_depths = []
    offline_times = []
    for i, frame in enumerate(frames):
        start = time.time()
        depth = offline_estimator.infer(frame)
        elapsed = time.time() - start
        offline_depths.append(depth)
        offline_times.append(elapsed)
        print(f"  Frame {i}: depth range [{depth.min():.2f}, {depth.max():.2f}], time {elapsed*1000:.1f}ms")

    # Flush remaining
    remaining = offline_estimator.flush()
    if remaining:
        print(f"  Flushed {len(remaining)} remaining frames")

    print(f"  Average time: {np.mean(offline_times)*1000:.1f}ms")
    print(f"  Total time: {sum(offline_times)*1000:.1f}ms")

    # Compare depths
    print("\n--- Depth Comparison ---")
    for i in range(min(len(stream_depths), len(offline_depths))):
        diff = np.abs(stream_depths[i] - offline_depths[i])
        print(f"  Frame {i}: MAE={diff.mean():.4f}, Max diff={diff.max():.4f}")


def test_runner(encoder: str = 'vitl'):
    """Test the evaluation runner."""
    print("\n" + "="*60)
    print("Testing VideoDepthAnythingRunner")
    print("="*60)

    from src.evaluation.runners import VideoDepthAnythingRunner

    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Test depth-only runner
    print("\n--- Depth-Only Runner (Streaming) ---")
    runner = VideoDepthAnythingRunner(
        encoder=encoder,
        metric=True,
        streaming=True,
    )

    runner.initialize(H=480, W=640, fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    print(f"  Runner name: {runner.name}")
    print(f"  Is metric: {runner.is_metric}")
    print(f"  Requires GT baseline: {runner.requires_gt_baseline}")

    result = runner.process_frame(dummy_img, baseline=1.0)
    print(f"  Success: {result.success}")
    print(f"  Depth shape: {result.extra.get('depth', np.array([])).shape}")
    print(f"  [OK] Runner works!")

    runner.reset()


def test_pose_runner(encoder: str = 'vitl'):
    """Test the combined VDA + PR-Depth runner."""
    print("\n" + "="*60)
    print("Testing VideoDepthAnythingPoseRunner")
    print("="*60)

    try:
        from src.evaluation.runners import VideoDepthAnythingPoseRunner

        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        print("\n--- VDA + PR-Depth Runner ---")
        runner = VideoDepthAnythingPoseRunner(
            encoder=encoder,
            vda_metric=True,
            streaming=True,
        )

        runner.initialize(H=480, W=640, fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        print(f"  Runner name: {runner.name}")
        print(f"  Is metric: {runner.is_metric}")

        # Process multiple frames
        for i in range(3):
            result = runner.process_frame(dummy_img, baseline=1.0)
            print(f"  Frame {i}: success={result.success}, inliers={result.num_inliers}")

        print(f"  [OK] Pose runner works!")
        runner.reset()

    except ImportError as e:
        print(f"  [SKIP] PR-Depth C++ module not available: {e}")


def main():
    parser = argparse.ArgumentParser(description='Test Video Depth Anything integration')
    parser.add_argument('--video', type=str, help='Path to test video')
    parser.add_argument('--frames', type=int, default=10, help='Number of frames to test')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'],
                        help='Encoder type')
    parser.add_argument('--skip-basic', action='store_true', help='Skip basic tests')
    parser.add_argument('--skip-comparison', action='store_true', help='Skip streaming vs offline comparison')
    parser.add_argument('--skip-runner', action='store_true', help='Skip runner tests')

    args = parser.parse_args()

    print("Video Depth Anything Integration Test")
    print("="*60)

    if not args.skip_basic:
        test_estimator_basic(encoder=args.encoder)

    if not args.skip_comparison:
        test_streaming_vs_offline(
            video_path=args.video,
            num_frames=args.frames,
            encoder=args.encoder
        )

    if not args.skip_runner:
        test_runner(encoder=args.encoder)
        test_pose_runner(encoder=args.encoder)

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == '__main__':
    main()
