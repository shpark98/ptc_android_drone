#!/usr/bin/env python3
"""Video Depth Anything evaluation script.

Supports both streaming and offline modes:
- Streaming: frame-by-frame processing (real-time capable)
- Offline: batch processing entire video (higher accuracy, official eval method)

Usage:
    # Streaming mode (default)
    python tools/eval/evaluate_vda.py --encoder vitl --metric

    # Offline mode (batch processing) - official evaluation method
    python tools/eval/evaluate_vda.py --encoder vitl --metric --offline

    # Compare streaming vs offline
    python tools/eval/evaluate_vda.py --encoder vitl --metric --compare

    # Different dataset
    python tools/eval/evaluate_vda.py -d tartanair --ta-scene abandonedfactory
    python tools/eval/evaluate_vda.py -d wheel -w outdoor
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add VDA to path
VDA_PATH = PROJECT_ROOT / "external" / "video_depth_anything"
sys.path.insert(0, str(VDA_PATH))

import torch
from video_depth_anything.video_depth import VideoDepthAnything as VDAOffline
from video_depth_anything.video_depth_stream import VideoDepthAnything as VDAStreaming

from configs import get_dataset_paths
from dataloader import KITTIEigenSplit, WheelLoader, WheelEvalWrapper, TartanairLoader, MS2Loader
from src.evaluation.metrics import compute_depth_metrics


# Model configurations
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

# Wheel dataset name mapping
WHEEL_DATASETS = {
    'indoor': '25_10_20_14_50',
    'outdoor': '25_10_20_14_30',
    'forest': '25_11_04_16_00',
}


def load_model(encoder: str, metric: bool, streaming: bool, device: str):
    """Load VDA model."""
    checkpoint_dir = VDA_PATH / "checkpoints"
    checkpoint_name = 'metric_video_depth_anything' if metric else 'video_depth_anything'
    checkpoint_path = checkpoint_dir / f"{checkpoint_name}_{encoder}.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = MODEL_CONFIGS[encoder].copy()
    if not streaming:
        config['metric'] = metric

    if streaming:
        model = VDAStreaming(**config)
    else:
        model = VDAOffline(**config)

    model.load_state_dict(torch.load(str(checkpoint_path), map_location='cpu'), strict=True)
    model = model.to(device).eval()

    return model


def evaluate_streaming(model, dataset, device, input_size=518, fp32=False,
                       start_frame=0, max_frames=None):
    """Evaluate using streaming mode (frame-by-frame)."""
    results = []
    times = []

    end_frame = len(dataset) if max_frames is None else min(start_frame + max_frames, len(dataset))

    for idx in tqdm(range(start_frame, end_frame), desc="Streaming"):
        data = dataset.get(idx)
        if data is None:
            continue

        img = data['image_og']
        gt_depth = data.get('depth_og')

        # BGR to RGB
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Inference
        start = time.time()
        depth = model.infer_video_depth_one(frame, input_size=input_size, device=device, fp32=fp32)
        elapsed = time.time() - start
        times.append(elapsed)

        # Compute metrics if GT available
        metrics = None
        if gt_depth is not None:
            metrics = compute_depth_metrics(depth, gt_depth)

        results.append({
            'frame_idx': idx,
            'depth': depth,
            'metrics': metrics,
            'time': elapsed,
        })

    return results, times


def evaluate_offline(model, dataset, device, input_size=518, fp32=False,
                     start_frame=0, max_frames=None):
    """Evaluate using offline mode (batch processing)."""
    end_frame = len(dataset) if max_frames is None else min(start_frame + max_frames, len(dataset))

    # Load all frames
    print("Loading frames...")
    frames = []
    gt_depths = []
    frame_indices = []
    for idx in tqdm(range(start_frame, end_frame), desc="Loading"):
        data = dataset.get(idx)
        if data is None:
            continue
        img = data['image_og']
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        gt_depths.append(data.get('depth_og'))
        frame_indices.append(idx)

    frames = np.stack(frames, axis=0)
    print(f"Loaded {len(frames)} frames, shape: {frames.shape}")

    # Batch inference
    print("Running batch inference...")
    start = time.time()
    depths, _ = model.infer_video_depth(frames, target_fps=1, input_size=input_size, device=device, fp32=fp32)
    total_time = time.time() - start
    print(f"Inference time: {total_time:.2f}s ({len(frames)/total_time:.2f} FPS)")

    # Compute metrics
    results = []
    for i, idx in enumerate(frame_indices):
        metrics = None
        if gt_depths[i] is not None:
            metrics = compute_depth_metrics(depths[i], gt_depths[i])

        results.append({
            'frame_idx': idx,
            'depth': depths[i],
            'metrics': metrics,
            'time': total_time / len(depths),
        })

    return results, [total_time / len(depths)] * len(depths)


def save_results(results, output_dir: Path, prefix: str):
    """Save depth results and metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save depths as npz
    depths = np.stack([r['depth'] for r in results], axis=0)
    np.savez_compressed(output_dir / f"{prefix}_depths.npz", depths=depths)
    print(f"Saved depths to {output_dir / f'{prefix}_depths.npz'}")

    # Save metrics as CSV
    import csv
    csv_path = output_dir / f"{prefix}_metrics.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_idx', 'time', 'MAE', 'RMSE', 'AbsRel', 'd105', 'd115', 'd125'])
        for r in results:
            m = r['metrics'] or {}
            writer.writerow([
                r['frame_idx'],
                r['time'],
                m.get('MAE', ''),
                m.get('RMSE', ''),
                m.get('AbsRel', ''),
                m.get('d105', ''),
                m.get('d115', ''),
                m.get('d125', ''),
            ])
    print(f"Saved metrics to {csv_path}")

    # Print summary
    valid_metrics = [r['metrics'] for r in results if r['metrics']]
    if valid_metrics:
        avg_mae = np.mean([m['MAE'] for m in valid_metrics])
        avg_absrel = np.mean([m['AbsRel'] for m in valid_metrics])
        avg_d125 = np.mean([m['d125'] for m in valid_metrics])
        print(f"\nSummary ({prefix}):")
        print(f"  MAE: {avg_mae:.4f}")
        print(f"  AbsRel: {avg_absrel:.4f}")
        print(f"  d<1.25: {avg_d125:.2f}%")


def load_dataset(args):
    """Load dataset based on arguments (same as evaluate.py)."""
    if args.dataset == 'kitti':
        dataset_id = f"kitti_{args.date}_{args.drive}"
        print(f"\n{'='*60}")
        print(f"Loading KITTI {args.date}/{args.drive}")
        print(f"{'='*60}")

        paths = get_dataset_paths('kitti')
        dataset = KITTIEigenSplit(
            rgb_path=paths['rgb_path'],
            depth_path=paths['depth_path'],
            date=args.date,
            drive=args.drive,
        )

    elif args.dataset == 'wheel':
        wheel_name = WHEEL_DATASETS.get(args.wheel_name, args.wheel_name)
        dataset_id = f"wheel_{wheel_name}"

        print(f"\n{'='*60}")
        print(f"Loading Wheel dataset: {wheel_name}")
        print(f"{'='*60}")

        loader = WheelLoader(name=wheel_name)
        dataset = WheelEvalWrapper(loader)

    elif args.dataset == 'tartanair':
        dataset_id = f"tartanair_{args.ta_scene}_{args.ta_level}_P{args.ta_num:03d}"

        print(f"\n{'='*60}")
        print(f"Loading TartanAir: {args.ta_scene}/{args.ta_level}/P{args.ta_num:03d}")
        print(f"{'='*60}")

        paths = get_dataset_paths('tartanair')
        dataset = TartanairLoader(
            dataset_path=paths['dataset_path'],
            scene=args.ta_scene,
            level=args.ta_level,
            num=args.ta_num,
        )

    else:  # ms2
        dataset_id = f"ms2_{args.ms2_timestamp}_{args.ms2_data_type}"

        print(f"\n{'='*60}")
        print(f"Loading MS2: {args.ms2_timestamp} ({args.ms2_data_type})")
        print(f"{'='*60}")

        paths = get_dataset_paths('ms2')
        dataset = MS2Loader(
            dataset_path=paths['dataset_path'],
            timestamp=args.ms2_timestamp,
            data_type=args.ms2_data_type,
        )

    print(f"Loaded {len(dataset)} frames")
    return dataset, dataset_id


def main():
    parser = argparse.ArgumentParser(
        description='Video Depth Anything Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Mode
    parser.add_argument('--offline', action='store_true',
                        help='Use offline (batch) mode instead of streaming')
    parser.add_argument('--compare', action='store_true',
                        help='Run both streaming and offline, compare results')

    # Model options
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'],
                        help='Encoder size')
    parser.add_argument('--metric', action='store_true', default=True,
                        help='Use metric depth model (default)')
    parser.add_argument('--relative', action='store_true',
                        help='Use relative depth model')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')

    # Dataset selection (same as evaluate.py)
    parser.add_argument('--dataset', '-d', type=str, default='kitti',
                        choices=['kitti', 'wheel', 'tartanair', 'ms2'],
                        help='Dataset type')

    # KITTI options
    parser.add_argument('--date', type=str, default='2011_10_03',
                        help='KITTI date')
    parser.add_argument('--drive', type=str, default='0027',
                        help='KITTI drive')

    # Wheel dataset options
    parser.add_argument('--wheel-name', '-w', type=str, default=None,
                        choices=['indoor', 'outdoor', 'forest',
                                 '25_10_20_14_30', '25_10_20_14_50', '25_11_04_16_00'],
                        help='Wheel dataset name')

    # TartanAir dataset options
    parser.add_argument('--ta-scene', type=str, default=None,
                        help='TartanAir scene name')
    parser.add_argument('--ta-level', type=str, default='Easy',
                        choices=['Easy', 'Hard'])
    parser.add_argument('--ta-num', type=int, default=0)

    # MS2 dataset options
    parser.add_argument('--ms2-timestamp', type=str, default=None)
    parser.add_argument('--ms2-data-type', type=str, default='thr',
                        choices=['thr', 'rgb'])

    # Evaluation options
    parser.add_argument('--start-frame', type=int, default=0,
                        help='Frame index to start evaluation from')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum frames to process')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Custom output directory')

    args = parser.parse_args()

    # Validate arguments
    if args.dataset == 'wheel' and args.wheel_name is None:
        parser.error("--wheel-name (-w) is required for wheel dataset")
    if args.dataset == 'tartanair' and args.ta_scene is None:
        parser.error("--ta-scene is required for tartanair dataset")
    if args.dataset == 'ms2' and args.ms2_timestamp is None:
        parser.error("--ms2-timestamp is required for ms2 dataset")

    if args.relative:
        args.metric = False

    # Load dataset
    dataset, dataset_id = load_dataset(args)

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / dataset_id / "vda_eval"
    depth_type = "metric" if args.metric else "relative"

    if args.compare:
        # Run both modes and compare
        print("\n" + "="*60)
        print("STREAMING MODE")
        print("="*60)
        model_stream = load_model(args.encoder, args.metric, streaming=True, device=args.device)
        results_stream, times_stream = evaluate_streaming(
            model_stream, dataset, args.device, args.input_size, args.fp32,
            args.start_frame, args.max_frames
        )
        save_results(results_stream, output_dir, f"{args.encoder}_{depth_type}_streaming")
        del model_stream
        torch.cuda.empty_cache()

        print("\n" + "="*60)
        print("OFFLINE MODE")
        print("="*60)
        model_offline = load_model(args.encoder, args.metric, streaming=False, device=args.device)
        results_offline, times_offline = evaluate_offline(
            model_offline, dataset, args.device, args.input_size, args.fp32,
            args.start_frame, args.max_frames
        )
        save_results(results_offline, output_dir, f"{args.encoder}_{depth_type}_offline")

        # Compare
        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        print(f"Streaming avg time: {np.mean(times_stream)*1000:.1f}ms")
        print(f"Offline avg time: {np.mean(times_offline)*1000:.1f}ms")

        # Depth difference
        for i in range(min(5, len(results_stream))):
            diff = np.abs(results_stream[i]['depth'] - results_offline[i]['depth'])
            print(f"Frame {i}: MAE diff = {diff.mean():.4f}")

    else:
        # Single mode
        streaming = not args.offline
        mode_name = "streaming" if streaming else "offline"
        print(f"\nMode: {mode_name}, Encoder: {args.encoder}, Depth: {depth_type}")

        model = load_model(args.encoder, args.metric, streaming=streaming, device=args.device)

        if streaming:
            results, times = evaluate_streaming(
                model, dataset, args.device, args.input_size, args.fp32,
                args.start_frame, args.max_frames
            )
        else:
            results, times = evaluate_offline(
                model, dataset, args.device, args.input_size, args.fp32,
                args.start_frame, args.max_frames
            )

        save_results(results, output_dir, f"{args.encoder}_{depth_type}_{mode_name}")
        print(f"\nAvg inference time: {np.mean(times)*1000:.1f}ms ({1/np.mean(times):.1f} FPS)")
        print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
