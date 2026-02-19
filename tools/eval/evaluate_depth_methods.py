#!/usr/bin/env python3
"""Unified depth estimation evaluation script.

Runs depth estimation and evaluates multiple methods (metric depth only):
- PR-Depth (with pose estimation, uses estimated poses for TAE)
- PR-Depth w/o fusion (triangulation + solve_metric only, no temporal fusion)
- UniDepth (metric depth, uses GT poses for TAE)
- Video Depth Anything (streaming, metric depth, uses GT poses for TAE)

Usage:
    # Compare all methods on KITTI (including ablation)
    python tools/eval/evaluate_depth_methods.py --methods pr_depth pr_depth_wo_fusion unidepth vda

    # Specific methods only
    python tools/eval/evaluate_depth_methods.py --methods pr_depth vda

    # Different KITTI sequence
    python tools/eval/evaluate_depth_methods.py --methods pr_depth vda --date 2011_09_26 --drive 0001

    # With max frames limit
    python tools/eval/evaluate_depth_methods.py --methods pr_depth vda --max-frames 100

    # Wheel dataset - Indoor/Outdoor/Forest
    python tools/eval/evaluate_depth_methods.py -d wheel -w indoor --methods pr_depth unidepth
    python tools/eval/evaluate_depth_methods.py -d wheel -w outdoor --methods pr_depth vda
    python tools/eval/evaluate_depth_methods.py -d wheel -w forest --methods pr_depth

    # TartanAir
    python tools/eval/evaluate_depth_methods.py -d tartanair --ta-scene abandonedfactory --ta-level Easy --ta-num 0 --methods pr_depth vda
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import cv2

from configs import get_dataset_paths
from dataloader import KITTIEigenSplit, WheelLoader, WheelEvalWrapper, MS2Loader, TartanairLoader
from src.evaluation.metrics import compute_depth_metrics, compute_tae


# Wheel dataset name mapping
WHEEL_DATASETS = {
    'indoor': '25_10_20_14_50',
    'outdoor': '25_10_20_14_30',
    'forest': '25_11_04_16_00',
}


class DepthEstimator:
    """Base class for depth estimators."""
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.name = "base"

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth from RGB image. Returns (H, W) depth map."""
        raise NotImplementedError


class UniDepthEstimator(DepthEstimator):
    """UniDepth V1 - metric depth with camera intrinsics."""
    def __init__(self, K: np.ndarray, device: str = 'cuda'):
        super().__init__(device)
        self.name = "UniDepth"
        self.K = K

        # Import and load model (V1 from local weights)
        sys.path.insert(0, str(PROJECT_ROOT / "external" / "UniDepth"))
        from unidepth.models import UniDepthV1

        model_path = PROJECT_ROOT / "weights" / "uni_depth"
        self.model = UniDepthV1.from_pretrained(str(model_path), local_files_only=True)
        self.model = self.model.to(device).eval()

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Returns metric depth."""
        with torch.no_grad():
            # UniDepth expects RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)
            image_tensor = image_tensor.to(self.device)

            K_tensor = torch.from_numpy(self.K).float().to(self.device)

            predictions = self.model.infer(image_tensor, K_tensor)
            depth = predictions["depth"].squeeze().cpu().numpy()

        return depth


class VDAEstimator(DepthEstimator):
    """Video Depth Anything - streaming mode."""
    def __init__(self, encoder: str = 'vitl', metric: bool = True, device: str = 'cuda'):
        super().__init__(device)
        self.name = f"VDA_{'metric' if metric else 'rel'}_{encoder}"
        self.metric = metric

        # Import and load model
        VDA_PATH = PROJECT_ROOT / "external" / "video_depth_anything"
        sys.path.insert(0, str(VDA_PATH))
        from video_depth_anything.video_depth_stream import VideoDepthAnything

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        self.model = VideoDepthAnything(**model_configs[encoder])
        checkpoint_name = 'metric_video_depth_anything' if metric else 'video_depth_anything'
        checkpoint_path = VDA_PATH / "checkpoints" / f"{checkpoint_name}_{encoder}.pth"
        self.model.load_state_dict(torch.load(str(checkpoint_path), map_location='cpu'))
        self.model = self.model.to(device).eval()

        self.input_size = 518

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Returns depth (metric or relative depending on config)."""
        with torch.no_grad():
            # Use streaming API - expects RGB numpy array
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            depth = self.model.infer_video_depth_one(
                image_rgb,
                input_size=self.input_size,
                device=self.device
            )
            # depth is already numpy array at original resolution
        return depth

    def reset(self):
        """Reset temporal state."""
        # Reset internal state of VideoDepthAnything streaming model
        self.model.id = -1
        self.model.transform = None
        self.model.frame_id_list = []
        self.model.frame_cache_list = []


class PRDepthEstimator(DepthEstimator):
    """PR-Depth - pose estimation + depth refinement."""
    def __init__(self, encoder: str = 'vitl', device: str = 'cuda',
                 fx: float = None, fy: float = None, cx: float = None, cy: float = None,
                 use_iterative: bool = True, iterative_iters: int = 1,
                 use_segmentation: bool = True,
                 use_rgb_guide: bool = True, metric_scale_mode: int = 2,
                 use_magsac_scoring: bool = True,
                 skip_temporal_fusion: bool = False,
                 use_gt_R: bool = False,
                 gt_pose_mode: str = 'none',  # 'none', 'gt_R', 'gt_pose'
                 # Pixel-count thresholds
                min_scale_overlap: int = 100,
                seg_min_size: int = 2000,
                max_points: int = 500):
        super().__init__(device)
        self.name = "PR-Depth"
        self.gt_pose_mode = gt_pose_mode  # 'none', 'gt_R', 'gt_pose'

        # gt_R mode: force use_gt_R in runner
        if gt_pose_mode == 'gt_R':
            use_gt_R = True

        from src.evaluation.runners import PRDepthRunner
        self.runner = PRDepthRunner(
            device=device,
            iterative=use_iterative,
            iterative_iters=iterative_iters,
            use_segmentation=use_segmentation,
            use_rgb_guide=use_rgb_guide,
            metric_scale_mode=metric_scale_mode,
            use_magsac_scoring=use_magsac_scoring,
            skip_temporal_fusion=skip_temporal_fusion,
            use_gt_R=use_gt_R,
            # gt_pose mode: use GT pose fallback with 0 threshold (always use GT)
            use_gt_pose_fallback=(gt_pose_mode == 'gt_pose'),
            gt_pose_rotation_threshold_deg=0.0 if gt_pose_mode == 'gt_pose' else 3.0,
            # Pixel-count thresholds
            min_scale_overlap=min_scale_overlap,
            seg_min_size=seg_min_size,
            max_points=max_points,
            min_baseline=0.1,
        )

        from src.estimators.depth import DepthAnythingEstimator
        self.depth_estimator = DepthAnythingEstimator(encoder=encoder)

        self.prev_image = None
        self.prev_depth = None
        self.poses = []
        self.baselines = []
        self._initialized = False
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def process_frame(self, image: np.ndarray, baseline: float,
                      gt_R: Optional[np.ndarray] = None,
                      gt_t: Optional[np.ndarray] = None,
                      ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """Process frame and return (depth, pose).

        Returns:
            depth: Refined depth map (H, W) - metric depth
            pose: (R, t) tuple or None for first frame
        """
        # Get monocular depth (infer returns normalized inverse depth [0,1])
        # PR-Depth runner expects raw inverse depth, not converted
        curr_depth = self.depth_estimator.infer(image)

        # Initialize runner on first frame
        if not self._initialized:
            H, W = image.shape[:2]
            self.runner.initialize(H=H, W=W, fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy)
            self._initialized = True

        if self.prev_image is None:
            self.prev_image = image
            self.prev_depth = curr_depth
            # Return metric depth for evaluation (convert from inverse depth)
            return 1.0 / (curr_depth + 1e-6), None

        # Run PR-Depth
        result = self.runner.process_frame(
            img_curr=image,
            img_prev=self.prev_image,
            depth_curr=curr_depth,
            depth_prev=self.prev_depth,
            baseline=baseline,
            gt_R=gt_R,
            gt_t=gt_t,
        )

        # Get refined depth (metric depth from triangulation)
        z_refined = result.extra.get('z_refined')
        if z_refined is None or z_refined.size == 0:
            # Fallback to converted inverse depth if z_refined not available
            z_refined = 1.0 / (curr_depth + 1e-6)

        # Store pose
        pose = (result.R, result.t)
        self.poses.append(pose)
        self.baselines.append(baseline)

        # Update prev - IMPORTANT: prev_depth must be raw inverse depth for runner
        self.prev_image = image
        self.prev_depth = curr_depth  # Raw inverse depth, NOT z_refined

        return z_refined, pose

    def reset(self):
        self.runner.reset()
        self.prev_image = None
        self.prev_depth = None
        self.poses = []
        self.baselines = []
        self._initialized = False


def align_scale(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Align prediction scale to GT using median scaling."""
    mask = (gt > 0) & (pred > 0) & np.isfinite(gt) & np.isfinite(pred)
    if mask.sum() < 100:
        return pred
    scale = np.median(gt[mask]) / np.median(pred[mask])
    return pred * scale


def evaluate_method(
    method_name: str,
    estimator,
    dataset,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
    align_scale_flag: bool = False,
) -> Dict:
    """Evaluate a depth estimation method.

    Returns:
        Dictionary with metrics and depth maps
    """
    end_frame = len(dataset) if max_frames is None else min(start_frame + max_frames, len(dataset))

    depths = []
    frame_indices = []
    all_metrics = []
    poses = []  # For PR-Depth
    baselines = []

    fx, fy, cx, cy = dataset.get_intrinsics()

    # Reset if needed
    if hasattr(estimator, 'reset'):
        estimator.reset()

    for idx in tqdm(range(start_frame, end_frame), desc=f"Evaluating {method_name}"):
        data = dataset.get(idx)
        if data is None:
            continue

        image = data['image_og']
        gt_depth = data.get('depth_og')
        baseline = dataset.get_baseline(idx)

        # Estimate depth
        if isinstance(estimator, PRDepthEstimator):
            # Get GT pose if needed for ablation modes
            gt_R_frame, gt_t_frame = None, None
            if estimator.gt_pose_mode in ('gt_R', 'gt_pose') and idx >= 1:
                try:
                    gt_R_frame, gt_t_frame = dataset.get_relative_pose(idx)
                except Exception:
                    pass
            pred_depth, pose = estimator.process_frame(
                image, baseline, gt_R=gt_R_frame, gt_t=gt_t_frame)
            if pose is not None:
                poses.append(pose)
                baselines.append(baseline)
        else:
            pred_depth = estimator.estimate(image)

        # Align scale if needed (for relative depth methods)
        if align_scale_flag and gt_depth is not None:
            pred_depth = align_scale(pred_depth, gt_depth)

        # Resize if needed
        if gt_depth is not None and pred_depth.shape != gt_depth.shape:
            pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)

        depths.append(pred_depth)
        frame_indices.append(idx)

        # Compute metrics
        if gt_depth is not None:
            metrics = compute_depth_metrics(pred_depth, gt_depth, max_depth=80.0)
            if metrics:
                metrics['frame_idx'] = idx
                all_metrics.append(metrics)

    # Aggregate metrics
    result = {'method': method_name, 'num_frames': len(depths)}

    if all_metrics:
        df = pd.DataFrame(all_metrics)
        result['AbsRel'] = float(df['AbsRel'].mean())
        result['RMSE'] = float(df['RMSE'].mean())
        result['MAE'] = float(df['MAE'].mean())
        result['d105'] = float(df['d105'].mean())
        result['d115'] = float(df['d115'].mean())
        result['d125'] = float(df['d125'].mean())

    # Compute TAE
    if len(depths) > 1:
        # Get GT poses and baselines for TAE
        gt_poses = []
        gt_baselines = []
        for idx in frame_indices[1:]:  # Skip first frame
            R_gt, t_gt = dataset.get_relative_pose(idx)
            baseline_gt = dataset.get_baseline(idx)
            gt_poses.append((R_gt, t_gt))
            gt_baselines.append(baseline_gt)

        # Choose poses for TAE
        if isinstance(estimator, PRDepthEstimator) and len(poses) > 0:
            # PR-Depth uses estimated poses
            tae_poses = poses
            tae_baselines = baselines
        else:
            # Other methods use GT poses
            tae_poses = gt_poses
            tae_baselines = gt_baselines

        if tae_poses and tae_baselines:
            tae_result = compute_tae(
                depths=depths,
                poses=tae_poses,
                baselines=tae_baselines,
                fx=fx, fy=fy, cx=cx, cy=cy
            )
            if tae_result:
                result['TAE'] = tae_result['TAE']

    # Store depths for optional saving
    result['depths'] = np.stack(depths, axis=0)
    result['frame_indices'] = np.array(frame_indices)

    return result


def main():
    parser = argparse.ArgumentParser(description='Unified depth estimation evaluation')

    parser.add_argument('--methods', type=str, nargs='+', required=True,
                        choices=['pr_depth', 'pr_depth_wo_fusion', 'pr_depth_global_scale',
                                 'pr_depth_gt_R', 'pr_depth_gt_pose',
                                 'unidepth', 'vda'],
                        help='Methods to evaluate (metric depth only)')

    # Dataset options
    parser.add_argument('--dataset', '-d', type=str, default='kitti',
                        choices=['kitti', 'wheel', 'ms2', 'tartanair'],
                        help='Dataset type')
    parser.add_argument('--date', type=str, default='2011_10_03',
                        help='KITTI date')
    parser.add_argument('--drive', type=str, default='0027',
                        help='KITTI drive')

    # Wheel dataset options
    parser.add_argument('--wheel-name', '-w', type=str, default=None,
                        choices=['indoor', 'outdoor', 'forest',
                                 '25_10_20_14_30', '25_10_20_14_50', '25_11_04_16_00'],
                        help='Wheel dataset name (indoor/outdoor/forest or full name)')

    # MS2 dataset options
    parser.add_argument('--ms2-timestamp', type=str, default=None,
                        help='MS2 timestamp/sequence name (e.g., _2021-08-06-10-59-33)')
    parser.add_argument('--ms2-data-type', type=str, default='thr',
                        choices=['thr', 'rgb'],
                        help='MS2 data type (thr=thermal, rgb=visible)')

    # TartanAir dataset options
    parser.add_argument('--ta-scene', type=str, default=None,
                        help='TartanAir scene name (e.g., abandonedfactory, amusement)')
    parser.add_argument('--ta-level', type=str, default='Easy',
                        choices=['Easy', 'Hard'],
                        help='TartanAir difficulty level')
    parser.add_argument('--ta-num', type=int, default=0,
                        help='TartanAir trajectory number (0=P000, 1=P001, ...)')

    # Evaluation options
    parser.add_argument('--start-frame', type=int, default=0,
                        help='Start frame index')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum frames to process')
    parser.add_argument('--encoder', type=str, default='vitl',
                        choices=['vits', 'vitb', 'vitl'],
                        help='Encoder size for depth models')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    # Output options
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--save-depths', action='store_true',
                        help='Save depth maps to npz')

    # PR-Depth ablation options
    parser.add_argument('--no-iterative', action='store_true',
                        help='Disable iterative refinement')
    parser.add_argument('--iter2', action='store_true',
                        help='Use 2 iterative refinement iterations instead of 1')
    parser.add_argument('--no-segmentation', action='store_true',
                        help='Disable edge-aware segmentation')
    parser.add_argument('--no-rgb-guide', action='store_true',
                        help='Disable RGB guiding')
    parser.add_argument('--no-metric-scale', action='store_true',
                        help='Disable per-segment scale estimation')
    parser.add_argument('--no-magsac', action='store_true',
                        help='Use paper MAD-based scoring instead of MAGSAC++')
    parser.add_argument('--skip-temporal-fusion', action='store_true',
                        help='Skip Bayesian update, only triangulation + solve_metric (ablation)')
    parser.add_argument('--use-gt-R', action='store_true',
                        help='Always use GT rotation when provided (ablation)')

    # Pixel-count thresholds (for different resolutions)
    parser.add_argument('--min-scale-overlap', type=int, default=100,
                        help='Minimum overlapping pixels for scale matching')
    parser.add_argument('--seg-min-size', type=int, default=2000,
                        help='Minimum segment size for segmentation')
    parser.add_argument('--max-points', type=int, default=2000,
                        help='Maximum points for motion estimation')

    args = parser.parse_args()

    # Validate wheel dataset argument
    if args.dataset == 'wheel' and args.wheel_name is None:
        parser.error("--wheel-name (-w) is required for wheel dataset")

    if args.dataset == 'ms2' and args.ms2_timestamp is None:
        parser.error("--ms2-timestamp is required for ms2 dataset")

    if args.dataset == 'tartanair' and args.ta_scene is None:
        parser.error("--ta-scene is required for tartanair dataset")

    # Load dataset
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
        print(f"Loaded {len(dataset)} frames")

    elif args.dataset == 'wheel':
        # Map friendly names to actual names
        wheel_name = WHEEL_DATASETS.get(args.wheel_name, args.wheel_name)
        dataset_id = f"wheel_{wheel_name}_thr"

        print(f"\n{'='*60}")
        print(f"Loading Wheel dataset: {wheel_name}")
        print(f"{'='*60}")

        loader = WheelLoader(name=wheel_name, data_type='thr')
        dataset = WheelEvalWrapper(loader)
        print(f"Loaded {len(dataset)} frames")

    elif args.dataset == 'ms2':
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

    elif args.dataset == 'tartanair':
        p_str = f"P{args.ta_num:03d}"
        dataset_id = f"tartanair_{args.ta_scene}_{args.ta_level}_{p_str}"
        print(f"\n{'='*60}")
        print(f"Loading TartanAir: {args.ta_scene}/{args.ta_level}/{p_str}")
        print(f"{'='*60}")
        paths = get_dataset_paths('tartanair')
        dataset = TartanairLoader(
            dataset_path=paths['dataset_path'],
            scene=args.ta_scene,
            level=args.ta_level,
            num=args.ta_num,
        )
        print(f"Loaded {len(dataset)} frames")

    # Get camera intrinsics for UniDepth
    fx, fy, cx, cy = dataset.get_intrinsics()
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "results" / dataset_id / "depth_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each method
    results = {}

    for method in args.methods:
        print(f"\n{'='*60}")
        print(f"Evaluating: {method.upper()}")
        print(f"{'='*60}")

        try:
            # Create estimator
            if method == 'pr_depth':
                estimator = PRDepthEstimator(
                    encoder=args.encoder, device=args.device,
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    use_iterative=not args.no_iterative,
                    iterative_iters=2 if args.iter2 else 1,
                    use_segmentation=not args.no_segmentation,
                    use_rgb_guide=not args.no_rgb_guide,
                    metric_scale_mode=0 if args.no_metric_scale else 2,  # per-segment (default)
                    use_magsac_scoring=not args.no_magsac,
                    skip_temporal_fusion=args.skip_temporal_fusion,
                    use_gt_R=args.use_gt_R,
                )
                align_scale = False
            elif method == 'pr_depth_wo_fusion':
                # PR-Depth without temporal fusion (triangulation + solve_metric only)
                estimator = PRDepthEstimator(
                    encoder=args.encoder, device=args.device,
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    use_iterative=not args.no_iterative,
                    iterative_iters=2 if args.iter2 else 1,
                    use_segmentation=not args.no_segmentation,
                    use_rgb_guide=not args.no_rgb_guide,
                    metric_scale_mode=0 if args.no_metric_scale else 2,  # per-segment (default)
                    use_magsac_scoring=not args.no_magsac,
                    skip_temporal_fusion=True,  # Key difference: no temporal fusion
                    use_gt_R=args.use_gt_R,
                )
                align_scale = False

            elif method == 'pr_depth_global_scale':
                # PR-Depth with global scale estimation instead of per-segment
                estimator = PRDepthEstimator(
                    encoder=args.encoder, device=args.device,
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    use_iterative=not args.no_iterative,
                    iterative_iters=2 if args.iter2 else 1,
                    use_segmentation=False,
                    use_rgb_guide=not args.no_rgb_guide,
                    metric_scale_mode=1,  # global scale (mode=1)
                    use_magsac_scoring=not args.no_magsac,
                    skip_temporal_fusion=args.skip_temporal_fusion,
                    use_gt_R=args.use_gt_R,
                )
                align_scale = False

            elif method == 'pr_depth_gt_R':
                # PR-Depth with GT rotation (pose ablation: R from GT, t estimated)
                estimator = PRDepthEstimator(
                    encoder=args.encoder, device=args.device,
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    use_iterative=not args.no_iterative,
                    iterative_iters=2 if args.iter2 else 1,
                    use_segmentation=not args.no_segmentation,
                    use_rgb_guide=not args.no_rgb_guide,
                    metric_scale_mode=0 if args.no_metric_scale else 2,
                    use_magsac_scoring=not args.no_magsac,
                    skip_temporal_fusion=args.skip_temporal_fusion,
                    gt_pose_mode='gt_R',
                )
                align_scale = False

            elif method == 'pr_depth_gt_pose':
                # PR-Depth with full GT pose (pose ablation: both R and t from GT)
                estimator = PRDepthEstimator(
                    encoder=args.encoder, device=args.device,
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    use_iterative=not args.no_iterative,
                    iterative_iters=2 if args.iter2 else 1,
                    use_segmentation=not args.no_segmentation,
                    use_rgb_guide=not args.no_rgb_guide,
                    metric_scale_mode=0 if args.no_metric_scale else 2,
                    use_magsac_scoring=not args.no_magsac,
                    skip_temporal_fusion=args.skip_temporal_fusion,
                    gt_pose_mode='gt_pose',
                )
                align_scale = False
            elif method == 'unidepth':
                estimator = UniDepthEstimator(K=K, device=args.device)
                align_scale = False  # UniDepth outputs metric depth
            elif method == 'vda':
                estimator = VDAEstimator(encoder=args.encoder, metric=True, device=args.device)
                align_scale = False  # Using metric VDA

            # Evaluate
            result = evaluate_method(
                method_name=method,
                estimator=estimator,
                dataset=dataset,
                start_frame=args.start_frame,
                max_frames=args.max_frames,
                align_scale_flag=align_scale,
            )
            results[method] = result

            # Save depths if requested
            if args.save_depths:
                depths_path = output_dir / f"{method}_depths.npz"
                np.savez_compressed(
                    depths_path,
                    depths=result['depths'],
                    frame_indices=result['frame_indices'],
                )
                print(f"Saved depths to: {depths_path}")

        except Exception as e:
            print(f"Error evaluating {method}: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison table
    print(f"\n{'='*60}")
    print("Results Comparison")
    print(f"{'='*60}")

    metrics_order = ['AbsRel', 'd125', 'RMSE', 'TAE']
    header = f"{'Method':<25}"
    for m in metrics_order:
        header += f"{m:>10}"
    print(header)
    print("-" * len(header))

    for method, result in results.items():
        row = f"{method:<25}"
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

    # Save summary CSV
    summary_data = []
    for method, result in results.items():
        row = {'method': method}
        for m in metrics_order + ['MAE', 'd105', 'd115', 'num_frames']:
            if m in result:
                row[m] = result[m]
        summary_data.append(row)

    df = pd.DataFrame(summary_data)
    csv_path = output_dir / "comparison_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary to: {csv_path}")


if __name__ == '__main__':
    main()
