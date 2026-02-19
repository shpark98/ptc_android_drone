#!/usr/bin/env python3
"""Unified evaluation script for PR-Depth and baseline methods.

Features:
- Real-time logging: CSV, NPZ, trajectory image updated every frame
- Clean folder structure separating PR-Depth experiments from baselines
- 'latest' symlink for quick access to most recent PR-Depth run
- Support for KITTI, Wheel, TartanAir, and MS2 datasets

Folder structure:
    results/
    ├── kitti_2011_10_03_0027/
    │   ├── baselines/                    # Other methods
    │   │   └── madpose/
    │   └── pr_depth/                     # Our method
    │       ├── v1_initial/
    │       └── latest -> v1_initial/
    ├── wheel_25_10_20_14_50/             # Wheel dataset (outdoor)
    │   └── pr_depth/
    │       └── v1_outdoor/
    ├── tartanair_abandonedfactory_Easy_P000/  # TartanAir dataset
    │   └── pr_depth/
    │       └── v1_tartanair/
    └── ms2_20231015_thr/                 # MS2 thermal dataset
        └── pr_depth/
            └── v1_thermal/

Usage:
    # KITTI - PR-Depth experiment
    python tools/eval/evaluate.py -e v1_initial
    python tools/eval/evaluate.py -e v2_dc_fix --max-frames 100

    # KITTI - Run baseline
    python tools/eval/evaluate.py --baseline madpose

    # KITTI - Different sequence
    python tools/eval/evaluate.py -e v1 --date 2011_09_26 --drive 0001

    # Wheel dataset - Indoor/Outdoor/Forest
    python tools/eval/evaluate.py -d wheel -w indoor -e v1_indoor
    python tools/eval/evaluate.py -d wheel -w outdoor -e v1_outdoor
    python tools/eval/evaluate.py -d wheel -w forest -e v1_forest

    # TartanAir dataset
    python tools/eval/evaluate.py -d tartanair --ta-scene abandonedfactory --ta-level Easy --ta-num 0 -e v1_tartanair
    python tools/eval/evaluate.py -d tartanair --ta-scene amusement --ta-level Hard --ta-num 1 -e v1_hard

    # MS2 dataset (thermal or RGB)
    python tools/eval/evaluate.py -d ms2 --ms2-timestamp 20231015 --ms2-data-type thr -e v1_thermal
    python tools/eval/evaluate.py -d ms2 --ms2-timestamp 20231015 --ms2-data-type rgb -e v1_rgb

Note: For Video Depth Anything evaluation, use tools/eval/evaluate_vda.py instead.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'cpp' / 'build'))

from configs import get_dataset_paths
from dataloader import KITTIEigenSplit, WheelLoader, WheelEvalWrapper, TartanairLoader, MS2Loader
from src.evaluation import (
    Evaluator,
    ResultsManager,
    Visualizer,
)
from src.evaluation.runners import PRDepthRunner, MADPoseRunner


# Wheel dataset name mapping
WHEEL_DATASETS = {
    'indoor': '25_10_20_14_50',
    'outdoor': '25_10_20_14_30',
    'forest': '25_11_04_16_00',
}


def get_runner(method: str, device: str,
               use_baseline_guard: bool = True,
               min_baseline: float = 0.05, baseline_ema_beta: float = 0.9,
               use_iterative: bool = True,
               iterative_iters: int = 1,
               use_segmentation: bool = True,
               use_rgb_guide: bool = True,
               metric_scale_mode: int = 2,
               use_magsac_scoring: bool = True,
               use_gt_pose_fallback: bool = False,
               gt_pose_rotation_threshold_deg: float = 3.0,
               skip_temporal_fusion: bool = False,
               use_gt_R: bool = False,
               # Pixel-count thresholds
               min_scale_overlap: int = 2000,
               seg_min_size: int = 200,
               max_points: int = 2000):
    """Get runner instance for method."""
    if method == 'pr_depth':
        return PRDepthRunner(
            device=device,
            iterative=use_iterative,
            iterative_iters=iterative_iters,
            use_baseline_guard=use_baseline_guard,
            min_baseline=min_baseline,
            baseline_ema_beta=baseline_ema_beta,
            use_segmentation=use_segmentation,
            use_rgb_guide=use_rgb_guide,
            metric_scale_mode=metric_scale_mode,
            use_magsac_scoring=use_magsac_scoring,
            use_gt_pose_fallback=use_gt_pose_fallback,
            gt_pose_rotation_threshold_deg=gt_pose_rotation_threshold_deg,
            skip_temporal_fusion=skip_temporal_fusion,
            use_gt_R=use_gt_R,
            # Pixel-count thresholds
            min_scale_overlap=min_scale_overlap,
            seg_min_size=seg_min_size,
            max_points=max_points,
        )
    elif method == 'madpose':
        return MADPoseRunner(device=device)
    elif method == 'batrack':
        from src.evaluation.runners import BaTrackRunner
        return BaTrackRunner(device=device)
    elif method == 'monst3r':
        from src.evaluation.runners import MonST3RRunner
        return MonST3RRunner(device=device)
    else:
        raise ValueError(f"Unknown method: {method}")


def get_depth_estimator(method: str, encoder: str = 'vitl', K: 'np.ndarray' = None,
                        device: str = 'cuda'):
    """Get appropriate depth estimator for method.

    Args:
        method: Method name ('pr_depth', 'madpose', etc.)
        encoder: DepthAnything encoder size
        K: Camera intrinsic matrix (required for UniDepth/MADPose)
        device: Device for inference
    """
    if method == 'pr_depth':
        from src.estimators.depth import DepthAnythingEstimator
        return DepthAnythingEstimator(encoder=encoder)
    elif method == 'madpose':
        try:
            from src.estimators.depth import UniDepthEstimator
            if K is None:
                raise ValueError("MADPose requires camera intrinsic matrix K for UniDepth")
            return UniDepthEstimator(K=K)
        except (ImportError, Exception) as e:
            print(f"UniDepth not available ({e}), using DepthAnything")
            from src.estimators.depth import DepthAnythingEstimator
            return DepthAnythingEstimator(encoder=encoder)
    else:
        return None


def run_evaluation(
    method: str,
    exp_dir: Path,
    dataset,
    device: str,
    encoder: str,
    start_frame: int,
    max_frames: int,
    traj_update_interval: int,
    quiet: bool,
    results_base_dir: str = None,
    dataset_id: str = None,
    use_baseline_guard: bool = True,
    min_baseline: float = 0.05,
    baseline_ema_beta: float = 0.9,
    use_iterative: bool = True,
    iterative_iters: int = 1,
    use_segmentation: bool = True,
    use_rgb_guide: bool = True,
    metric_scale_mode: int = 2,
    use_gt_flow: bool = False,
    flow_method: str = None,
    save_depths: bool = False,
    odom_noise: float = 0.0,
    use_magsac_scoring: bool = True,
    use_gps_baseline: bool = False,
    use_gt_pose_fallback: bool = False,
    gt_pose_rotation_threshold_deg: float = 3.0,
    skip_temporal_fusion: bool = False,
    use_gt_R: bool = False,
    # Pixel-count thresholds
    min_scale_overlap: int = 2000,
    seg_min_size: int = 200,
    max_points: int = 2000,
):
    """Run evaluation for a single method."""
    import numpy as np
    print(f"Output directory: {exp_dir}")

    runner = get_runner(method, device,
                        use_baseline_guard=use_baseline_guard,
                        min_baseline=min_baseline, baseline_ema_beta=baseline_ema_beta,
                        use_iterative=use_iterative,
                        iterative_iters=iterative_iters,
                        use_segmentation=use_segmentation,
                        use_rgb_guide=use_rgb_guide,
                        metric_scale_mode=metric_scale_mode,
                        use_magsac_scoring=use_magsac_scoring,
                        use_gt_pose_fallback=use_gt_pose_fallback,
                        gt_pose_rotation_threshold_deg=gt_pose_rotation_threshold_deg,
                        skip_temporal_fusion=skip_temporal_fusion,
                        use_gt_R=use_gt_R,
                        # Pixel-count thresholds
                        min_scale_overlap=min_scale_overlap,
                        seg_min_size=seg_min_size,
                        max_points=max_points)

    # Build camera intrinsic matrix K for methods that need it
    fx, fy, cx, cy = dataset.get_intrinsics()
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    depth_estimator = get_depth_estimator(
        method, encoder, K=K,
        device=device,
    )

    # Get external flow estimator if specified
    flow_estimator = None
    if flow_method:
        from src.evaluation.flow_estimators import get_flow_estimator
        flow_estimator = get_flow_estimator(flow_method, device=device)
        if flow_estimator:
            print(f"Using external flow estimator: {flow_estimator.name}")
        else:
            print(f"Warning: Unknown flow method '{flow_method}', using DIS flow")

    evaluator = Evaluator(
        runner, dataset, depth_estimator,
        results_base_dir=results_base_dir,
        use_gt_flow=use_gt_flow,
        flow_estimator=flow_estimator,
        odom_noise=odom_noise,
        use_gps_baseline=use_gps_baseline,
        use_gt_pose_fallback=use_gt_pose_fallback,
    )

    if odom_noise > 0:
        print(f"Adding ±{odom_noise*100:.0f}% random noise to odometry")

    if use_gps_baseline:
        print("Using GPS baseline instead of wheel odometry")

    if use_gt_pose_fallback:
        print(f"GT pose fallback enabled (rotation threshold: {gt_pose_rotation_threshold_deg} deg)")

    summary = evaluator.run(
        start_frame=start_frame,
        max_frames=max_frames,
        verbose=not quiet,
        output_dir=str(exp_dir),
        save_trajectory_img=True,
        trajectory_update_interval=traj_update_interval,
        dataset_id=dataset_id,
        save_depths=save_depths,
    )

    return summary, evaluator.get_trajectories()


def print_summary(method: str, summary):
    """Print evaluation summary."""
    print(f"\n{method.upper()} Results:")
    print(f"  Frames: {summary.total_frames} ({summary.success_frames} successful)")
    print(f"  Time: {summary.elapsed_time:.1f}s ({summary.fps:.2f} FPS)")
    if summary.rot_error_mean is not None:
        print(f"  Rotation Error: {summary.rot_error_mean:.3f}° (median: {summary.rot_error_median:.3f}°)")
    if summary.trans_error_mean is not None:
        print(f"  Translation Error: {summary.trans_error_mean:.3f}° (median: {summary.trans_error_median:.3f}°)")
    if summary.ATE_RMSE is not None:
        print(f"  ATE RMSE: {summary.ATE_RMSE:.3f} m")
    if summary.tri_d125_mean is not None:
        print(f"  Triangulation δ<1.25: {summary.tri_d125_mean:.1f}%")
    if summary.ref_d125_mean is not None:
        print(f"  Refined δ<1.25: {summary.ref_d125_mean:.1f}%")
    if summary.TAE is not None:
        print(f"  TAE: {summary.TAE:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='PR-Depth evaluation with real-time logging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Experiment mode (PR-Depth)
    parser.add_argument('--exp-name', '-e', type=str, default=None,
                        help='PR-Depth experiment name (e.g., v1_initial, v2_dc_fix)')

    # Baseline mode
    parser.add_argument('--baseline', '-b', type=str, nargs='+', default=None,
                        choices=['madpose', 'batrack', 'monst3r'],
                        help='Run baseline method(s)')

    # Dataset selection
    parser.add_argument('--dataset', '-d', type=str, default='kitti',
                        choices=['kitti', 'wheel', 'tartanair', 'ms2'],
                        help='Dataset type (kitti, wheel, tartanair, or ms2)')

    # KITTI options
    parser.add_argument('--date', type=str, default='2011_10_03',
                        help='KITTI date')
    parser.add_argument('--drive', type=str, default='0027',
                        help='KITTI drive')

    # Wheel dataset options
    parser.add_argument('--wheel-name', '-w', type=str, default=None,
                        choices=['indoor', 'outdoor', 'forest',
                                 '25_10_20_14_30', '25_10_20_14_50', '25_11_04_16_00'],
                        help='Wheel dataset name (indoor/outdoor/forest or full name)')

    # TartanAir dataset options
    parser.add_argument('--ta-scene', type=str, default=None,
                        help='TartanAir scene name (e.g., abandonedfactory, amusement)')
    parser.add_argument('--ta-level', type=str, default='Easy',
                        choices=['Easy', 'Hard'],
                        help='TartanAir difficulty level')
    parser.add_argument('--ta-num', type=int, default=0,
                        help='TartanAir trajectory number (e.g., 0 for P000)')

    # MS2 dataset options
    parser.add_argument('--ms2-timestamp', type=str, default=None,
                        help='MS2 timestamp (sequence name)')
    parser.add_argument('--ms2-data-type', type=str, default='thr',
                        choices=['thr', 'rgb'],
                        help='MS2 data type (thr=thermal, rgb=visible)')

    # Options
    parser.add_argument('--start-frame', type=int, default=0,
                        help='Frame index to start evaluation from (default: 0)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum frames to process')
    parser.add_argument('--encoder', type=str, default='vitl',
                        choices=['vits', 'vitb', 'vitl'],
                        help='DepthAnything encoder size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Custom output base directory')
    parser.add_argument('--traj-update-interval', type=int, default=10,
                        help='Update trajectory image every N frames')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')
    parser.add_argument('--no-baseline-guard', action='store_true',
                        help='Disable baseline guard (skip triangulation when baseline too short)')
    parser.add_argument('--min-baseline', type=float, default=0.1,
                        help='Minimum baseline for triangulation (default: 0.05m)')
    parser.add_argument('--baseline-ema-beta', type=float, default=0.9,
                        help='Baseline EMA smoothing factor (default: 0.9)')
    # Fusion ablation options
    parser.add_argument('--no-iterative', action='store_true',
                        help='Disable iterative refinement (ablation: w/o iter)')
    parser.add_argument('--iter2', action='store_true',
                        help='Use 2 iterative refinement iterations instead of 1')
    parser.add_argument('--no-segmentation', action='store_true',
                        help='Disable edge-aware segmentation (ablation: w/o seg)')
    parser.add_argument('--no-rgb-guide', action='store_true',
                        help='Disable RGB guiding (ablation: w/o rgb)')
    parser.add_argument('--no-metric-scale', action='store_true',
                        help='Disable metric scale estimation entirely (ablation: w/o scale)')
    parser.add_argument('--global-scale', action='store_true',
                        help='Use global scale instead of per-segment (ablation: global vs seg)')
    parser.add_argument('--no-magsac', action='store_true',
                        help='Use paper MAD-based scoring instead of MAGSAC++ (ablation: paper mode)')
    parser.add_argument('--use-gt-flow', action='store_true',
                        help='Use GT optical flow from dataset (TartanAir only)')
    parser.add_argument('--flow', type=str, default=None,
                        choices=['raft', 'raft_large', 'raft_small', 'neuflow', 'neuflow_mixed', 'neuflow_sintel', 'neuflow_things'],
                        help='Use external optical flow estimator (default: DIS). raft=best generalization')
    parser.add_argument('--save-depths', action='store_true',
                        help='Save depth maps (z_tri, z_refined) to npz for visualization')
    parser.add_argument('--odom-noise', type=float, default=0.0,
                        help='Relative noise level for odometry (0.1 = ±10%% random noise)')
    parser.add_argument('--use-gps-baseline', action='store_true',
                        help='Use GPS baseline instead of wheel odometry (Wheel dataset only)')
    parser.add_argument('--use-gt-pose', action='store_true',
                        help='Use GT pose fallback when rotation exceeds threshold')
    parser.add_argument('--gt-pose-threshold', type=float, default=3.0,
                        help='Rotation threshold (degrees) for GT pose fallback (default: 3.0)')
    parser.add_argument('--skip-temporal-fusion', action='store_true',
                        help='Skip Bayesian update, only triangulation + solve_metric (ablation)')
    parser.add_argument('--use-gt-R', action='store_true',
                        help='Always use GT rotation when provided (ablation)')

    # Pixel-count thresholds (for different resolutions)
    parser.add_argument('--min-scale-overlap', type=int, default=2000,
                        help='Minimum overlapping pixels for scale matching')
    parser.add_argument('--seg-min-size', type=int, default=200,
                        help='Minimum segment size for segmentation')
    parser.add_argument('--max-points', type=int, default=2000,
                        help='Maximum points for motion estimation')

    args = parser.parse_args()

    # Validate arguments
    if args.exp_name is None and args.baseline is None:
        parser.error("Either --exp-name (-e) or --baseline (-b) is required")

    # Validate wheel dataset argument
    if args.dataset == 'wheel' and args.wheel_name is None:
        parser.error("--wheel-name (-w) is required for wheel dataset")

    # Validate TartanAir dataset arguments
    if args.dataset == 'tartanair' and args.ta_scene is None:
        parser.error("--ta-scene is required for tartanair dataset")

    # Validate MS2 dataset arguments
    if args.dataset == 'ms2' and args.ms2_timestamp is None:
        parser.error("--ms2-timestamp is required for ms2 dataset")

    # Compute metric_scale_mode from args
    # 0 = off (--no-metric-scale)
    # 1 = global (--global-scale)
    # 2 = per-segment (default)
    if args.no_metric_scale:
        metric_scale_mode = 0
    elif args.global_scale:
        metric_scale_mode = 1
    else:
        metric_scale_mode = 2

    # Load dataset based on type
    if args.dataset == 'kitti':
        # Dataset identifier
        dataset_id = f"kitti_{args.date}_{args.drive}"

        # Load dataset
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
        dataset_id = f"wheel_{wheel_name}"

        # Load dataset
        print(f"\n{'='*60}")
        print(f"Loading Wheel dataset: {wheel_name}")
        print(f"{'='*60}")

        loader = WheelLoader(name=wheel_name)
        dataset = WheelEvalWrapper(loader)
        print(f"Loaded {len(dataset)} frames")

    elif args.dataset == 'tartanair':
        dataset_id = f"tartanair_{args.ta_scene}_{args.ta_level}_P{args.ta_num:03d}"

        # Load dataset
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
        print(f"Loaded {len(dataset)} frames")

    else:  # ms2 dataset
        dataset_id = f"ms2_{args.ms2_timestamp}_{args.ms2_data_type}"

        # Load dataset
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

    # Setup
    base_dir = args.output_dir or str(PROJECT_ROOT / 'results')
    manager = ResultsManager(base_dir=base_dir)

    # Run PR-Depth experiment
    if args.exp_name:
        print(f"\n{'='*60}")
        print(f"PR-Depth Experiment: {args.exp_name}")
        print(f"{'='*60}")

        exp_dir = manager.create_experiment(args.exp_name, dataset_id)

        try:
            summary, trajectories = run_evaluation(
                method='pr_depth',
                exp_dir=exp_dir,
                dataset=dataset,
                device=args.device,
                encoder=args.encoder,
                start_frame=args.start_frame,
                max_frames=args.max_frames,
                traj_update_interval=args.traj_update_interval,
                quiet=args.quiet,
                results_base_dir=base_dir,
                dataset_id=dataset_id,
                use_baseline_guard=not args.no_baseline_guard,
                min_baseline=args.min_baseline,
                baseline_ema_beta=args.baseline_ema_beta,
                use_iterative=not args.no_iterative,
                iterative_iters=2 if args.iter2 else 1,
                use_segmentation=not args.no_segmentation,
                use_rgb_guide=not args.no_rgb_guide,
                metric_scale_mode=metric_scale_mode,
                use_gt_flow=args.use_gt_flow,
                flow_method=args.flow,
                save_depths=args.save_depths,
                odom_noise=args.odom_noise,
                use_magsac_scoring=not args.no_magsac,
                use_gps_baseline=args.use_gps_baseline,
                use_gt_pose_fallback=args.use_gt_pose,
                gt_pose_rotation_threshold_deg=args.gt_pose_threshold,
                skip_temporal_fusion=args.skip_temporal_fusion,
                use_gt_R=args.use_gt_R,
                # Pixel-count thresholds
                min_scale_overlap=args.min_scale_overlap,
                seg_min_size=args.seg_min_size,
                max_points=args.max_points,
            )

            # Save final summary
            if args.dataset == 'kitti':
                dataset_info = {'date': args.date, 'drive': args.drive, 'frames': summary.total_frames}
            elif args.dataset == 'wheel':
                dataset_info = {'name': wheel_name, 'type': args.wheel_name, 'frames': summary.total_frames}
            elif args.dataset == 'tartanair':
                dataset_info = {'scene': args.ta_scene, 'level': args.ta_level, 'num': args.ta_num, 'frames': summary.total_frames}
            else:  # ms2
                dataset_info = {'timestamp': args.ms2_timestamp, 'data_type': args.ms2_data_type, 'frames': summary.total_frames}
            manager.save_summary(summary, exp_dir, dataset_info=dataset_info)
            manager.save_trajectory_tum(trajectories['gt_positions'], exp_dir, 'gt_traj.txt')
            manager.save_trajectory_tum(trajectories['est_positions'], exp_dir, 'est_traj.txt')

            print_summary('pr_depth', summary)
            print(f"\nResults saved to: {exp_dir}")
            print(f"Latest symlink: {manager.get_latest(dataset_id)}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    # Run baselines
    if args.baseline:
        for method in args.baseline:
            print(f"\n{'='*60}")
            print(f"Baseline: {method.upper()}")
            print(f"{'='*60}")

            exp_dir = manager.create_baseline(method, dataset_id)

            # Check if already exists
            summary_file = exp_dir / 'summary.json'
            if summary_file.exists():
                print(f"Already exists: {exp_dir}")
                print("Skipping (delete folder to re-run)")
                continue

            try:
                summary, trajectories = run_evaluation(
                    method=method,
                    exp_dir=exp_dir,
                    dataset=dataset,
                    device=args.device,
                    encoder=args.encoder,
                    start_frame=args.start_frame,
                    max_frames=args.max_frames,
                    traj_update_interval=args.traj_update_interval,
                    quiet=args.quiet,
                    results_base_dir=base_dir,
                    dataset_id=dataset_id,
                    use_gps_baseline=args.use_gps_baseline,
                )

                if args.dataset == 'kitti':
                    dataset_info = {'date': args.date, 'drive': args.drive, 'frames': summary.total_frames}
                elif args.dataset == 'wheel':
                    dataset_info = {'name': wheel_name, 'type': args.wheel_name, 'frames': summary.total_frames}
                elif args.dataset == 'tartanair':
                    dataset_info = {'scene': args.ta_scene, 'level': args.ta_level, 'num': args.ta_num, 'frames': summary.total_frames}
                else:  # ms2
                    dataset_info = {'timestamp': args.ms2_timestamp, 'data_type': args.ms2_data_type, 'frames': summary.total_frames}
                manager.save_summary(summary, exp_dir, dataset_info=dataset_info)
                manager.save_trajectory_tum(trajectories['gt_positions'], exp_dir, 'gt_traj.txt')
                manager.save_trajectory_tum(trajectories['est_positions'], exp_dir, 'est_traj.txt')

                print_summary(method, summary)
                print(f"\nResults saved to: {exp_dir}")

            except Exception as e:
                print(f"Error evaluating {method}: {e}")
                import traceback
                traceback.print_exc()

    # Show folder structure
    print(f"\n{'='*60}")
    print(f"Results location: {base_dir}/{dataset_id}/")
    print(f"{'='*60}")

    experiments = manager.list_experiments(dataset_id)
    baselines = manager.list_baselines(dataset_id)

    if experiments:
        print(f"\nPR-Depth experiments:")
        for exp in experiments:
            print(f"  - {exp}")
        latest = manager.get_latest(dataset_id)
        if latest:
            print(f"  latest -> {latest.name}")

    if baselines:
        print(f"\nBaselines:")
        for b in baselines:
            print(f"  - {b}")


if __name__ == '__main__':
    main()
