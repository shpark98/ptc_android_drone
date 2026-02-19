"""Rerun 3D depth visualization viewer.

Usage:
    # Sequential — frame-by-frame GT vs PR-Depth comparison (default)
    python -m tools.viz.rerun_depth.viewer \
        --dataset wheel -w indoor --start 35 --end 200 \
        --depths gt pr_depth --spawn

    # SLAM mode — temporal accumulation
    python -m tools.viz.rerun_depth.viewer \
        --dataset kitti --date 2011_09_26 --drive 0052 \
        --mode slam --start 1 --end 50 --step 2 \
        --depths gt pr_depth --color-mode rgb

    # MS2 thermal
    python -m tools.viz.rerun_depth.viewer \
        --dataset ms2 --ms2-timestamp <ts> --ms2-data-type thr \
        --mode slam --start 5700 --end 5720 \
        --depths gt da_v2

Available depth methods: gt, pr_depth, da_v2, vda, unidepth
"""

import sys
import argparse
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add cpp build to path
cpp_build = PROJECT_ROOT / "cpp" / "build"
if cpp_build.exists() and str(cpp_build) not in sys.path:
    sys.path.insert(0, str(cpp_build))


def create_dataset(args):
    """Create dataset from CLI arguments."""
    from configs import get_dataset_paths

    if args.dataset == "kitti":
        from dataloader.dataset.kitti import KITTIEigenSplit
        paths = get_dataset_paths("kitti")
        dataset = KITTIEigenSplit(
            rgb_path=paths["rgb_path"],
            depth_path=paths.get("depth_path"),
            date=args.date,
            drive=args.drive,
        )
        print(f"KITTI {args.date}/{args.drive}: {len(dataset)} frames")

    elif args.dataset == "ms2":
        from dataloader.dataset.ms2 import MS2Loader
        paths = get_dataset_paths("ms2")
        dataset = MS2Loader(
            dataset_path=paths["dataset_path"],
            timestamp=args.ms2_timestamp,
            data_type=args.ms2_data_type,
        )
        print(f"MS2 {args.ms2_timestamp} ({args.ms2_data_type}): {len(dataset)} frames")

    elif args.dataset == "tartanair":
        from dataloader.dataset.tartanair import TartanairLoader
        paths = get_dataset_paths("tartanair")
        dataset = TartanairLoader(
            dataset_path=paths["dataset_path"],
            scene=args.ta_scene,
            level=args.ta_level,
            num=args.ta_num,
        )
        print(f"TartanAir {args.ta_scene}/{args.ta_level}/P{args.ta_num:03d}: {len(dataset)} frames")

    elif args.dataset == "euroc":
        from dataloader.dataset.euroc import EuRoCLoader
        paths = get_dataset_paths("euroc")
        dataset = EuRoCLoader(
            dataset_path=paths["dataset_path"],
            sequence=args.euroc_seq,
        )
        print(f"EuRoC {args.euroc_seq}: {len(dataset)} frames")

    elif args.dataset == "wheel":
        from dataloader import WheelLoader, WheelEvalWrapper
        WHEEL_DATASETS = {
            'indoor': '25_10_20_14_50',
            'outdoor': '25_10_20_14_30',
            'forest': '25_11_04_16_00',
        }
        wheel_name = WHEEL_DATASETS.get(args.wheel_name, args.wheel_name)
        loader = WheelLoader(name=wheel_name)
        dataset = WheelEvalWrapper(loader)
        print(f"Wheel {args.wheel_name} ({wheel_name}): {len(dataset)} frames")

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return dataset


def run_seq(source, logger, args):
    """Sequential mode: frame-by-frame depth comparison (non-accumulating)."""
    import rerun as rr
    from .blueprint import make_seq_blueprint

    method_names = [m.name for m in source.methods.values()]
    if "gt" in args.depths:
        method_names = ["GT"] + method_names

    blueprint = make_seq_blueprint(method_names)
    rr.send_blueprint(blueprint)

    start = args.start
    end = min(args.end, len(source)) if args.end else len(source)
    step = args.step

    print(f"Sequential mode: frames {start} to {end}, step {step}")
    print(f"Methods: {', '.join(method_names)}")
    print(f"Color: {args.color_mode}, Subsample: {args.subsample}\n")

    source.reset()

    n_logged = 0
    for idx in range(start, end, step):
        t0 = time.time()
        frame = source.get_frame(idx)

        if frame is None:
            print(f"  Frame {idx}: skipped (load error)")
            continue

        logger.log_frame_seq(
            frame,
            color_mode=args.color_mode,
            subsample=args.subsample,
            max_depth=args.max_depth,
            point_size=args.point_size,
        )

        dt = time.time() - t0
        src_info = ", ".join(
            f"{s.name}:{s.mask.sum():,}" for s in frame.depth_sources
        )
        print(f"  Frame {idx}: {src_info} ({dt:.2f}s)")
        n_logged += 1

    print(f"\nDone. {n_logged} frames logged.")
    print("Use the timeline slider in Rerun to scrub through frames.\n")

    if args.save_rrd:
        print(f"Saved to {args.save_rrd}")
    else:
        print("Keep running to view. Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nDone.")


def run_slam(source, logger, args):
    """SLAM mode: temporal accumulation of point clouds."""
    import rerun as rr
    from .blueprint import make_slam_blueprint

    method_names = [m.name for m in source.methods.values()]
    if "gt" in args.depths:
        method_names = ["GT"] + method_names

    blueprint = make_slam_blueprint(method_names)
    rr.send_blueprint(blueprint)

    start = args.start
    end = min(args.end, len(source)) if args.end else len(source)
    step = args.step

    print(f"SLAM mode: frames {start} to {end}, step {step}")
    print(f"Methods: {', '.join(method_names)}")
    print(f"Color: {args.color_mode}, Subsample: {args.subsample}\n")

    source.reset()

    for idx in range(start, end, step):
        t0 = time.time()
        frame = source.get_frame(idx)

        if frame is None:
            print(f"  Frame {idx}: skipped (load error)")
            continue

        logger.log_frame(
            frame,
            color_mode=args.color_mode,
            subsample=args.subsample,
            max_depth=args.max_depth,
            point_size=args.point_size,
        )

        dt = time.time() - t0
        n_pts = sum(src.mask.sum() for src in frame.depth_sources)
        print(f"  Frame {idx}: {n_pts:,} total points ({dt:.2f}s)")

    print(f"\nDone. {(end - start) // step} frames logged.")

    if args.save_rrd:
        print(f"Saved to {args.save_rrd}")
    else:
        print("Keep running to view in browser. Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Rerun 3D depth visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode
    parser.add_argument("--mode", choices=["seq", "slam"], default="seq",
                        help="seq: frame-by-frame comparison (default), slam: accumulation")

    # Dataset
    parser.add_argument("--dataset", required=True,
                        choices=["kitti", "ms2", "tartanair", "euroc", "wheel"],
                        help="Dataset type")

    # KITTI args
    parser.add_argument("--date", default="2011_09_26", help="KITTI date")
    parser.add_argument("--drive", default="0052", help="KITTI drive")

    # MS2 args
    parser.add_argument("--ms2-timestamp", help="MS2 timestamp")
    parser.add_argument("--ms2-data-type", default="thr",
                        choices=["thr", "rgb"], help="MS2 data type")

    # TartanAir args
    parser.add_argument("--ta-scene", help="TartanAir scene")
    parser.add_argument("--ta-level", default="Easy", help="TartanAir difficulty")
    parser.add_argument("--ta-num", type=int, default=0, help="TartanAir trajectory")

    # EuRoC args
    parser.add_argument("--euroc-seq", help="EuRoC sequence name")

    # Wheel args
    parser.add_argument("--wheel-name", "-w", help="Wheel dataset name (indoor/outdoor/forest or raw name)")

    # Depth methods
    parser.add_argument("--depths", nargs="+", default=["gt"],
                        help="Depth sources: gt, pr_depth, da_v2, vda, unidepth")

    # Frame selection
    parser.add_argument("--start", type=int, default=0,
                        help="Start frame index")
    parser.add_argument("--end", type=int, default=None,
                        help="End frame index (default: all)")
    parser.add_argument("--step", type=int, default=1,
                        help="Frame step")

    # Visualization options
    parser.add_argument("--color-mode", default="rgb",
                        choices=["rgb", "turbo", "viridis", "plasma", "inferno"],
                        help="Point cloud coloring mode")
    parser.add_argument("--max-depth", type=float, default=80.0,
                        help="Maximum depth in meters")
    parser.add_argument("--subsample", type=int, default=1,
                        help="Subsample factor (1=all, 2=every 2nd pixel)")
    parser.add_argument("--point-size", type=float, default=5.0,
                        help="Point size in pixels")

    # Runtime
    parser.add_argument("--device", default="cuda", help="Compute device")
    parser.add_argument("--encoder", default="vitl",
                        choices=["vits", "vitb", "vitl"],
                        help="DepthAnything encoder")
    parser.add_argument("--port", type=int, default=9876,
                        help="Rerun web viewer port")
    parser.add_argument("--spawn", action="store_true",
                        help="Spawn native viewer instead of web server")
    parser.add_argument("--save-rrd", type=str, default=None,
                        help="Save to .rrd file instead of serving (e.g. output.rrd)")

    args = parser.parse_args()

    # Build methods
    from .data_source import build_methods, DatasetSource
    from .logger import RerunLogger

    print("=" * 60)
    print("Rerun 3D Depth Viewer")
    print("=" * 60)

    # Create dataset
    dataset = create_dataset(args)

    # Build depth methods
    method_names = [d for d in args.depths if d != "gt"]
    methods = {}
    if method_names:
        print(f"Initializing methods: {method_names}")
        methods = build_methods(method_names, device=args.device, encoder=args.encoder)

    include_gt = "gt" in args.depths

    # Create data source
    source = DatasetSource(
        dataset=dataset,
        methods=methods,
        include_gt=include_gt,
        max_depth=args.max_depth,
    )

    # Initialize Rerun
    logger = RerunLogger(app_name=f"pr_depth_{args.dataset}")
    logger.init(port=args.port, serve=not args.spawn, save_path=args.save_rrd)

    # Run visualization
    if args.mode == "seq":
        run_seq(source, logger, args)
    else:
        run_slam(source, logger, args)


if __name__ == "__main__":
    main()
