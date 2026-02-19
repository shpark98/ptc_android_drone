#!/usr/bin/env python3
"""Run ablation experiments for PR-Depth analysis."""

import subprocess
import sys
import os

# Ablation experiments configuration
ABLATION_EXPERIMENTS = {
    # DC threshold variations
    "dc_thresh_050": {
        "description": "DC threshold = 0.50 (looser)",
        "config_override": {"depth_consistency_threshold": 0.50}
    },
    "dc_thresh_055": {
        "description": "DC threshold = 0.55",
        "config_override": {"depth_consistency_threshold": 0.55}
    },
    "dc_thresh_060": {
        "description": "DC threshold = 0.60",
        "config_override": {"depth_consistency_threshold": 0.60}
    },
    "dc_thresh_070": {
        "description": "DC threshold = 0.70 (stricter)",
        "config_override": {"depth_consistency_threshold": 0.70}
    },
    "dc_thresh_075": {
        "description": "DC threshold = 0.75 (very strict)",
        "config_override": {"depth_consistency_threshold": 0.75}
    },
}

DATASETS = {
    "kitti": {
        "args": ["--dataset", "kitti", "--kitti-drive", "0027"],
        "max_frames": 500
    },
    "tartanair_seasonsforest": {
        "args": ["--dataset", "tartanair", "--ta-scene", "seasonsforest", "--ta-level", "Easy", "--ta-num", "1"],
        "max_frames": 320
    },
    "tartanair_abandonedfactory": {
        "args": ["--dataset", "tartanair", "--ta-scene", "abandonedfactory", "--ta-level", "Easy", "--ta-num", "0"],
        "max_frames": 500
    },
}

def run_experiment(exp_name, exp_config, dataset_name, dataset_config):
    """Run a single ablation experiment."""
    print(f"\n{'='*60}")
    print(f"Running: {exp_name} on {dataset_name}")
    print(f"Description: {exp_config['description']}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "tools/eval/evaluate.py",
        "-e", f"ablation_{exp_name}",
        "--max-frames", str(dataset_config["max_frames"]),
    ] + dataset_config["args"]

    # Add config overrides
    for key, value in exp_config.get("config_override", {}).items():
        cmd.extend(["--config-override", f"{key}={value}"])

    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--experiments", "-e", nargs="+",
                       choices=list(ABLATION_EXPERIMENTS.keys()) + ["all"],
                       default=["all"],
                       help="Which experiments to run")
    parser.add_argument("--datasets", "-d", nargs="+",
                       choices=list(DATASETS.keys()) + ["all"],
                       default=["all"],
                       help="Which datasets to use")
    args = parser.parse_args()

    # Resolve "all"
    experiments = list(ABLATION_EXPERIMENTS.keys()) if "all" in args.experiments else args.experiments
    datasets = list(DATASETS.keys()) if "all" in args.datasets else args.datasets

    print(f"Running {len(experiments)} experiments on {len(datasets)} datasets")
    print(f"Experiments: {experiments}")
    print(f"Datasets: {datasets}")

    results = {}
    for exp_name in experiments:
        for dataset_name in datasets:
            key = f"{exp_name}_{dataset_name}"
            success = run_experiment(
                exp_name,
                ABLATION_EXPERIMENTS[exp_name],
                dataset_name,
                DATASETS[dataset_name]
            )
            results[key] = success

    # Summary
    print("\n" + "="*60)
    print("ABLATION RESULTS SUMMARY")
    print("="*60)
    for key, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {key}")

if __name__ == "__main__":
    main()
