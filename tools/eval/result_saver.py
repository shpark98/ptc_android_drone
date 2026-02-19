"""
Result saving utilities for evaluation scripts.

Experiment-based structure (2026-01-12 redesigned):

results/experiments/<exp_name>/
    config.json           # Experiment configuration
    summary.md            # Summary report
    depth/                # Depth metrics
    pose/                 # Pose metrics
    trajectory/           # Trajectory visualizations
    temporal/             # Temporal plots
    raw/                  # Raw data (npz, csv)

Usage:
    # Create experiment
    exp = Experiment("iterative_v1", description="Test iterative refinement")
    exp.set_config({...})

    # Save results
    exp.save_depth_metrics(...)
    exp.save_pose_metrics(...)

    # Finalize
    exp.save_summary()
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_results_root() -> Path:
    """Get the root results directory."""
    repo_root = Path(__file__).parent.parent.parent
    return repo_root / "results"


def get_timestamp() -> str:
    """Get current timestamp for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_date_str() -> str:
    """Get current date string."""
    return datetime.now().strftime("%Y%m%d")


class Experiment:
    """
    Experiment-based result saver.

    All results for one experiment are stored in:
        results/experiments/<exp_name>/

    Example:
        exp = Experiment("iterative_v1", description="Test iterative refinement")
        exp.set_config({"iterative_iters": 1, "dc_threshold": 0.65})
        exp.save_pose_metrics("pr_depth", "0001", rot_errors, trans_errors)
        exp.save_summary()
    """

    def __init__(self, name: str, description: str = "", create: bool = True):
        """
        Initialize experiment.

        Args:
            name: Experiment name (e.g., "iterative_v1", "baseline", "dc_sweep")
                  Timestamp will be prepended automatically.
            description: Short description of the experiment
            create: Whether to create directories
        """
        self.timestamp = get_date_str()
        self.full_name = f"{self.timestamp}_{name}"
        self.name = name
        self.description = description
        self.created_at = datetime.now().isoformat()

        # Root directory for this experiment
        self.root = get_results_root() / "experiments" / self.full_name

        if create:
            self._create_dirs()
            self._init_config()

        # Storage for summary
        self.results = {
            'depth': {},
            'pose': {},
            'info': {}
        }

    def _create_dirs(self):
        """Create experiment directory structure."""
        subdirs = ['depth', 'pose', 'trajectory', 'temporal', 'raw']
        for subdir in subdirs:
            (self.root / subdir).mkdir(parents=True, exist_ok=True)

    def _init_config(self):
        """Initialize config file."""
        config_path = self.root / "config.json"
        if not config_path.exists():
            config = {
                'name': self.name,
                'full_name': self.full_name,
                'description': self.description,
                'created_at': self.created_at,
                'parameters': {}
            }
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

    def set_config(self, parameters: Dict[str, Any]):
        """Set experiment configuration parameters."""
        config_path = self.root / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        config['parameters'] = parameters
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[Experiment] Config saved: {self.full_name}/config.json")

    def get_path(self, category: str) -> Path:
        """Get path for a specific category."""
        return self.root / category

    # =========================================================================
    # Depth metrics
    # =========================================================================
    def save_depth_metrics(
        self,
        method: str,
        sequence: str,
        metrics: Dict[str, List[float]],
        extra_info: Dict[str, Any] = None
    ) -> Path:
        """Save depth metrics for a method/sequence combination."""
        data = {
            'method': method,
            'sequence': sequence,
            'experiment': self.full_name,
            'timestamp': get_timestamp(),
            'metrics': {}
        }

        # Compute statistics
        for key, values in metrics.items():
            if len(values) > 0:
                data['metrics'][key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }

        if extra_info:
            data['info'] = extra_info

        # Save JSON
        json_path = self.root / "depth" / f"{method}_{sequence}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Save raw NPZ
        npz_path = self.root / "raw" / f"{method}_{sequence}_depth.npz"
        np.savez(npz_path, **{k: np.array(v) for k, v in metrics.items()})

        # Store for summary
        key = f"{method}_{sequence}"
        self.results['depth'][key] = data['metrics']

        print(f"  Depth metrics: {self.full_name}/depth/{method}_{sequence}.json")
        return json_path

    # =========================================================================
    # Pose metrics
    # =========================================================================
    def save_pose_metrics(
        self,
        method: str,
        sequence: str,
        rot_errors: List[float],
        trans_errors: List[float],
        extra_info: Dict[str, Any] = None
    ) -> Path:
        """Save pose metrics for a method/sequence combination."""
        data = {
            'method': method,
            'sequence': sequence,
            'experiment': self.full_name,
            'timestamp': get_timestamp(),
            'rotation_error_deg': {
                'mean': float(np.mean(rot_errors)) if rot_errors else None,
                'std': float(np.std(rot_errors)) if rot_errors else None,
                'median': float(np.median(rot_errors)) if rot_errors else None,
                'count': len(rot_errors)
            },
            'translation_error_deg': {
                'mean': float(np.mean(trans_errors)) if trans_errors else None,
                'std': float(np.std(trans_errors)) if trans_errors else None,
                'median': float(np.median(trans_errors)) if trans_errors else None,
                'count': len(trans_errors)
            }
        }

        if extra_info:
            data['info'] = extra_info

        # Save JSON
        json_path = self.root / "pose" / f"{method}_{sequence}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Save raw NPZ
        npz_path = self.root / "raw" / f"{method}_{sequence}_pose.npz"
        np.savez(npz_path, rot_errors=rot_errors, trans_errors=trans_errors)

        # Store for summary
        key = f"{method}_{sequence}"
        self.results['pose'][key] = {
            'rot_mean': data['rotation_error_deg']['mean'],
            'rot_std': data['rotation_error_deg']['std'],
            'trans_mean': data['translation_error_deg']['mean'],
            'trans_std': data['translation_error_deg']['std'],
        }

        print(f"  Pose metrics: {self.full_name}/pose/{method}_{sequence}.json")
        return json_path

    # =========================================================================
    # Trajectory visualization
    # =========================================================================
    def save_trajectory(
        self,
        method: str,
        sequence: str,
        gt_positions: List[np.ndarray],
        est_positions: List[np.ndarray],
        title: str = None,
        suffix: str = ""
    ) -> Path:
        """Save trajectory plot."""
        gt_pos = np.array(gt_positions)
        est_pos = np.array(est_positions)

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot(gt_pos[:, 0], gt_pos[:, 2], 'b-', linewidth=2, label='Ground Truth')
        ax.plot(est_pos[:, 0], est_pos[:, 2], 'r--', linewidth=2, label=method)
        ax.scatter([gt_pos[0, 0]], [gt_pos[0, 2]], c='green', s=100, marker='o', label='Start')
        ax.scatter([gt_pos[-1, 0]], [gt_pos[-1, 2]], c='red', s=100, marker='x', label='End')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title(title or f'{method} - {sequence}')
        ax.legend()
        ax.axis('equal')
        ax.grid(True, alpha=0.3)

        fname = f"{method}_{sequence}{suffix}.png"
        plot_path = self.root / "trajectory" / fname
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Trajectory: {self.full_name}/trajectory/{fname}")
        return plot_path

    # =========================================================================
    # Per-frame data
    # =========================================================================
    def save_per_frame_csv(
        self,
        name: str,
        frame_indices: List[int],
        data: Dict[str, List[float]]
    ) -> Path:
        """Save per-frame data as CSV."""
        df_data = {'frame': frame_indices}
        df_data.update(data)
        df = pd.DataFrame(df_data)

        csv_path = self.root / "raw" / f"{name}_frames.csv"
        df.to_csv(csv_path, index=False)

        print(f"  Per-frame: {self.full_name}/raw/{name}_frames.csv")
        return csv_path

    # =========================================================================
    # Temporal plots
    # =========================================================================
    def save_temporal_plot(
        self,
        name: str,
        frame_indices: List[int],
        metrics: Dict[str, List[float]],
        title: str = None
    ):
        """Plot metrics over frames."""
        for metric_name, values in metrics.items():
            if len(values) == 0:
                continue

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(frame_indices[:len(values)], values, 'b-', linewidth=1, alpha=0.7)
            ax.axhline(np.mean(values), color='r', linestyle='--',
                       label=f'Mean: {np.mean(values):.4f}')

            ax.set_xlabel('Frame')
            ax.set_ylabel(metric_name)
            ax.set_title(title or f'{name} - {metric_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plot_path = self.root / "temporal" / f"{name}_{metric_name}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

        print(f"  Temporal plots: {self.full_name}/temporal/")

    # =========================================================================
    # Summary report
    # =========================================================================
    def save_summary(self, extra_content: str = "") -> Path:
        """Generate and save summary report."""
        lines = [
            f"# Experiment: {self.name}",
            "",
            f"**Full name:** {self.full_name}",
            f"**Description:** {self.description}",
            f"**Created:** {self.created_at}",
            "",
        ]

        # Load config
        config_path = self.root / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            if config.get('parameters'):
                lines.append("## Configuration")
                lines.append("```json")
                lines.append(json.dumps(config['parameters'], indent=2))
                lines.append("```")
                lines.append("")

        # Pose results
        if self.results['pose']:
            lines.append("## Pose Results")
            lines.append("")
            lines.append("| Method | Sequence | Rot Error (deg) | Trans Error (deg) |")
            lines.append("|--------|----------|-----------------|-------------------|")
            for key, metrics in self.results['pose'].items():
                parts = key.rsplit('_', 1)
                method = parts[0]
                seq = parts[1] if len(parts) > 1 else "unknown"
                rot = f"{metrics['rot_mean']:.3f} +/- {metrics['rot_std']:.3f}"
                trans = f"{metrics['trans_mean']:.3f} +/- {metrics['trans_std']:.3f}"
                lines.append(f"| {method} | {seq} | {rot} | {trans} |")
            lines.append("")

        # Depth results
        if self.results['depth']:
            lines.append("## Depth Results")
            lines.append("")
            for key, metrics in self.results['depth'].items():
                lines.append(f"### {key}")
                for metric, stats in metrics.items():
                    if isinstance(stats, dict):
                        lines.append(f"- **{metric}**: {stats['mean']:.4f} +/- {stats['std']:.4f}")
                lines.append("")

        if extra_content:
            lines.append("## Notes")
            lines.append(extra_content)
            lines.append("")

        # Write summary
        summary_path = self.root / "summary.md"
        with open(summary_path, 'w') as f:
            f.write("\n".join(lines))

        print(f"[Experiment] Summary: {self.full_name}/summary.md")
        return summary_path

    @classmethod
    def list_experiments(cls) -> List[str]:
        """List all experiments."""
        exp_root = get_results_root() / "experiments"
        if not exp_root.exists():
            return []
        return sorted([d.name for d in exp_root.iterdir() if d.is_dir()])

    @classmethod
    def load(cls, name: str) -> 'Experiment':
        """Load existing experiment by name."""
        return cls(name.split('_', 1)[1] if '_' in name else name, create=False)


def list_experiments():
    """List all experiments."""
    experiments = Experiment.list_experiments()
    if not experiments:
        print("No experiments found.")
        return

    print(f"\n{'='*60}")
    print("EXPERIMENTS")
    print(f"{'='*60}")

    for exp_name in experiments:
        exp_root = get_results_root() / "experiments" / exp_name
        config_path = exp_root / "config.json"

        desc = ""
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            desc = config.get('description', '')

        print(f"\n{exp_name}/")
        if desc:
            print(f"  {desc}")

        # Count files
        for subdir in ['depth', 'pose', 'trajectory', 'temporal', 'raw']:
            subdir_path = exp_root / subdir
            if subdir_path.exists():
                files = list(subdir_path.glob('*'))
                if files:
                    print(f"  {subdir}/: {len(files)} files")


if __name__ == "__main__":
    list_experiments()
