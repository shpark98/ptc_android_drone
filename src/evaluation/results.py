"""Results management for saving/loading evaluation results."""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from .base import EvalSummary, FrameMetrics


class ResultsManager:
    """Manages saving and loading of evaluation results.

    Folder structure:
        results/
        └── {dataset}/                    # e.g., kitti_2011_10_03_0027
            ├── baselines/                # Other methods (run once)
            │   ├── madpose/
            │   ├── batrack/
            │   └── monst3r/
            └── pr_depth/                 # Our method (tuning)
                ├── v1_initial/
                ├── v2_dc_reject_fix/
                └── latest -> v2_dc_reject_fix/  # symlink to latest

    Example:
        >>> manager = ResultsManager(base_dir='results')
        >>> # For baseline methods
        >>> exp_dir = manager.create_baseline('madpose', 'kitti_2011_10_03_0027')
        >>> # For PR-Depth experiments
        >>> exp_dir = manager.create_experiment('v2_dc_reject_fix', 'kitti_2011_10_03_0027')
    """

    def __init__(self, base_dir: str = "results"):
        """Initialize results manager.

        Args:
            base_dir: Base directory for results
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment(
        self,
        exp_name: str,
        dataset: str,
        update_latest: bool = True,
    ) -> Path:
        """Create PR-Depth experiment directory.

        Structure: results/{dataset}/pr_depth/{exp_name}/

        Args:
            exp_name: Experiment name (e.g., 'v1_initial', 'v2_dc_fix')
            dataset: Dataset identifier (e.g., 'kitti_2011_10_03_0027')
            update_latest: Whether to update 'latest' symlink

        Returns:
            Path to experiment directory
        """
        exp_dir = self.base_dir / dataset / 'pr_depth' / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Update 'latest' symlink
        if update_latest:
            latest_link = self.base_dir / dataset / 'pr_depth' / 'latest'
            if latest_link.is_symlink():
                latest_link.unlink()
            elif latest_link.exists():
                import shutil
                shutil.rmtree(latest_link)
            latest_link.symlink_to(exp_name)

        return exp_dir

    def create_baseline(
        self,
        method: str,
        dataset: str,
    ) -> Path:
        """Create baseline method directory.

        Structure: results/{dataset}/baselines/{method}/

        Args:
            method: Method name (e.g., 'madpose', 'batrack', 'monst3r')
            dataset: Dataset identifier

        Returns:
            Path to baseline directory
        """
        exp_dir = self.base_dir / dataset / 'baselines' / method
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def get_latest(self, dataset: str) -> Optional[Path]:
        """Get latest PR-Depth experiment directory.

        Args:
            dataset: Dataset identifier

        Returns:
            Path to latest experiment, or None if not found
        """
        latest_link = self.base_dir / dataset / 'pr_depth' / 'latest'
        if latest_link.exists():
            return latest_link.resolve()
        return None

    def save_frame_metrics(
        self,
        metrics: List[FrameMetrics],
        exp_dir: Path,
        filename: str = "frame_metrics.csv"
    ):
        """Save per-frame metrics to CSV.

        Args:
            metrics: List of FrameMetrics
            exp_dir: Experiment directory
            filename: Output filename
        """
        df = pd.DataFrame([m.to_dict() for m in metrics])
        csv_path = exp_dir / filename
        df.to_csv(csv_path, index=False)
        return csv_path

    def load_frame_metrics(self, exp_dir: Path, filename: str = "frame_metrics.csv") -> pd.DataFrame:
        """Load per-frame metrics from CSV.

        Args:
            exp_dir: Experiment directory
            filename: CSV filename

        Returns:
            DataFrame with frame metrics
        """
        csv_path = exp_dir / filename
        return pd.read_csv(csv_path)

    def save_trajectory(
        self,
        gt_positions: np.ndarray,
        est_positions: np.ndarray,
        exp_dir: Path,
        filename: str = "trajectory.npz"
    ):
        """Save trajectories to NPZ file.

        Args:
            gt_positions: (N, 3) ground truth positions
            est_positions: (N, 3) estimated positions
            exp_dir: Experiment directory
            filename: Output filename
        """
        npz_path = exp_dir / filename
        np.savez(
            npz_path,
            gt_positions=gt_positions,
            est_positions=est_positions,
        )
        return npz_path

    def load_trajectory(self, exp_dir: Path, filename: str = "trajectory.npz") -> Dict[str, np.ndarray]:
        """Load trajectories from NPZ file.

        Args:
            exp_dir: Experiment directory
            filename: NPZ filename

        Returns:
            Dict with 'gt_positions' and 'est_positions'
        """
        npz_path = exp_dir / filename
        data = np.load(npz_path)
        return {
            'gt_positions': data['gt_positions'],
            'est_positions': data['est_positions'],
        }

    def save_trajectory_tum(
        self,
        positions: np.ndarray,
        exp_dir: Path,
        filename: str,
        orientations: Optional[List[np.ndarray]] = None,
    ):
        """Save trajectory in TUM format for evo evaluation.

        Format: timestamp tx ty tz qx qy qz qw

        Args:
            positions: (N, 3) positions
            exp_dir: Experiment directory
            filename: Output filename
            orientations: Optional list of 3x3 rotation matrices
        """
        from scipy.spatial.transform import Rotation

        tum_path = exp_dir / filename
        with open(tum_path, 'w') as f:
            for i, pos in enumerate(positions):
                if orientations is not None and i < len(orientations):
                    quat = Rotation.from_matrix(orientations[i]).as_quat()  # xyzw
                else:
                    quat = [0, 0, 0, 1]

                f.write(f"{float(i):.6f} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                       f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n")

        return tum_path

    def save_summary(
        self,
        summary: EvalSummary,
        exp_dir: Path,
        dataset_info: Optional[Dict] = None
    ):
        """Save evaluation summary to text and JSON files.

        Args:
            summary: EvalSummary object
            exp_dir: Experiment directory
            dataset_info: Optional additional dataset information
        """
        # Save text summary
        txt_path = exp_dir / 'summary.txt'
        with open(txt_path, 'w') as f:
            f.write(f"Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Method: {summary.method_name}\n")
            f.write(f"Dataset: {summary.dataset_name}\n")
            if dataset_info:
                for k, v in dataset_info.items():
                    f.write(f"  {k}: {v}\n")
            f.write(f"\nFrames: {summary.total_frames} ({summary.success_frames} successful)\n")
            f.write(f"Time: {summary.elapsed_time:.1f}s ({summary.fps:.2f} FPS)\n\n")

            f.write("Pose Estimation\n")
            f.write("-" * 40 + "\n")
            if summary.rot_error_mean is not None:
                f.write(f"  Rotation Error:    {summary.rot_error_mean:.3f}° (median: {summary.rot_error_median:.3f}°)\n")
            if summary.trans_error_mean is not None:
                f.write(f"  Translation Error: {summary.trans_error_mean:.3f}° (median: {summary.trans_error_median:.3f}°)\n")

            f.write("\nTrajectory (ATE)\n")
            f.write("-" * 40 + "\n")
            if summary.ATE_RMSE is not None:
                f.write(f"  RMSE: {summary.ATE_RMSE:.3f} m\n")
                f.write(f"  Mean: {summary.ATE_mean:.3f} m\n")
            if summary.final_drift is not None:
                f.write(f"  Final Drift: {summary.final_drift:.3f} m\n")
            if summary.trajectory_length is not None:
                f.write(f"  Trajectory Length: {summary.trajectory_length:.1f} m\n")

            f.write("\nDepth Metrics\n")
            f.write("-" * 40 + "\n")
            if summary.tri_d125_mean is not None:
                f.write(f"  Triangulation δ<1.25: {summary.tri_d125_mean:.1f}%\n")
            if summary.tri_MAE_mean is not None:
                f.write(f"  Triangulation MAE: {summary.tri_MAE_mean:.2f} m\n")
            if summary.ref_d125_mean is not None:
                f.write(f"  Refined δ<1.25: {summary.ref_d125_mean:.1f}%\n")
            if summary.ref_MAE_mean is not None:
                f.write(f"  Refined MAE: {summary.ref_MAE_mean:.2f} m\n")

        # Save JSON summary
        json_path = exp_dir / 'summary.json'
        summary_dict = {
            'method_name': summary.method_name,
            'dataset_name': summary.dataset_name,
            'total_frames': summary.total_frames,
            'success_frames': summary.success_frames,
            'elapsed_time': summary.elapsed_time,
            'fps': summary.fps,
            'rot_error_mean': summary.rot_error_mean,
            'rot_error_median': summary.rot_error_median,
            'trans_error_mean': summary.trans_error_mean,
            'trans_error_median': summary.trans_error_median,
            'ATE_RMSE': summary.ATE_RMSE,
            'ATE_mean': summary.ATE_mean,
            'final_drift': summary.final_drift,
            'trajectory_length': summary.trajectory_length,
            'tri_d125_mean': summary.tri_d125_mean,
            'tri_MAE_mean': summary.tri_MAE_mean,
            'ref_d125_mean': summary.ref_d125_mean,
            'ref_MAE_mean': summary.ref_MAE_mean,
        }
        if dataset_info:
            summary_dict['dataset_info'] = dataset_info
        if summary.extra:
            summary_dict['extra'] = summary.extra

        with open(json_path, 'w') as f:
            json.dump(summary_dict, f, indent=2, default=_json_serializer)

        return txt_path, json_path

    def load_summary(self, exp_dir: Path) -> Dict:
        """Load summary from JSON file.

        Args:
            exp_dir: Experiment directory

        Returns:
            Dictionary with summary data
        """
        json_path = exp_dir / 'summary.json'
        with open(json_path, 'r') as f:
            return json.load(f)

    def list_experiments(self, dataset: str) -> List[str]:
        """List PR-Depth experiment names.

        Args:
            dataset: Dataset identifier

        Returns:
            List of experiment names (sorted alphabetically)
        """
        pr_depth_dir = self.base_dir / dataset / 'pr_depth'
        if not pr_depth_dir.exists():
            return []

        experiments = []
        for path in pr_depth_dir.iterdir():
            if path.is_dir() and path.name != 'latest' and not path.is_symlink():
                experiments.append(path.name)

        return sorted(experiments)

    def list_baselines(self, dataset: str) -> List[str]:
        """List baseline methods.

        Args:
            dataset: Dataset identifier

        Returns:
            List of baseline method names
        """
        baselines_dir = self.base_dir / dataset / 'baselines'
        if not baselines_dir.exists():
            return []

        baselines = []
        for path in baselines_dir.iterdir():
            if path.is_dir():
                baselines.append(path.name)

        return sorted(baselines)

    def list_datasets(self) -> List[str]:
        """List available datasets."""
        datasets = []
        for path in self.base_dir.iterdir():
            if path.is_dir() and not path.name.startswith('.'):
                datasets.append(path.name)
        return sorted(datasets)

    def get_experiment_path(self, dataset: str, exp_name: str) -> Path:
        """Get path to a PR-Depth experiment."""
        return self.base_dir / dataset / 'pr_depth' / exp_name

    def get_baseline_path(self, dataset: str, method: str) -> Path:
        """Get path to a baseline method."""
        return self.base_dir / dataset / 'baselines' / method


def _json_serializer(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
