"""Evaluation framework for depth and pose estimation.

This module provides a unified evaluation framework with:
- Base classes for method runners and datasets
- Main Evaluator class for running evaluations
- Concrete runners for various methods (PR-Depth, MADPose, etc.)
- Results management (save/load)
- Visualization utilities

Example usage:
    >>> from src.evaluation import Evaluator, ResultsManager, Visualizer
    >>> from src.evaluation.runners import PRDepthRunner
    >>> from dataloader import KITTIEigenSplit
    >>> from src.estimators.depth import DepthAnythingEstimator
    >>>
    >>> # Setup
    >>> dataset = KITTIEigenSplit(...)
    >>> runner = PRDepthRunner(device='cuda')
    >>> depth_estimator = DepthAnythingEstimator(encoder='vitl')
    >>>
    >>> # Run evaluation
    >>> evaluator = Evaluator(runner, dataset, depth_estimator)
    >>> summary = evaluator.run(max_frames=100)
    >>>
    >>> # Save results
    >>> manager = ResultsManager()
    >>> exp_dir = manager.create_experiment('my_experiment')
    >>> manager.save_summary(summary, exp_dir)
    >>> manager.save_frame_metrics(evaluator.frame_metrics, exp_dir)
    >>>
    >>> # Visualize
    >>> viz = Visualizer()
    >>> traj = evaluator.get_trajectories()
    >>> viz.plot_trajectory(traj['gt_positions'], traj['est_positions'], 'PR-Depth', exp_dir / 'traj.png')
"""

# Core metrics functions
from .metrics import (
    compute_depth_metrics,
    compute_pose_error,
    compute_ate,
    integrate_trajectory,
)

# Base classes
from .base import (
    BaseMethodRunner,
    BaseDataset,
    PoseResult,
    FrameMetrics,
    EvalSummary,
)

# Main evaluator
from .evaluator import Evaluator

# Results management
from .results import ResultsManager

# Visualization
from .visualizer import Visualizer, enable_interactive, disable_interactive

# Method runners
from . import runners

__all__ = [
    # Metrics
    'compute_depth_metrics',
    'compute_pose_error',
    'compute_ate',
    'integrate_trajectory',
    # Base classes
    'BaseMethodRunner',
    'BaseDataset',
    'PoseResult',
    'FrameMetrics',
    'EvalSummary',
    # Main components
    'Evaluator',
    'ResultsManager',
    'Visualizer',
    'enable_interactive',
    'disable_interactive',
    # Runners module
    'runners',
]
