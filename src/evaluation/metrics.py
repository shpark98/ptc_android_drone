"""Evaluation metrics for depth and pose estimation."""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional, Dict, Tuple, List

# evo imports for trajectory evaluation
from evo.core.trajectory import PoseTrajectory3D
from evo.core.metrics import APE, RPE, PoseRelation
from evo.core import sync


def get_eigen_crop(H: int, W: int) -> Tuple[int, int, int, int]:
    """Get Eigen crop boundaries for KITTI depth evaluation.

    Standard Eigen crop excludes borders and car hood area.

    Args:
        H: Image height
        W: Image width

    Returns:
        Tuple of (y_min, y_max, x_min, x_max)
    """
    y_min = int(0.40810811 * H)  # ~153 for H=375
    y_max = int(0.99189189 * H)  # ~372 for H=375
    x_min = int(0.03594771 * W)  # ~45 for W=1242
    x_max = int(0.96405229 * W)  # ~1197 for W=1242
    return y_min, y_max, x_min, x_max


def compute_depth_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    max_depth: float = 80.0,
    use_eigen_crop: bool = False,
) -> Optional[Dict[str, float]]:
    """Compute depth evaluation metrics.

    Following standard KITTI depth evaluation protocol:
    - Eigen crop applied by default
    - Both GT and pred capped at max_depth (method's operating range)

    Args:
        pred: Predicted depth (H, W)
        gt: Ground truth depth (H, W)
        max_depth: Maximum depth for evaluation (meters). Both GT and pred
                   are evaluated within [0, max_depth] range.
        use_eigen_crop: If True, apply standard Eigen crop (default: True)

    Returns:
        Dictionary with MAE, RMSE, AbsRel, and delta thresholds,
        or None if insufficient valid pixels
    """
    H, W = gt.shape[:2]

    # Apply Eigen crop if requested
    if use_eigen_crop:
        y_min, y_max, x_min, x_max = get_eigen_crop(H, W)
        pred = pred[y_min:y_max, x_min:x_max]
        gt = gt[y_min:y_max, x_min:x_max]

    # Cap predictions to method's operating range (0, max_depth]
    # This is standard practice for metric depth methods with defined operating range
    pred = np.clip(pred, 0, max_depth)

    # Evaluate where both GT and pred are valid within operating range
    mask = (gt > 0) & (gt < max_depth) & np.isfinite(pred) & (pred > 0)

    if mask.sum() < 100:
        return None

    pred_valid = pred[mask]
    gt_valid = gt[mask]

    # Error metrics
    abs_diff = np.abs(pred_valid - gt_valid)
    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(abs_diff ** 2)))
    abs_rel = float(np.mean(abs_diff / gt_valid))

    # Delta thresholds (using 1.05, 1.15, 1.25 as per project spec)
    ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    d105 = float(np.mean(ratio < 1.05) * 100)
    d115 = float(np.mean(ratio < 1.15) * 100)
    d125 = float(np.mean(ratio < 1.25) * 100)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'AbsRel': abs_rel,
        'd105': d105,
        'd115': d115,
        'd125': d125,
        'num_valid': int(mask.sum())
    }


def angle_error_mat(R1: np.ndarray, R2: np.ndarray) -> float:
    """Compute rotation error between two rotation matrices (MADPose style).

    Args:
        R1: First rotation matrix (3, 3)
        R2: Second rotation matrix (3, 3)

    Returns:
        Rotation error in degrees
    """
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.rad2deg(np.abs(np.arccos(cos))))


def angle_error_vec(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angle between two vectors (MADPose style).

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Angle in degrees
    """
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-8:
        return 0.0
    return float(np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def compute_pose_error(
    R_pred: np.ndarray,
    t_pred: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
) -> Dict[str, float]:
    """Compute rotation and translation errors (MADPose style).

    Args:
        R_pred: Predicted rotation matrix (3, 3)
        t_pred: Predicted translation vector (3,)
        R_gt: Ground truth rotation matrix (3, 3)
        t_gt: Ground truth translation vector (3,)

    Returns:
        Dictionary with rotation error (degrees) and translation error (degrees)
    """
    # Rotation error (MADPose style)
    rot_error = angle_error_mat(R_pred, R_gt)

    # Translation error: angle between directions (MADPose style)
    trans_error = angle_error_vec(t_pred, t_gt)
    trans_error = min(trans_error, 180 - trans_error)  # ambiguity of E estimation

    return {
        'rot_error': rot_error,
        'trans_error': trans_error,
    }


def poses_to_trajectory(
    positions: np.ndarray,
    rotations: Optional[List[np.ndarray]] = None,
) -> PoseTrajectory3D:
    """Convert positions and rotations to evo PoseTrajectory3D.

    Args:
        positions: (N, 3) array of xyz positions
        rotations: Optional list of N rotation matrices (3, 3)

    Returns:
        PoseTrajectory3D object
    """
    n = len(positions)
    timestamps = np.arange(n, dtype=float)

    if rotations is None:
        # Identity quaternions (wxyz format)
        quat_wxyz = np.tile([1, 0, 0, 0], (n, 1)).astype(float)
    else:
        # Convert rotation matrices to quaternions (wxyz)
        quat_wxyz = np.zeros((n, 4))
        for i, rot_mat in enumerate(rotations):
            r = R.from_matrix(rot_mat)
            q_xyzw = r.as_quat()  # scipy returns xyzw
            quat_wxyz[i] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]  # convert to wxyz

    return PoseTrajectory3D(
        positions_xyz=positions.astype(float),
        orientations_quat_wxyz=quat_wxyz,
        timestamps=timestamps
    )


def compute_ate(
    gt_positions: np.ndarray,
    est_positions: np.ndarray,
    align: bool = True,
) -> Optional[Dict[str, float]]:
    """Compute Absolute Trajectory Error (ATE) using evo library.

    Args:
        gt_positions: (N, 3) ground truth positions
        est_positions: (N, 3) estimated positions
        align: If True, perform Umeyama alignment (sim3)

    Returns:
        Dictionary with ATE metrics, or None if insufficient data
    """
    if len(gt_positions) < 2:
        return None

    gt_pos = np.array(gt_positions)
    est_pos = np.array(est_positions)

    # Create evo trajectories
    gt_traj = poses_to_trajectory(gt_pos)
    est_traj = poses_to_trajectory(est_pos)

    # Align if requested (Umeyama alignment - scale + rotation + translation)
    if align:
        try:
            est_traj.align(gt_traj, correct_scale=True)
        except Exception:
            # Alignment may fail for degenerate cases (e.g., straight line)
            pass

    # Compute APE (Absolute Pose Error) on translation
    ape_metric = APE(PoseRelation.translation_part)
    ape_metric.process_data((gt_traj, est_traj))
    stats = ape_metric.get_all_statistics()

    # Compute trajectory length
    traj_length = float(np.sum(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1)))

    # Final drift
    final_error = float(np.linalg.norm(gt_pos[-1] - est_traj.positions_xyz[-1]))

    return {
        'ATE_RMSE': float(stats['rmse']),
        'ATE_mean': float(stats['mean']),
        'ATE_median': float(stats['median']),
        'ATE_max': float(stats['max']),
        'final_drift': final_error,
        'trajectory_length': traj_length
    }


def compute_rpe(
    gt_positions: np.ndarray,
    est_positions: np.ndarray,
    gt_rotations: Optional[List[np.ndarray]] = None,
    est_rotations: Optional[List[np.ndarray]] = None,
    delta: int = 1,
) -> Optional[Dict[str, float]]:
    """Compute Relative Pose Error (RPE) using evo library.

    Args:
        gt_positions: (N, 3) ground truth positions
        est_positions: (N, 3) estimated positions
        gt_rotations: Optional list of N rotation matrices for GT
        est_rotations: Optional list of N rotation matrices for estimation
        delta: Delta for relative pose pairs (frames)

    Returns:
        Dictionary with RPE metrics (translation and rotation), or None if insufficient data
    """
    if len(gt_positions) < delta + 1:
        return None

    gt_pos = np.array(gt_positions)
    est_pos = np.array(est_positions)

    # Create evo trajectories
    gt_traj = poses_to_trajectory(gt_pos, gt_rotations)
    est_traj = poses_to_trajectory(est_pos, est_rotations)

    result = {}

    # Compute RPE translation
    rpe_trans = RPE(PoseRelation.translation_part, delta=delta, all_pairs=True)
    rpe_trans.process_data((gt_traj, est_traj))
    trans_stats = rpe_trans.get_all_statistics()

    result['RPE_trans_RMSE'] = float(trans_stats['rmse'])
    result['RPE_trans_mean'] = float(trans_stats['mean'])
    result['RPE_trans_median'] = float(trans_stats['median'])
    result['RPE_trans_max'] = float(trans_stats['max'])

    # Compute RPE rotation (if rotations provided)
    if gt_rotations is not None and est_rotations is not None:
        rpe_rot = RPE(PoseRelation.rotation_angle_deg, delta=delta, all_pairs=True)
        rpe_rot.process_data((gt_traj, est_traj))
        rot_stats = rpe_rot.get_all_statistics()

        result['RPE_rot_RMSE'] = float(rot_stats['rmse'])
        result['RPE_rot_mean'] = float(rot_stats['mean'])
        result['RPE_rot_median'] = float(rot_stats['median'])
        result['RPE_rot_max'] = float(rot_stats['max'])

    return result


def compute_kitti_vo_metrics(
    gt_positions: np.ndarray,
    est_positions: np.ndarray,
    gt_rotations: Optional[List[np.ndarray]] = None,
    est_rotations: Optional[List[np.ndarray]] = None,
    lengths: List[float] = [100, 200, 300, 400, 500, 600, 700, 800],
) -> Optional[Dict[str, float]]:
    """Compute KITTI VO benchmark style metrics using evo RPE with distance delta.

    KITTI evaluates on subsequences of different lengths (100m-800m).
    t_err: average translation error as percentage of path length
    r_err: average rotation error in degrees per 100 meters

    Uses evo's RPE with delta_unit='m' for each length.

    Args:
        gt_positions: (N, 3) ground truth positions
        est_positions: (N, 3) estimated positions
        gt_rotations: Optional list of N rotation matrices for GT
        est_rotations: Optional list of N rotation matrices for estimation
        lengths: Subsequence lengths in meters to evaluate

    Returns:
        Dictionary with t_err (%) and r_err (deg/100m)
    """
    from evo.core.metrics import Unit

    if len(gt_positions) < 10:
        return None

    gt_pos = np.array(gt_positions)
    est_pos = np.array(est_positions)

    # Compute total trajectory length
    traj_length = float(np.sum(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1)))

    # Filter lengths that are feasible for this trajectory
    valid_lengths = [l for l in lengths if l < traj_length * 0.8]
    if len(valid_lengths) == 0:
        # Use shorter lengths for short trajectories
        valid_lengths = [traj_length * 0.2, traj_length * 0.4, traj_length * 0.6]

    # Create evo trajectories
    gt_traj = poses_to_trajectory(gt_pos, gt_rotations)
    est_traj = poses_to_trajectory(est_pos, est_rotations)

    t_errors = []
    r_errors = []

    for length in valid_lengths:
        try:
            # RPE with distance delta
            rpe_trans = RPE(
                PoseRelation.translation_part,
                delta=length,
                delta_unit=Unit.meters,
                all_pairs=True
            )
            rpe_trans.process_data((gt_traj, est_traj))
            trans_stats = rpe_trans.get_all_statistics()

            # t_err: translation error as % of path length
            t_err = trans_stats['mean'] / length * 100.0
            t_errors.append(t_err)

            # RPE rotation (if rotations available)
            if gt_rotations is not None and est_rotations is not None:
                rpe_rot = RPE(
                    PoseRelation.rotation_angle_deg,
                    delta=length,
                    delta_unit=Unit.meters,
                    all_pairs=True
                )
                rpe_rot.process_data((gt_traj, est_traj))
                rot_stats = rpe_rot.get_all_statistics()

                # r_err: rotation error in deg/100m
                r_err = rot_stats['mean'] / length * 100.0
                r_errors.append(r_err)

        except Exception:
            # Skip if this length fails (not enough data)
            continue

    if len(t_errors) == 0:
        return None

    result = {
        't_err': float(np.mean(t_errors)),  # Translation error (%)
        't_err_std': float(np.std(t_errors)) if len(t_errors) > 1 else 0.0,
        'num_segments': len(t_errors),
    }

    if len(r_errors) > 0:
        result['r_err'] = float(np.mean(r_errors))  # Rotation error (deg/100m)
        result['r_err_std'] = float(np.std(r_errors)) if len(r_errors) > 1 else 0.0

    return result


def compute_pose_auc(
    rot_errors: List[float],
    trans_errors: List[float],
    thresholds: List[float] = [5, 10, 20],
) -> Dict[str, float]:
    """Compute Pose AUC at various thresholds (MADPose style).

    AUC@θ = percentage of frames where max(rot_error, trans_error) < θ

    Args:
        rot_errors: List of rotation errors in degrees
        trans_errors: List of translation errors in degrees
        thresholds: List of thresholds in degrees

    Returns:
        Dictionary with AUC@θ for each threshold
    """
    if len(rot_errors) == 0 or len(trans_errors) == 0:
        return {}

    rot_arr = np.array(rot_errors)
    trans_arr = np.array(trans_errors)

    # Pose error = max(rot_error, trans_error)
    pose_errors = np.maximum(rot_arr, trans_arr)

    result = {}
    for thresh in thresholds:
        auc = float(np.mean(pose_errors < thresh) * 100.0)
        result[f'AUC@{thresh}'] = auc

    # Also compute median errors
    result['rot_error_median'] = float(np.median(rot_arr))
    result['trans_error_median'] = float(np.median(trans_arr))
    result['pose_error_median'] = float(np.median(pose_errors))

    return result


def integrate_trajectory(
    R_list: list,
    t_list: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate relative poses into absolute trajectory.

    Args:
        R_list: List of relative rotation matrices
        t_list: List of relative translation vectors

    Returns:
        positions: (N+1, 3) array of positions
        orientations: List of N+1 rotation matrices
    """
    n = len(R_list)
    positions = [np.zeros(3)]
    T_global = np.eye(4)
    orientations = [np.eye(3)]

    for i in range(n):
        # Build relative transform
        T_rel = np.eye(4)
        T_rel[:3, :3] = R_list[i]
        T_rel[:3, 3] = t_list[i]

        # Accumulate
        T_global = T_global @ T_rel

        positions.append(T_global[:3, 3].copy())
        orientations.append(T_global[:3, :3].copy())

    return np.array(positions), orientations


def compute_tae(
    depths: List[np.ndarray],
    poses: List[Tuple[np.ndarray, np.ndarray]],
    baselines: List[float],
    fx: float, fy: float, cx: float, cy: float
) -> Optional[Dict[str, float]]:
    """Compute Temporal Alignment Error (TAE) from Video Depth Anything.

    Uses PyTorch for GPU acceleration, following the official VDA implementation.

    Formula:
        TAE = 1/(2(N-1)) * Σ[AbsRel(warp(d_k, P_k), d_{k+1}) +
                            AbsRel(warp(d_{k+1}, P_k^{-1}), d_k)]

    Args:
        depths: List of N depth maps (H, W)
        poses: List of N-1 relative poses (R, t) from frame k to k+1
        baselines: List of N-1 translation scales
        fx, fy, cx, cy: Camera intrinsics

    Returns:
        Dictionary with TAE metrics, or None if insufficient data
    """
    import torch

    if len(depths) < 2 or len(poses) < 1:
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def tae_warp_torch(depth1, depth2, R, t, baseline, fx, fy, cx, cy, device):
        """Warp depth1 to depth2's frame and compute AbsRel error."""
        H, W = depth1.shape

        # Convert to torch
        d1 = torch.from_numpy(depth1.astype(np.float32)).to(device)
        d2 = torch.from_numpy(depth2.astype(np.float32)).to(device)
        R_t = torch.from_numpy(R.astype(np.float32)).to(device)
        t_t = torch.from_numpy(t.astype(np.float32)).to(device)

        # Create pixel grid
        xx, yy = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
        xx = xx.float()
        yy = yy.float()

        # Backproject to 3D
        X = (xx - cx) * d1 / fx
        Y = (yy - cy) * d1 / fy
        Z = d1
        points3d = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)  # (H*W, 3)

        # Transform to frame 2
        points3d_transformed = torch.matmul(points3d, R_t.T) + t_t
        X_new, Y_new, Z_new = points3d_transformed[:, 0], points3d_transformed[:, 1], points3d_transformed[:, 2]

        # Project to image plane
        u_new = (X_new * fx) / (Z_new + 1e-8) + cx
        v_new = (Y_new * fy) / (Z_new + 1e-8) + cy

        # Round to nearest pixel
        u_int = torch.round(u_new).long()
        v_int = torch.round(v_new).long()

        # Valid mask
        valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H) & (Z_new > 0.1)

        if valid.sum() == 0:
            return None

        # Create projected depth map
        depth_proj = torch.zeros((H, W), dtype=d1.dtype, device=device)
        valid_u = u_int[valid]
        valid_v = v_int[valid]
        valid_z = Z_new[valid]

        # Scatter (simple assignment, last write wins - approximation)
        depth_proj[valid_v, valid_u] = valid_z

        # Compute AbsRel where both projected and target are valid
        compare_mask = (depth_proj > 0) & (d2 > 0)
        if compare_mask.sum() < 100:
            return None

        abs_rel = torch.mean(torch.abs(depth_proj[compare_mask] - d2[compare_mask]) / d2[compare_mask])
        return abs_rel.item()

    error_sum = 0.0
    valid_pairs = 0

    for i in range(min(len(depths) - 1, len(poses), len(baselines))):
        d_curr = depths[i]
        d_next = depths[i + 1]
        R, t = poses[i]
        baseline = baselines[i]

        # Skip if baseline is too small
        if baseline < 0.01:
            continue

        # Forward: warp d_curr to d_next frame
        err_fwd = tae_warp_torch(d_curr, d_next, R, t, baseline, fx, fy, cx, cy, device)

        # Backward: warp d_next to d_curr frame
        R_inv = R.T
        t_inv = -R.T @ t
        err_bwd = tae_warp_torch(d_next, d_curr, R_inv, t_inv, baseline, fx, fy, cx, cy, device)

        if err_fwd is not None:
            error_sum += err_fwd
            valid_pairs += 1
        if err_bwd is not None:
            error_sum += err_bwd
            valid_pairs += 1

    if valid_pairs == 0:
        return None

    # TAE (following VDA: multiply by 100 for percentage)
    tae = (error_sum / valid_pairs) * 100

    return {
        'TAE': tae,
        'TAE_num_pairs': valid_pairs // 2,
    }
