/**
 * @file motion_field.hpp
 * @brief Motion field estimation from optical flow (Eq. 1-2)
 *
 * Paper: PR-Depth, Eq. (1): ṗ = B(x,y)ω + (1/αz)A(x,y)t
 *        Eq. (2): Linear system for joint rotation/translation estimation
 *
 * This module estimates camera rotation (ω) and translation (t) from optical flow
 * using the motion field equation with inverse depth.
 */
#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <optional>
#include "pr_depth/types.hpp"

namespace pr_depth {

/**
 * Configuration for motion field estimation
 */
struct MotionFieldConfig {
    // RANSAC parameters
    int ransac_max_iters = 500;          // Maximum RANSAC iterations
    int ransac_min_sample = 6;           // Minimum sample size for RANSAC
    float ransac_thresh_ratio = 1.5f;    // Inlier threshold (ratio of flow magnitude)
    float min_flow_px = 0.01f;            // Minimum flow magnitude to consider (px)

    // IRLS parameters
    int lo_irls_iters = 5;               // Local optimization IRLS iterations
    float huber_delta_rel = 3.5f;        // Huber loss delta (dimensionless)

    // Filtering
    float mad_scale = 5.5f;              // MAD scale for adaptive thresholding

    // Point sampling
    int max_points = 2000;               // Maximum points for RANSAC
    int depth_bins = 3;                  // Number of depth bins for stratified sampling
    float margin_x_pct = 0.0f;          // Horizontal margin (fraction)
    float margin_y_pct = 0.0f;          // Vertical margin (fraction)
    float adaptive_flow_depth_scale = 0.0f;  // Scale for adaptive min_flow (0=disabled)
    bool use_ransac = true;              // Use RANSAC (false = simple least squares)

    // Direction-based inlier criterion
    // cos_sim = obs.dot(pred) / (|obs| * |pred|)
    // If cos_sim < cos_sim_thresh, point is outlier regardless of magnitude ratio
    float cos_sim_thresh = 0.5f;         // Cosine similarity threshold (0.5 = 60°)
    bool use_direction_check = true;     // Enable direction check in RANSAC

    // Depth scale mode: how to handle depth scale/shift ambiguity
    // 0 = none (use depth as-is)
    // 1 = scale only (r' = s * r, estimate s)
    // 2 = affine (r' = s * r + o, estimate s and o)
    int depth_scale_mode = 0;

    // Scoring method toggle
    // true  = MAGSAC++ soft weighting (weight = max(0, 1 - rel_res/rel_sigma_max))
    // false = MAD-based binary threshold (paper method: median + k*MAD)
    bool use_magsac_scoring = true;

    // MAGSAC++ parameters (only used when use_magsac_scoring = true)
    float magsac_rel_sigma_max = 0.25;   // Maximum relative error threshold. weight = max(0, 1 - rel_res/rel_sigma_max)
    float magsac_inlier_weight = 0.5f;   // Weight threshold for inlier count (0.5 -> rel_res < 0.5*rel_sigma_max = 10%)

    // Random seed (0 = use random_device)
    unsigned int seed = 42;               // Random seed for reproducibility
};

/**
 * Result of motion field estimation
 */
struct MotionFieldResult {
    Eigen::Matrix3d R;                   // 3x3 rotation matrix
    Eigen::Vector3d t;                   // 3x1 translation vector (direction only, scale arbitrary)
    Eigen::Vector3d omega;               // 3x1 rotation vector (axis-angle)

    cv::Mat flow_refined;                // (H,W,2) refined flow (outliers replaced)
    cv::Mat inlier_mask;                 // (H,W) bool mask of inliers

    // Inlier matches for triangulation
    std::vector<float> u0;               // Source x coordinates (inliers only)
    std::vector<float> v0;               // Source y coordinates (inliers only)
    std::vector<float> u1;               // Target x coordinates (inliers only)
    std::vector<float> v1;               // Target y coordinates (inliers only)

    int num_inliers = 0;                 // Number of inlier points
    int num_points_used = 0;             // Total points used in estimation
    float mean_residual = 0.0f;          // Mean residual of inliers

    // Depth scale/shift estimation (if depth_scale_mode > 0)
    double depth_scale = 1.0;            // Estimated depth scale (s in r' = s*r + o)
    double depth_offset = 0.0;           // Estimated depth offset (o in r' = s*r + o)
};

/**
 * Compute B and A matrices for motion field equation
 *
 * Given normalized image coordinates (x, y):
 *   B = [[x*y,      -(1+x²), y ],
 *        [(1+y²),   -x*y,    -x]]
 *
 *   A = [[-1, 0, x],
 *        [ 0,-1, y]]
 *
 * @param x Normalized x coordinate: (u - cx) / fx
 * @param y Normalized y coordinate: (v - cy) / fy
 * @param B Output B matrix (2x3)
 * @param A Output A matrix (2x3)
 */
inline void compute_motion_matrices(double x, double y,
                                    Eigen::Matrix<double, 2, 3>& B,
                                    Eigen::Matrix<double, 2, 3>& A) {
    // B matrix (rotation term)
    B(0, 0) = x * y;
    B(0, 1) = -(1.0 + x * x);
    B(0, 2) = y;

    B(1, 0) = (1.0 + y * y);
    B(1, 1) = -(x * y);
    B(1, 2) = -x;

    // A matrix (translation term)
    A(0, 0) = -1.0;
    A(0, 1) =  0.0;
    A(0, 2) =  x;

    A(1, 0) =  0.0;
    A(1, 1) = -1.0;
    A(1, 2) =  y;
}

/**
 * Motion field estimator
 */
class MotionFieldEstimator {
public:
    explicit MotionFieldEstimator(const MotionFieldConfig& cfg = MotionFieldConfig());

    /**
     * Estimate camera motion from optical flow
     *
     * @param flow Optical flow (H x W x 2) in pixels [u, v]
     * @param inv_depth Inverse depth map (H x W) from prior depth estimate
     * @param intrinsics Camera intrinsics
     * @param mask Optional mask (H x W) of valid pixels (1=valid, 0=skip)
     * @param known_omega Optional known rotation vector (axis-angle, camera frame).
     *                    If provided, only translation will be estimated by subtracting
     *                    the rotation component: flow_trans = flow - B @ omega.
     *                    Useful for ablation studies with GT rotation.
     * @return MotionFieldResult containing R, t, and refined flow
     */
    MotionFieldResult estimate(const cv::Mat& flow,
                               const cv::Mat& inv_depth,
                               const CameraIntrinsics& intrinsics,
                               const cv::Mat& mask = cv::Mat(),
                               const std::optional<Eigen::Vector3d>& known_omega = std::nullopt);

private:
    MotionFieldConfig cfg_;

    // Least-squares solve
    Eigen::VectorXd solve_least_squares(
        const std::vector<Eigen::Vector2d>& flow_normalized,
        const std::vector<Eigen::Vector2d>& coords_normalized,
        const std::vector<double>& inv_depths);

    // IRLS refinement with Huber weights
    Eigen::VectorXd refine_irls(
        const std::vector<Eigen::Vector2d>& flow_normalized,
        const std::vector<Eigen::Vector2d>& coords_normalized,
        const std::vector<double>& inv_depths,
        const Eigen::VectorXd& theta_init,
        const CameraIntrinsics& intrinsics);

    // Full RANSAC estimation with cell-based sampling and adaptive threshold
    Eigen::VectorXd ransac_estimate(
        const std::vector<Eigen::Vector2d>& flow_normalized,
        const std::vector<Eigen::Vector2d>& coords_normalized,
        const std::vector<double>& inv_depths,
        const std::vector<double>& flow_magnitudes,
        const std::vector<int>& cell_ids,
        const CameraIntrinsics& intrinsics,
        int max_cells);
};

} // namespace pr_depth
