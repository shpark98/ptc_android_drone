#pragma once

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#ifndef __ANDROID__
#include <opencv2/ximgproc/segmentation.hpp>
#endif
#include <Eigen/Dense>
#include <memory>
#include <optional>
#include <vector>

#include "depth_fusion.hpp"  // For BaselineAutoState

namespace pr_depth {

/**
 * Configuration for DepthRefinement pipeline
 */
struct DepthRefinementConfig {
    // Camera intrinsics
    float fx = 721.5f;
    float fy = 721.5f;
    float cx = 609.6f;
    float cy = 172.9f;
    int H = 375;
    int W = 1242;

    // Motion estimation
    int ransac_max_iters = 50;  // Reduced from 100 - same accuracy, faster
    int ransac_min_sample = 6;
    float ransac_thresh_ratio = 1.5f;
    float min_flow_px = 0.01f;
    int max_points = 2000;
    float margin_x_pct = 0.05f;  // Horizontal margin (exclude edge pixels)
    float margin_y_pct = 0.15f;  // Vertical margin (exclude top/bottom)
    int depth_bins = 3;          // Number of depth bins for stratified sampling
    float adaptive_flow_depth_scale = 0.0f;  // Scale for adaptive min_flow threshold (0=disabled)

    // Triangulation
    float max_depth = 80.0f;
    float fill_value = NAN;

    // Baseline guard
    bool use_baseline_guard = false;  // Skip processing when baseline too short
    float min_baseline = 0.05f;
    float baseline_ema_beta = 0.9f;   // EMA smoothing factor (higher = more smoothing)
    int baseline_hist_len = 200;       // History length for percentile calculation

    // Scale matching
    int min_scale_overlap = 2000;
    float scale_tol_median = 0.3f;

    // Sky mask threshold (inverse depth)
    // Pixels with inv_depth < this threshold are marked as sky.
    // Set to 0 to disable sky masking entirely (indoor mode).
    float sky_mask_inv_thresh = 1e-7f;

    // Sky fallback depth (meters) — assigned to sky pixels in segmentation
    float sky_fallback_depth = 100.0f;

    // EdgeAware Segmentation
    bool use_segmentation = true;
    float seg_sigma = 0.5f;
    float seg_k = 500.0f;
    int seg_min_size = 200;
    float seg_down = 0.5f;           // Downsample factor (0.5 = 50% resolution)

    // Edge-aware guide weights
    bool use_rgb_guide = true;
    float wrgb = 0.4f;               // Weight for Lab a,b channels
    float wx = 1.0f;                 // Weight for depth channel
    float wgrad = 1.2f;              // Weight for edge channel
    float grad_power = 1.2f;         // Power for edge magnitude

    // Metric scale estimation mode
    // 0 = off (no scale correction, raw fusion output)
    // 1 = global (single global scale for entire image)
    // 2 = per-segment (per-segment scale estimation, default)
    int metric_scale_mode = 2;

    // Legacy: use_metric_scale maps to metric_scale_mode
    // true -> 2 (per-segment), false -> 0 (off)
    bool use_metric_scale = true;  // Deprecated: use metric_scale_mode instead

    // RANSAC scoring mode
    bool use_magsac_scoring = true;  // Use MAGSAC++ soft scoring (true) or paper MAD-based binary (false)

    // Debug
    bool debug = false;
    bool timing = false;  // Print timing breakdown

    // Iterative refinement (depth → pose → depth cycle)
    // After initial depth refinement, use refined depth to re-estimate pose
    // and triangulate again. This can improve results when initial monocular
    // depth has significant errors.
    bool enable_iterative_refinement = true;   // Enable iterative refinement (default: on)
    int iterative_refinement_iters = 1;        // Number of additional iterations (1 recommended)

    // GT Pose fallback mode
    // When rotation angle exceeds threshold, use GT pose instead of estimated pose.
    // This is useful for handling sharp turns where triangulation quality degrades.
    // Pipeline: GT pose -> dense 3D warp -> scale estimation (skip triangulation & fusion)
    bool use_gt_pose_fallback = false;         // Enable GT pose fallback mode
    float gt_pose_rotation_threshold_deg = 3.0f;  // Rotation threshold in degrees

    // Ablation: Skip temporal fusion
    // When true, skip Bayesian update and only do triangulation + solve_metric_from_rel
    // Each frame is processed independently without temporal propagation
    // Useful for ablation study comparing "triangulation only" vs "full pipeline"
    bool skip_temporal_fusion = false;

    // Ablation: Always use GT rotation
    // When true and GT pose is provided, always use GT rotation for motion estimation
    // (ignores rotation threshold, always uses GT R)
    // Useful for ablation study isolating rotation estimation contribution
    bool use_gt_R = false;

    // Skip forward-backward consistency check
    // When true, skip backward flow computation and FB consistency check
    // All flow pixels are treated as valid (faster but may include occluded regions)
    bool skip_fb_consistency = false;

    // ========== Fusion parameters ==========
    // Variance estimation (rpx_to_variance_angle_aware)
    float fusion_tau0_deg = 5.0f;        // Reference angle in degrees
    float fusion_sigma2_at_tau = 1.0f;   // Variance scale at reference angle
    float fusion_var_floor = 3e-3f;      // Minimum variance (prevents overconfidence)
    float fusion_var_cap = 1e+3f;        // Maximum variance (prevents numerical issues)

    // Bayesian fusion
    float fusion_chi2_soft = 6.635f;     // Soft chi-square threshold (p=0.01, df=1)
    float fusion_chi2_hard = 10.828f;    // Hard chi-square threshold (p=0.001, df=1)
    float fusion_kcap_floor = 0.30f;     // Minimum Kalman gain cap
    float fusion_lambda_forget = 0.35f;  // Forgetting factor for prior prediction

    // Real-time mode presets (call set_realtime_mode() to configure)
    // When true, optimizes for speed over accuracy:
    // - Reduces RANSAC iterations
    // - Skips per-segment scale estimation
    // - Uses segmentation downsampling (0.5 = half resolution)
    void set_realtime_mode(bool enable) {
        if (enable) {
            ransac_max_iters = 50;
            use_metric_scale = false;
            seg_down = 0.5f;  // Half resolution for faster segmentation
            seg_min_size = 100;
        } else {
            ransac_max_iters = 500;
            use_metric_scale = true;
            seg_down = 1.0f;
            seg_min_size = 200;
        }
    }
};

/**
 * Debug info for each iteration (forward/backward)
 */
struct IterationDebugInfo {
    int iter;                       // Iteration number (0=forward, 1=backward)
    bool is_backward;               // True if backward iteration
    Eigen::Matrix3d R;              // Pose R for this iteration
    Eigen::Vector3d t;              // Pose t for this iteration (unit vector)
    int num_inliers;                // Number of inliers from motion estimation
    int num_valid_tri;              // Number of valid triangulated points
    float metric_scale;             // Scale from solve_metric_from_rel (if applied)

    // Per-iteration depth maps (for debugging/analysis)
    cv::Mat z_tri;                  // Triangulated depth for this iteration
    cv::Mat z_fused_sparse;         // Sparse fusion result (before solve_metric_from_rel)
    cv::Mat z_refined;              // Refined depth after fusion for this iteration (after solve_metric)
};

/**
 * Result from depth refinement
 */
struct DepthRefinementResult {
    cv::Mat z_refined;      // Refined depth (H, W) float32
    cv::Mat z_tri;          // Raw triangulated depth (H, W) float32
    cv::Mat confidence;     // Confidence map (H, W) float32
    cv::Mat seg_labels;     // Segmentation labels (H, W) int32
    Eigen::Matrix3d R;      // Estimated rotation matrix (frame t-1 to t)
    Eigen::Vector3d t;      // Estimated translation direction (unit vector)
    float baseline;         // Baseline between frames
    int num_matches;        // Number of matches used
    int num_valid_tri;      // Number of valid triangulated points
    int num_segments;       // Number of segments
    bool tri_disabled;      // Whether triangulation was disabled
    float baseline_correction;      // Baseline correction factor applied (1.0 = no correction)

    // === Debug info (only populated when config.debug = true) ===
    cv::Mat z_tri_forward;          // Triangulated depth from forward pass
    cv::Mat z_tri_backward;         // Triangulated depth from backward pass (if any)
    cv::Mat z_warp_flow;            // Optical flow warped prev_depth
    cv::Mat z_warp_pose;            // Pose-based 3D warped prev_depth
    cv::Mat prev_depth_used;        // prev_depth_ that was used for this frame
    cv::Mat V_prior;                // Variance of prior (before fusion)
    cv::Mat V_post;                 // Variance after fusion
    float metric_scale_forward;     // Metric scale from forward
    float metric_scale_backward;    // Metric scale from backward
    Eigen::Matrix3d R_forward;      // R from forward motion estimation
    Eigen::Vector3d t_forward;      // t from forward motion estimation
    Eigen::Matrix3d R_backward;     // R from backward motion estimation
    Eigen::Vector3d t_backward;     // t from backward motion estimation
    std::vector<IterationDebugInfo> iteration_info;  // Per-iteration debug info
    bool used_backward;             // Whether final result used backward pose

    // GT pose fallback info
    bool used_gt_pose;              // Whether GT pose was used (rotation > threshold)
    float rotation_angle_deg;       // Estimated rotation angle in degrees
    cv::Mat z_warp_gt;              // Dense 3D warped depth using GT pose (when GT pose used)

    // Optical flow (always populated for visualization)
    cv::Mat flow;                   // Forward optical flow (H, W) CV_32FC2
};

/**
 * Unified Depth Refinement Pipeline
 *
 * This class encapsulates the entire depth refinement pipeline:
 * 1. Optical flow computation (DIS)
 * 2. Motion estimation (RANSAC-based)
 * 3. Triangulation
 * 4. Scale matching
 * 5. Temporal fusion (optional)
 *
 * Usage:
 *   DepthRefinement pipeline(config);
 *   for each frame:
 *       result = pipeline.refine(img, inv_depth, pose);
 */
class DepthRefinement {
public:
    /**
     * Constructor
     * @param config Pipeline configuration
     */
    explicit DepthRefinement(const DepthRefinementConfig& config);

    /**
     * Reset pipeline state (for new sequence)
     */
    void reset();

    /**
     * Main refinement function
     *
     * @param img Current frame (H, W, 3) BGR uint8
     * @param inv_depth Inverse depth from monocular model (H, W) float32
     *                  Values in [0, 1], 0=far, 1=close
     * @param baseline Translation magnitude between frames (meters)
     * @param seg_labels Optional segmentation labels (H, W) int32
     * @return DepthRefinementResult with refined depth and diagnostics
     */
    DepthRefinementResult refine(
        const cv::Mat& img,
        const cv::Mat& inv_depth,
        float baseline,
        const cv::Mat& seg_labels = cv::Mat()
    );

    /**
     * Main refinement function with optional GT pose
     *
     * @param img Current frame (H, W, 3) BGR uint8
     * @param inv_depth Inverse depth from monocular model (H, W) float32
     * @param baseline Translation magnitude between frames (meters)
     * @param gt_R Optional GT rotation matrix (3x3) - used when rotation > threshold
     * @param gt_t Optional GT translation vector (3x1, metric) - used when rotation > threshold
     * @param seg_labels Optional segmentation labels (H, W) int32
     * @return DepthRefinementResult with refined depth and diagnostics
     */
    DepthRefinementResult refine(
        const cv::Mat& img,
        const cv::Mat& inv_depth,
        float baseline,
        const std::optional<Eigen::Matrix3d>& gt_R,
        const std::optional<Eigen::Vector3d>& gt_t,
        const cv::Mat& seg_labels = cv::Mat()
    );

    /**
     * Prepare optical flow in advance (Phase 1 of pipelined execution).
     * Call this while QNN inference is running on NPU.
     * The computed flow will be used by the next refine() call.
     * @param img Current frame (BGR)
     */
    void prepare_flow(const cv::Mat& img);

    /**
     * Check if precomputed flow is available
     */
    bool has_precomputed_flow() const { return flow_ready_; }

    /**
     * Get current frame count
     */
    int frame_count() const { return frame_count_; }

    /**
     * Update pipeline configuration at runtime
     * @param config New configuration
     */
    void updateConfig(const DepthRefinementConfig& config);

    /**
     * Get current configuration
     * @return Current configuration
     */
    DepthRefinementConfig getConfig() const;

private:
    DepthRefinementConfig config_;
    int frame_count_;

    // Previous frame state
    cv::Mat prev_img_;
    cv::Mat prev_inv_depth_;
    cv::Mat prev_depth_;
    cv::Mat prev_V_;  // Variance
    float kappa_state_;  // EMA state for consistency kappa
    float scale_state_;  // EMA state for global scale tracking (for scale jump detection)
    bool scale_state_initialized_;  // Whether scale_state_ has been initialized

    // Baseline tracking
    BaselineAutoState baseline_state_;  // EMA for baseline adaptive threshold

    // DIS optical flow (two instances for parallel forward/backward computation)
    cv::Ptr<cv::DISOpticalFlow> dis_flow_;
    cv::Ptr<cv::DISOpticalFlow> dis_flow_backward_;  // For backward flow in iterative refinement

    // Precomputed flow for pipelined execution (NPU/CPU overlap)
    cv::Mat precomputed_flow_;
    cv::Mat precomputed_img_;  // Current frame stored during prepare_flow()
    bool flow_ready_ = false;

    // Felzenszwalb segmentation
#ifndef __ANDROID__
    cv::Ptr<cv::ximgproc::segmentation::GraphSegmentation> graph_seg_;
#endif

    // Helper methods

    /**
     * Compute optical flow between two frames
     */
    cv::Mat compute_flow(const cv::Mat& img_prev, const cv::Mat& img_curr);

    /**
     * Compute edge map from depth using Sobel
     * @param depth Depth map (H, W) float32
     * @return Edge magnitude map (H, W) float32
     */
    cv::Mat compute_edge(const cv::Mat& depth);

    /**
     * Build edge-aware guide image for segmentation
     * @param img Input image (H, W, 3) BGR uint8
     * @param depth Depth map (H, W) float32
     * @param edge_map Edge magnitude map (H, W) float32
     * @param sky_mask Sky mask (H, W) uint8, 1=sky
     * @return Multi-channel guide image for segmentation
     */
    cv::Mat build_guide(const cv::Mat& img, const cv::Mat& depth,
                        const cv::Mat& edge_map, const cv::Mat& sky_mask);

    /**
     * Compute EdgeAware segmentation (matches Python EdgeAwareSegmentation)
     * @param img Input image (H, W, 3) BGR uint8
     * @param depth Depth map (H, W) float32
     * @param sky_mask Sky mask (H, W) uint8, 1=sky
     * @return Segmentation labels (H, W) int32, and number of segments
     */
    std::pair<cv::Mat, int> compute_segmentation(const cv::Mat& img,
                                                  const cv::Mat& depth,
                                                  const cv::Mat& sky_mask);

    /**
     * Estimate motion from flow and depth
     * Returns rotation R, translation t (unit), and matched points
     * @param known_omega Optional known rotation vector (camera frame). If provided,
     *                    only translation will be estimated (for use_gt_R ablation).
     */
    struct MotionResult {
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t = Eigen::Vector3d(0, 0, -1);  // Default: forward (-Z)
        Eigen::Vector3d omega = Eigen::Vector3d::Zero();  // Rotation vector (axis-angle)
        std::vector<float> u0, v0, u1, v1;
        int num_inliers = 0;
        bool success = false;
    };
    MotionResult estimate_motion(
        const cv::Mat& flow,
        const cv::Mat& inv_depth,
        const cv::Mat& mask,
        const std::optional<Eigen::Vector3d>& known_omega = std::nullopt
    );

    /**
     * Triangulate depth from correspondences
     */
    struct TriResult {
        cv::Mat z1_tri;
        cv::Mat rpx_tri;
        int num_valid;
    };
    TriResult triangulate(
        const std::vector<float>& u0,
        const std::vector<float>& v0,
        const std::vector<float>& u1,
        const std::vector<float>& v1,
        const Eigen::Matrix3d& R,
        const Eigen::Vector3d& t
    );

    /**
     * Robust scale matching between triangulated and warped depth
     */
    struct ScaleResult {
        float scale;
        bool scale_ok;
        int overlap;
        float median_relerr;
    };
    ScaleResult robust_scale_match(
        const cv::Mat& z_tri,
        const cv::Mat& z_ref,
        const cv::Mat& mask
    );

    /**
     * Update internal state after processing
     */
    void update_state(
        const cv::Mat& img,
        const cv::Mat& inv_depth,
        const cv::Mat& depth,
        const cv::Mat& V
    );
};

}  // namespace pr_depth
