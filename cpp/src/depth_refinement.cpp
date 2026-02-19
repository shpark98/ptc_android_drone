/**
 * @file depth_refinement.cpp
 * @brief Unified depth refinement pipeline implementation
 */
#include "pr_depth/depth_refinement.hpp"
#include "pr_depth/optical_flow.hpp"
#include "pr_depth/motion_field.hpp"
#include "pr_depth/triangulation.hpp"
#include "pr_depth/depth_fusion.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <chrono>
#include <omp.h>

#ifdef __ANDROID__
#include <android/log.h>
#define PR_TIMING_LOG(...) __android_log_print(ANDROID_LOG_INFO, "PR-Depth-Timing", __VA_ARGS__)
#else
#define PR_TIMING_LOG(...) do { fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#endif

namespace pr_depth {

// Compute confidence from variance: conf = 1 / (1 + V)
static inline cv::Mat variance_to_confidence(const cv::Mat& V) {
    const int H = V.rows, W = V.cols;
    cv::Mat conf(H, W, CV_32F);
    for (int r = 0; r < H; ++r) {
        const float* v_row = V.ptr<float>(r);
        float* c_row = conf.ptr<float>(r);
        for (int c = 0; c < W; ++c) {
            c_row[c] = 1.0f / (1.0f + v_row[c]);
        }
    }
    return conf;
}

/**
 * Convert rotation matrix to axis-angle representation (omega)
 * Uses Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
 * where K is the skew-symmetric matrix of the axis
 *
 * @param R 3x3 rotation matrix
 * @return omega 3x1 axis-angle vector (axis * angle)
 */
static Eigen::Vector3d rotation_matrix_to_omega(const Eigen::Matrix3d& R) {
    // Compute rotation angle: θ = acos((trace(R) - 1) / 2)
    double trace = R.trace();
    double cos_angle = (trace - 1.0) / 2.0;
    cos_angle = std::max(-1.0, std::min(1.0, cos_angle));  // Clamp for numerical stability
    double angle = std::acos(cos_angle);

    // Handle special cases
    if (angle < 1e-10) {
        // Near identity: R ≈ I + θK, so K ≈ (R - I) / θ ≈ R - I for small θ
        // omega ≈ [R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1)] / 2
        return Eigen::Vector3d(
            (R(2,1) - R(1,2)) / 2.0,
            (R(0,2) - R(2,0)) / 2.0,
            (R(1,0) - R(0,1)) / 2.0
        );
    }

    if (angle > M_PI - 1e-10) {
        // Near 180 degrees: use eigenvector of R corresponding to eigenvalue 1
        // (R + I) v = 2v for the rotation axis v
        Eigen::Matrix3d B = R + Eigen::Matrix3d::Identity();
        // Find the column with largest norm
        int max_col = 0;
        double max_norm = B.col(0).norm();
        for (int i = 1; i < 3; ++i) {
            double norm = B.col(i).norm();
            if (norm > max_norm) {
                max_norm = norm;
                max_col = i;
            }
        }
        Eigen::Vector3d axis = B.col(max_col).normalized();
        return axis * angle;
    }

    // General case: extract axis from skew-symmetric part
    double sin_angle = std::sin(angle);
    Eigen::Vector3d axis(
        (R(2,1) - R(1,2)) / (2.0 * sin_angle),
        (R(0,2) - R(2,0)) / (2.0 * sin_angle),
        (R(1,0) - R(0,1)) / (2.0 * sin_angle)
    );

    return axis * angle;
}

// Fusion config parameters (matching Python defaults)
struct FusionConfig {
    float chi2_soft = 6.635f;        // 90% chi2(1) quantile
    float chi2_hard = 10.828f;       // 99% chi2(1) quantile
    float kcap_floor = 0.35f;        // Minimum K_cap - ensures triangulation always has influence for scale recovery
    float gate_loosen = 0.9f;        // Gate loosening for low consistency
    float min_var = 5e-3f;           // Minimum variance
    float lambda_forget = 0.4f;      // Process noise factor
    float diff_mad_mul = 3.0f;       // MAD multiplier for kappa
    float kappa_min = 0.01f;         // Min kappa (1%)
    float kappa_max = 0.05f;         // Max kappa (5%)
    float kappa_ema_beta = 0.8f;     // EMA coefficient for kappa
    // Frame reject parameters
    bool frame_reject_enable = true;
    int frame_reject_min_valid = 1000;  // Minimum valid pixels to compute bad_frac
    float frame_reject_bad_frac = 0.5f; // Reject if >50% pixels fail chi2_hard
    // Scale jump detection parameters
    bool scale_jump_reject_enable = false; // Enable scale jump detection (disabled by default)
    float scale_jump_threshold = 0.3f;     // Reject if |scale_ratio - 1| > threshold (30% change)
    float scale_ema_beta = 0.9f;           // EMA coefficient for scale tracking
};

/**
 * Combined dense warp: warp both depth and variance in a single pass
 * More cache-efficient than separate functions
 */
static void warp_dense_combined(
    const cv::Mat& prev_depth,
    const cv::Mat& prev_V,
    const cv::Mat& flow,
    int H, int W,
    float lambda_forget,
    float min_var,
    cv::Mat& z_warp_out,
    cv::Mat& V_warp_out
) {
    z_warp_out = cv::Mat(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    V_warp_out = cv::Mat(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));

    if (prev_depth.empty() || prev_V.empty() || flow.empty()) {
        return;
    }

    // For Z-buffer: track minimum depth at each pixel
    cv::Mat min_depth(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::infinity()));

    for (int v0 = 0; v0 < H; ++v0) {
        const float* flow_row = flow.ptr<float>(v0);
        const float* depth_row = prev_depth.ptr<float>(v0);
        const float* V_row = prev_V.ptr<float>(v0);

        for (int u0 = 0; u0 < W; ++u0) {
            float z0 = depth_row[u0];
            float var = V_row[u0];

            // Get flow displacement
            float fx = flow_row[u0 * 2 + 0];
            float fy = flow_row[u0 * 2 + 1];

            // Compute target coordinates
            int u1 = static_cast<int>(std::round(u0 + fx));
            int v1_coord = static_cast<int>(std::round(v0 + fy));

            // Check target bounds
            if (u1 < 0 || u1 >= W || v1_coord < 0 || v1_coord >= H) continue;

            // Use ptr for faster access (no bounds checking)
            float* z_warp_target = z_warp_out.ptr<float>(v1_coord);
            float* V_warp_target = V_warp_out.ptr<float>(v1_coord);
            float* min_depth_target = min_depth.ptr<float>(v1_coord);

            // Depth + variance warp with Z-buffer (closest pixel wins both)
            if (std::isfinite(z0) && z0 > 1e-8f && z0 < min_depth_target[u1]) {
                min_depth_target[u1] = z0;
                z_warp_target[u1] = z0;
                if (std::isfinite(var)) {
                    V_warp_target[u1] = var * (1.0f + lambda_forget) + min_var;
                }
            }
        }
    }
}

/**
 * Warp prev_depth to current frame using 3D transformation
 *
 * This does proper 3D warping:
 * 1. Create 3D points from prev_depth using ray directions
 * 2. Transform: P1 = R @ P0 + t to cam1 frame
 *    (where R, t follow motion field convention: p_curr = R @ p_prev + t)
 * 3. Extract Z coordinate as warped depth
 * 4. Splat to target pixels using Z-buffer
 *
 * IMPORTANT: This function expects R_tri, t_tri in triangulation convention:
 *   - R_tri = R_motion^T (transpose of motion R)
 *   - t_tri = -R_motion^T @ t_motion (camera center in prev frame)
 * It internally converts back to motion convention for proper warping.
 *
 * @param prev_depth Previous frame depth map (HxW, float32)
 * @param u0, v0 Source pixel coordinates (sparse matches) in prev frame
 * @param u1, v1 Target pixel coordinates (sparse matches) in curr frame
 * @param R_tri Rotation matrix in triangulation convention (R_motion^T)
 * @param t_tri Translation in triangulation convention (-R_motion^T @ t_motion)
 * @param fx, fy, cx, cy Camera intrinsics
 * @param H, W Image dimensions
 * @return Warped depth map (HxW, float32) with NaN for invalid pixels
 */
/**
 * Dense 3D warp: transform ALL pixels of prev_depth to curr frame using R,t
 *
 * For each pixel (u,v) in prev_depth:
 *   1. Back-project to 3D: P0 = z * [(u-cx)/fx, (v-cy)/fy, 1]
 *   2. Transform: P1 = R_motion @ P0 + t_motion
 *   3. Project: u1 = fx*P1.x/P1.z + cx, v1 = fy*P1.y/P1.z + cy
 *   4. Z-buffer at target pixel
 *
 * Also warps variance with inflation: V_warp = V * (1 + lambda_forget) + min_var
 *
 * Convention: R_tri = R_motion^T, t_tri = -R_motion^T @ t_motion
 */
static void warp_depth_3d_dense(
    const cv::Mat& prev_depth,
    const cv::Mat& prev_V,
    const Eigen::Matrix3d& R_tri, const Eigen::Vector3d& t_tri,
    float fx, float fy, float cx, float cy,
    int H, int W,
    float lambda_forget, float min_var,
    cv::Mat& z_warp_out,
    cv::Mat& V_warp_out
) {
    z_warp_out = cv::Mat(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    V_warp_out = cv::Mat(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));

    if (prev_depth.empty()) {
        return;
    }

    // Convert from triangulation convention back to motion convention:
    // R_tri = R_motion^T  =>  R_motion = R_tri^T
    // t_tri = -R_motion^T @ t_motion  =>  t_motion = -R_motion @ t_tri = -R_tri^T @ t_tri
    Eigen::Matrix3d R_motion = R_tri.transpose();
    Eigen::Vector3d t_motion = -R_motion * t_tri;

    // Pre-extract R rows for faster inner loop
    const double r00 = R_motion(0,0), r01 = R_motion(0,1), r02 = R_motion(0,2);
    const double r10 = R_motion(1,0), r11 = R_motion(1,1), r12 = R_motion(1,2);
    const double r20 = R_motion(2,0), r21 = R_motion(2,1), r22 = R_motion(2,2);
    const double tx = t_motion(0), ty = t_motion(1), tz = t_motion(2);

    // Z-buffer: track minimum depth at each pixel
    cv::Mat min_depth(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::infinity()));

    bool has_V = !prev_V.empty();

    for (int v0 = 0; v0 < H; ++v0) {
        const float* depth_row = prev_depth.ptr<float>(v0);
        const float* V_row = has_V ? prev_V.ptr<float>(v0) : nullptr;

        for (int u0 = 0; u0 < W; ++u0) {
            float z0 = depth_row[u0];
            if (!std::isfinite(z0) || z0 <= 1e-8f) continue;

            // Back-project to 3D in cam0 (prev frame)
            double X0 = (u0 - cx) / fx * z0;
            double Y0 = (v0 - cy) / fy * z0;
            double Z0 = z0;

            // Transform to cam1 (curr frame): P1 = R_motion @ P0 + t_motion
            double X1 = r00 * X0 + r01 * Y0 + r02 * Z0 + tx;
            double Y1 = r10 * X0 + r11 * Y0 + r12 * Z0 + ty;
            double Z1 = r20 * X0 + r21 * Y0 + r22 * Z0 + tz;

            if (Z1 <= 1e-8) continue;

            // Project to curr frame
            double u1d = fx * X1 / Z1 + cx;
            double v1d = fy * Y1 / Z1 + cy;

            int u1 = static_cast<int>(std::round(u1d));
            int v1 = static_cast<int>(std::round(v1d));

            // Check target bounds
            if (u1 < 0 || u1 >= W || v1 < 0 || v1 >= H) continue;

            // Z-buffer: keep closest point
            float z1_f = static_cast<float>(Z1);
            float* min_ptr = min_depth.ptr<float>(v1);
            if (z1_f < min_ptr[u1]) {
                min_ptr[u1] = z1_f;
                z_warp_out.ptr<float>(v1)[u1] = z1_f;

                if (has_V) {
                    float var = V_row[u0];
                    if (std::isfinite(var)) {
                        V_warp_out.ptr<float>(v1)[u1] = var * (1.0f + lambda_forget) + min_var;
                    }
                }
            }
        }
    }
}

/**
 * Compute rotation angle from rotation matrix
 *
 * @param R 3x3 rotation matrix
 * @return Rotation angle in degrees
 */
static float compute_rotation_angle_deg(const Eigen::Matrix3d& R) {
    // Rotation angle from trace: angle = acos((trace(R) - 1) / 2)
    double trace = R.trace();
    // Clamp to valid range for acos
    double cos_angle = std::clamp((trace - 1.0) / 2.0, -1.0, 1.0);
    double angle_rad = std::acos(cos_angle);
    return static_cast<float>(angle_rad * 180.0 / M_PI);
}

/**
 * Dense 3D warp: warp entire prev_depth to current frame using pose
 *
 * Unlike warp_depth_3d which uses sparse matches, this function warps all pixels.
 * Used for GT pose fallback mode where triangulation is skipped.
 *
 * Motion convention: p_curr = R @ p_prev + t
 *
 * @param prev_depth Previous frame depth map (HxW, float32)
 * @param R_motion Rotation matrix (motion convention: prev->curr)
 * @param t_motion Translation vector (motion convention, metric)
 * @param fx, fy, cx, cy Camera intrinsics
 * @param H, W Image dimensions
 * @param min_depth Minimum valid depth
 * @param max_depth Maximum valid depth
 * @return Warped depth map in current frame (HxW, float32)
 */
static cv::Mat warp_depth_3d_dense(
    const cv::Mat& prev_depth,
    const Eigen::Matrix3d& R_motion,
    const Eigen::Vector3d& t_motion,
    float fx, float fy, float cx, float cy,
    int H, int W,
    float min_depth = 0.5f,
    float max_depth = 200.0f
) {
    cv::Mat z_warp(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));

    if (prev_depth.empty()) {
        return z_warp;
    }

    // Z-buffer for occlusion handling
    cv::Mat min_z(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::infinity()));

    // Pre-extract R rows for faster inner loop
    const double r00 = R_motion(0,0), r01 = R_motion(0,1), r02 = R_motion(0,2);
    const double r10 = R_motion(1,0), r11 = R_motion(1,1), r12 = R_motion(1,2);
    const double r20 = R_motion(2,0), r21 = R_motion(2,1), r22 = R_motion(2,2);
    const double tx = t_motion(0), ty = t_motion(1), tz = t_motion(2);

    for (int v0 = 0; v0 < H; ++v0) {
        const float* z_row = prev_depth.ptr<float>(v0);

        for (int u0 = 0; u0 < W; ++u0) {
            float z0 = z_row[u0];
            if (!std::isfinite(z0) || z0 < min_depth || z0 > max_depth) continue;

            // Back-project to 3D in prev frame
            double X0 = (u0 - cx) / fx * z0;
            double Y0 = (v0 - cy) / fy * z0;
            double Z0 = z0;

            // Transform to current frame: P1 = R @ P0 + t
            double X1 = r00 * X0 + r01 * Y0 + r02 * Z0 + tx;
            double Y1 = r10 * X0 + r11 * Y0 + r12 * Z0 + ty;
            double Z1 = r20 * X0 + r21 * Y0 + r22 * Z0 + tz;

            if (!std::isfinite(Z1) || Z1 < min_depth || Z1 > max_depth) continue;

            // Project to current frame pixel coordinates
            double u1_d = fx * X1 / Z1 + cx;
            double v1_d = fy * Y1 / Z1 + cy;

            int u1 = static_cast<int>(std::round(u1_d));
            int v1 = static_cast<int>(std::round(v1_d));

            if (u1 < 0 || u1 >= W || v1 < 0 || v1 >= H) continue;

            // Z-buffer: keep closest point
            float z1_f = static_cast<float>(Z1);
            float* min_ptr = min_z.ptr<float>(v1);
            if (z1_f < min_ptr[u1]) {
                min_ptr[u1] = z1_f;
                z_warp.ptr<float>(v1)[u1] = z1_f;
            }
        }
    }

    return z_warp;
}



// DepthRefinement Implementation
DepthRefinement::DepthRefinement(const DepthRefinementConfig& config)
    : config_(config), frame_count_(0),
      kappa_state_(0.1f), scale_state_(1.0f), scale_state_initialized_(false),
      baseline_state_(config.baseline_ema_beta, config.baseline_hist_len) {
    // Initialize DIS optical flow (two instances for parallel forward/backward)
    OpticalFlowConfig flow_cfg;
    flow_cfg.preset = cv::DISOpticalFlow::PRESET_MEDIUM;  // Default - good accuracy
    flow_cfg.finest_scale = 0;
    flow_cfg.use_spatial_propagation = true;

    dis_flow_ = cv::DISOpticalFlow::create(flow_cfg.preset);
    dis_flow_->setFinestScale(flow_cfg.finest_scale);
    dis_flow_->setUseSpatialPropagation(flow_cfg.use_spatial_propagation);

    // Second DIS instance for backward flow (thread-safe parallel computation)
    dis_flow_backward_ = cv::DISOpticalFlow::create(flow_cfg.preset);
    dis_flow_backward_->setFinestScale(flow_cfg.finest_scale);
    dis_flow_backward_->setUseSpatialPropagation(flow_cfg.use_spatial_propagation);

    // Initialize Felzenszwalb segmentation
#ifndef __ANDROID__
    if (config_.use_segmentation) {
        graph_seg_ = cv::ximgproc::segmentation::createGraphSegmentation(
            config_.seg_sigma,
            config_.seg_k,
            config_.seg_min_size
        );
    }
#else
    // Android: Force segmentation off (ximgproc not available)
    config_.use_segmentation = false;
#endif
}

void DepthRefinement::reset() {
    frame_count_ = 0;
    prev_img_.release();
    prev_inv_depth_.release();
    prev_depth_.release();
    prev_V_.release();
    kappa_state_ = 0.1f;  // Initial kappa (10% relative error)
    scale_state_ = 1.0f;  // Initial global scale
    scale_state_initialized_ = false;  // Will be set on first valid frame
    // Reset baseline EMA state
    baseline_state_ = BaselineAutoState(config_.baseline_ema_beta, config_.baseline_hist_len);
}

void DepthRefinement::updateConfig(const DepthRefinementConfig& config) {
    config_ = config;
    // Reinitialize DIS flow if parameters changed
    // (For now, we keep existing instances - can be enhanced if needed)
}

DepthRefinementConfig DepthRefinement::getConfig() const {
    return config_;
}

cv::Mat DepthRefinement::compute_flow(const cv::Mat& img_prev, const cv::Mat& img_curr) {
    // Convert to grayscale if needed
    cv::Mat gray_prev, gray_curr;

    if (img_prev.channels() == 3) {
        cv::cvtColor(img_prev, gray_prev, cv::COLOR_BGR2GRAY);
    } else {
        gray_prev = img_prev;
    }

    if (img_curr.channels() == 3) {
        cv::cvtColor(img_curr, gray_curr, cv::COLOR_BGR2GRAY);
    } else {
        gray_curr = img_curr;
    }

    cv::Mat flow;
    dis_flow_->calc(gray_prev, gray_curr, flow);
    return flow;
}

cv::Mat DepthRefinement::compute_edge(const cv::Mat& depth) {
    cv::Mat depth_f;
    if (depth.type() != CV_32F) {
        depth.convertTo(depth_f, CV_32F);
    } else {
        depth_f = depth.clone();
    }

    // Replace NaN/Inf with 0
    for (int r = 0; r < depth_f.rows; ++r) {
        float* row = depth_f.ptr<float>(r);
        for (int c = 0; c < depth_f.cols; ++c) {
            if (!std::isfinite(row[c])) {
                row[c] = 0.0f;
            }
        }
    }

    // Sobel edge detection
    cv::Mat gx, gy;
    cv::Sobel(depth_f, gx, CV_32F, 1, 0, 3);
    cv::Sobel(depth_f, gy, CV_32F, 0, 1, 3);

    // Edge magnitude
    cv::Mat edge;
    cv::magnitude(gx, gy, edge);

    return edge;
}

cv::Mat DepthRefinement::build_guide(const cv::Mat& img, const cv::Mat& depth,
                                      const cv::Mat& edge_map, const cv::Mat& sky_mask) {
    const int H = img.rows;
    const int W = img.cols;
    int num_channels = 1 + 1;  // depth + edge
    if (config_.use_rgb_guide) {
        num_channels += 1;  // Lab chroma (combined a,b → single channel)
    }

    cv::Mat guide(H, W, CV_32FC(num_channels));

    // Process each pixel
    cv::Mat lab;
    if (config_.use_rgb_guide && img.channels() == 3) {
        cv::Mat img_8u;
        if (img.type() != CV_8UC3) {
            img.convertTo(img_8u, CV_8UC3);
        } else {
            img_8u = img;
        }
        cv::cvtColor(img_8u, lab, cv::COLOR_BGR2Lab);
    }

    const bool has_lab = config_.use_rgb_guide && !lab.empty();

    for (int r = 0; r < H; ++r) {
        float* guide_row = guide.ptr<float>(r);
        const float* depth_row = depth.ptr<float>(r);
        const float* edge_row = edge_map.ptr<float>(r);
        const cv::Vec3b* lab_row = has_lab ? lab.ptr<cv::Vec3b>(r) : nullptr;

        for (int c = 0; c < W; ++c) {
            int ch = 0;

            // Lab chroma channel (combined a,b → single color-distance channel)
            // Skip L for illumination invariance; merge a,b as Euclidean norm
            if (has_lab) {
                float A_norm = (static_cast<float>(lab_row[c][1]) - 128.0f) / 128.0f;
                float B_norm = (static_cast<float>(lab_row[c][2]) - 128.0f) / 128.0f;
                float chroma = std::sqrt(A_norm * A_norm + B_norm * B_norm);
                guide_row[c * num_channels + ch++] = config_.wrgb * chroma;
            }

            // Depth channel
            float d = depth_row[c];
            if (!std::isfinite(d)) d = 0.0f;
            guide_row[c * num_channels + ch++] = config_.wx * d;

            // Edge channel (with power)
            float e = edge_row[c];
            if (!std::isfinite(e)) e = 0.0f;
            float e_pow = std::pow(e, config_.grad_power);
            guide_row[c * num_channels + ch++] = config_.wgrad * e_pow;
        }
    }

    // Sky mask: set sky pixels to median of non-sky pixels
    if (!sky_mask.empty()) {
        // Compute median of non-sky pixels for each channel
        std::vector<std::vector<float>> non_sky_values(num_channels);
        for (int ch = 0; ch < num_channels; ++ch) {
            non_sky_values[ch].reserve(H * W);
        }

        for (int r = 0; r < H; ++r) {
            const uint8_t* mask_row = sky_mask.ptr<uint8_t>(r);
            const float* guide_row = guide.ptr<float>(r);
            for (int c = 0; c < W; ++c) {
                if (mask_row[c] == 0) {  // Not sky
                    for (int ch = 0; ch < num_channels; ++ch) {
                        non_sky_values[ch].push_back(guide_row[c * num_channels + ch]);
                    }
                }
            }
        }

        // Compute medians
        std::vector<float> medians(num_channels, 0.0f);
        for (int ch = 0; ch < num_channels; ++ch) {
            if (!non_sky_values[ch].empty()) {
                std::nth_element(non_sky_values[ch].begin(),
                                non_sky_values[ch].begin() + non_sky_values[ch].size() / 2,
                                non_sky_values[ch].end());
                medians[ch] = non_sky_values[ch][non_sky_values[ch].size() / 2];
            }
        }

        // Set sky pixels to median
        for (int r = 0; r < H; ++r) {
            const uint8_t* mask_row = sky_mask.ptr<uint8_t>(r);
            float* guide_row = guide.ptr<float>(r);
            for (int c = 0; c < W; ++c) {
                if (mask_row[c] != 0) {  // Is sky
                    for (int ch = 0; ch < num_channels; ++ch) {
                        guide_row[c * num_channels + ch] = medians[ch];
                    }
                }
            }
        }
    }

    return guide;
}

std::pair<cv::Mat, int> DepthRefinement::compute_segmentation(const cv::Mat& img,
                                                               const cv::Mat& depth,
                                                               const cv::Mat& sky_mask) {
#ifdef __ANDROID__
    // Android: Segmentation not available (ximgproc not in Android SDK)
    cv::Mat labels = cv::Mat::zeros(img.rows, img.cols, CV_32S);
    return {labels, 1};
#else
    if (!config_.use_segmentation || !graph_seg_) {
        // Return single segment covering entire image
        cv::Mat labels = cv::Mat::zeros(img.rows, img.cols, CV_32S);
        return {labels, 1};
    }

    const int H = img.rows;
    const int W = img.cols;

    // 1) Compute edge map from depth
    cv::Mat edge_map = compute_edge(depth);

    // 2) Build multi-channel guide image
    cv::Mat guide = build_guide(img, depth, edge_map, sky_mask);

    // 3) Optional downsample (seg_down < 1.0 means downsample)
    cv::Mat guide_for_seg;
    int seg_H = H, seg_W = W;

    if (config_.seg_down > 0.0f && config_.seg_down < 0.999f) {
        seg_W = std::max(1, static_cast<int>(W * config_.seg_down));
        seg_H = std::max(1, static_cast<int>(H * config_.seg_down));
        cv::resize(guide, guide_for_seg, cv::Size(seg_W, seg_H), 0, 0, cv::INTER_AREA);
    } else {
        guide_for_seg = guide;
    }

    // 4) Convert guide to 8-bit for GraphSegmentation
    // GraphSegmentation expects CV_8UC3, so we need to convert our multi-channel guide
    // Option: normalize and convert to pseudo-RGB or use first 3 channels
    cv::Mat guide_8u;
    int guide_channels = guide_for_seg.channels();

    if (guide_channels >= 3) {
        // Use first 3 channels, normalize to 0-255
        std::vector<cv::Mat> channels(guide_channels);
        cv::split(guide_for_seg, channels);

        std::vector<cv::Mat> rgb_channels(3);
        for (int i = 0; i < 3; ++i) {
            double minVal, maxVal;
            cv::minMaxLoc(channels[i], &minVal, &maxVal);
            if (maxVal - minVal > 1e-6) {
                channels[i].convertTo(rgb_channels[i], CV_8U, 255.0 / (maxVal - minVal),
                                      -minVal * 255.0 / (maxVal - minVal));
            } else {
                rgb_channels[i] = cv::Mat::zeros(seg_H, seg_W, CV_8U);
            }
        }
        cv::merge(rgb_channels, guide_8u);
    } else {
        // Single channel guide - convert to grayscale and then to BGR
        double minVal, maxVal;
        cv::minMaxLoc(guide_for_seg, &minVal, &maxVal);
        cv::Mat guide_norm;
        if (maxVal - minVal > 1e-6) {
            guide_for_seg.convertTo(guide_norm, CV_8U, 255.0 / (maxVal - minVal),
                                    -minVal * 255.0 / (maxVal - minVal));
        } else {
            guide_norm = cv::Mat::zeros(seg_H, seg_W, CV_8U);
        }
        cv::cvtColor(guide_norm, guide_8u, cv::COLOR_GRAY2BGR);
    }

    // 5) Apply Gaussian blur (as in Python skimage felzenszwalb)
    if (config_.seg_sigma > 0) {
        int ksize = static_cast<int>(config_.seg_sigma * 4) | 1;
        ksize = std::max(ksize, 3);
        cv::GaussianBlur(guide_8u, guide_8u, cv::Size(ksize, ksize), config_.seg_sigma);
    }

    // 6) Run Felzenszwalb segmentation
    cv::Mat labels_small;
    graph_seg_->processImage(guide_8u, labels_small);

    // 7) Upsample labels if downsampled
    cv::Mat labels;
    if (seg_H != H || seg_W != W) {
        cv::resize(labels_small, labels, cv::Size(W, H), 0, 0, cv::INTER_NEAREST);
    } else {
        labels = labels_small;
    }

    // Ensure int32
    if (labels.type() != CV_32S) {
        labels.convertTo(labels, CV_32S);
    }

    // Count unique labels
    double minVal, maxVal;
    cv::minMaxLoc(labels, &minVal, &maxVal);
    int num_segments = static_cast<int>(maxVal) + 1;

    return {labels, num_segments};
#endif  // __ANDROID__
}

DepthRefinement::MotionResult DepthRefinement::estimate_motion(
    const cv::Mat& flow,
    const cv::Mat& inv_depth,
    const cv::Mat& mask,
    const std::optional<Eigen::Vector3d>& known_omega
) {
    MotionResult result;
    result.success = false;
    result.num_inliers = 0;

    // Create motion field estimator
    MotionFieldConfig mf_cfg;
    mf_cfg.ransac_max_iters = config_.ransac_max_iters;
    mf_cfg.ransac_min_sample = config_.ransac_min_sample;
    mf_cfg.ransac_thresh_ratio = config_.ransac_thresh_ratio;
    mf_cfg.min_flow_px = config_.min_flow_px;
    mf_cfg.max_points = config_.max_points;
    mf_cfg.margin_x_pct = config_.margin_x_pct;
    mf_cfg.margin_y_pct = config_.margin_y_pct;
    mf_cfg.depth_bins = config_.depth_bins;
    mf_cfg.adaptive_flow_depth_scale = config_.adaptive_flow_depth_scale;
    mf_cfg.seed = 42;  // Fixed seed for reproducibility

    mf_cfg.lo_irls_iters = 5;
    mf_cfg.huber_delta_rel = 3.5f;         // ipynb: 3.5
    mf_cfg.mad_scale = 5.5f;               // ipynb: 5.5

    // Match ipynb MOTION_CONFIG for MAGSAC and depth_scale_mode
    mf_cfg.depth_scale_mode = 0;           // ipynb: 0 (none)
    mf_cfg.magsac_rel_sigma_max = 0.25f;   // ipynb: 0.25
    mf_cfg.magsac_inlier_weight = 0.5f;    // ipynb: 0.5
    mf_cfg.use_magsac_scoring = config_.use_magsac_scoring;  // Paper vs MAGSAC++ mode

    MotionFieldEstimator motion_estimator(mf_cfg);

    // Set up camera intrinsics
    CameraIntrinsics intrinsics;
    intrinsics.fx = config_.fx;
    intrinsics.fy = config_.fy;
    intrinsics.cx = config_.cx;
    intrinsics.cy = config_.cy;

    try {
        MotionFieldResult mf_result = motion_estimator.estimate(flow, inv_depth, intrinsics, mask, known_omega);

        result.R = mf_result.R;
        result.t = mf_result.t;
        result.omega = mf_result.omega;
        result.u0 = mf_result.u0;
        result.v0 = mf_result.v0;
        result.u1 = mf_result.u1;
        result.v1 = mf_result.v1;
        result.num_inliers = mf_result.num_inliers;
        result.success = (mf_result.num_inliers >= 50);

    } catch (const std::exception& e) {
        // Always print exception for debugging - motion estimation failures are critical
        std::cerr << "[MotionEstimation] FAILED: " << e.what() << std::endl;
        std::cerr << "  flow shape: " << flow.rows << "x" << flow.cols
                  << ", inv_depth shape: " << inv_depth.rows << "x" << inv_depth.cols << std::endl;
        result.success = false;
    }

    return result;
}

DepthRefinement::TriResult DepthRefinement::triangulate(
    const std::vector<float>& u0,
    const std::vector<float>& v0,
    const std::vector<float>& u1,
    const std::vector<float>& v1,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t
) {
    TriResult result;
    result.num_valid = 0;

    if (u0.empty()) {
        result.z1_tri = cv::Mat(config_.H, config_.W, CV_32F, cv::Scalar(config_.fill_value));
        result.rpx_tri = cv::Mat(config_.H, config_.W, CV_32F, cv::Scalar(config_.fill_value));
        return result;
    }

    // Set up triangulator config
    TriangulationConfig tri_cfg;
    tri_cfg.fill_value = config_.fill_value;

    Triangulator triangulator(config_.H, config_.W,
                              config_.fx, config_.fy, config_.cx, config_.cy,
                              tri_cfg);

    // Use optimized vector-based triangulation (no cv::Mat conversion)
    TriangulationResult tri_result = triangulator.triangulate_vec(u0, v0, u1, v1, R, t);

    result.z1_tri = tri_result.z1_tri;
    result.rpx_tri = tri_result.rpx_tri;
    result.num_valid = tri_result.num_valid;

    return result;
}

DepthRefinement::ScaleResult DepthRefinement::robust_scale_match(
    const cv::Mat& z_tri,
    const cv::Mat& z_ref,
    const cv::Mat& mask
) {
    ScaleResult result;
    result.scale = 1.0f;
    result.scale_ok = false;
    result.overlap = 0;
    result.median_relerr = std::numeric_limits<float>::infinity();

    RobustScaleConfig scale_cfg;
    scale_cfg.min_overlap = config_.min_scale_overlap;
    scale_cfg.tol_median = config_.scale_tol_median;
    scale_cfg.max_depth = config_.max_depth;

    RobustScaleResult rs_result = pr_depth::robust_scale_match(z_tri, z_ref, mask, scale_cfg);

    result.scale = rs_result.scale;
    result.scale_ok = rs_result.scale_ok;
    result.overlap = rs_result.overlap;
    result.median_relerr = rs_result.median_relerr;

    return result;
}

void DepthRefinement::update_state(
    const cv::Mat& img,
    const cv::Mat& inv_depth,
    const cv::Mat& depth,
    const cv::Mat& V
) {
    img.copyTo(prev_img_);
    inv_depth.copyTo(prev_inv_depth_);
    depth.copyTo(prev_depth_);
    V.copyTo(prev_V_);
}

void DepthRefinement::prepare_flow(const cv::Mat& img) {
    const int H = config_.H;
    const int W = config_.W;

    // Resize if needed
    cv::Mat img_resized;
    if (img.rows != H || img.cols != W) {
        cv::resize(img, img_resized, cv::Size(W, H));
    } else {
        img_resized = img;
    }

    // Compute flow only if we have a previous frame
    if (!prev_img_.empty()) {
        precomputed_flow_ = compute_flow(prev_img_, img_resized);
    } else {
        precomputed_flow_ = cv::Mat();
    }

    precomputed_img_ = img_resized;
    flow_ready_ = true;
}

// 4-param version: calls 6-param version with no GT pose
DepthRefinementResult DepthRefinement::refine(
    const cv::Mat& img,
    const cv::Mat& inv_depth,
    float baseline,
    const cv::Mat& seg_labels
) {
    return refine(img, inv_depth, baseline, std::nullopt, std::nullopt, seg_labels);
}

// 6-param version: main implementation with optional GT pose
DepthRefinementResult DepthRefinement::refine(
    const cv::Mat& img,
    const cv::Mat& inv_depth,
    float baseline,
    const std::optional<Eigen::Matrix3d>& gt_R,
    const std::optional<Eigen::Vector3d>& gt_t,
    const cv::Mat& seg_labels
) {
    using Clock = std::chrono::high_resolution_clock;
    auto t_start = Clock::now();
    auto t_prev = t_start;
    auto log_time = [&](const char* name) {
        if (config_.timing) {
            auto now = Clock::now();
            double ms = std::chrono::duration<double, std::milli>(now - t_prev).count();
            PR_TIMING_LOG("[Timing] %s: %.1f ms", name, ms);
            t_prev = now;
        }
    };

    DepthRefinementResult result;
    result.baseline = baseline;
    result.R = Eigen::Matrix3d::Identity();
    result.t = Eigen::Vector3d(0, 0, -1);  // Default forward direction (-Z in camera frame)
    result.num_matches = 0;
    result.num_valid_tri = 0;
    result.num_segments = 0;
    result.tri_disabled = false;
    result.baseline_correction = 1.0f;

    // Initialize debug fields (only used when config_.debug = true)
    result.used_backward = false;

    // Initialize GT pose fallback fields
    result.used_gt_pose = false;
    result.rotation_angle_deg = 0.0f;

    const int H = config_.H;
    const int W = config_.W;

    // Consume precomputed flow state on early returns
    // (actual consumption happens in the flow section below)
    auto clear_precomputed_flow = [&]() {
        if (flow_ready_) {
            precomputed_flow_ = cv::Mat();
            precomputed_img_ = cv::Mat();
            flow_ready_ = false;
        }
    };

    // Resize inputs if needed
    cv::Mat img_resized, inv_depth_resized;
    if (img.rows != H || img.cols != W) {
        cv::resize(img, img_resized, cv::Size(W, H));
    } else {
        img_resized = img;
    }

    if (inv_depth.rows != H || inv_depth.cols != W) {
        cv::resize(inv_depth, inv_depth_resized, cv::Size(W, H));
    } else {
        inv_depth_resized = inv_depth;
    }

    // Ensure float32
    cv::Mat inv_depth_f;
    if (inv_depth_resized.type() != CV_32F) {
        inv_depth_resized.convertTo(inv_depth_f, CV_32F);
    } else {
        inv_depth_f = inv_depth_resized;
    }

    // ===== OPTIMIZED: Create depth_for_seg, sky_mask_for_seg in ONE pass =====
    cv::Mat depth_for_seg(H, W, CV_32F);
    cv::Mat sky_mask_for_seg(H, W, CV_8U);

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < H; ++r) {
        const float* inv_row = inv_depth_f.ptr<float>(r);
        float* d_row = depth_for_seg.ptr<float>(r);
        uint8_t* sky_seg_row = sky_mask_for_seg.ptr<uint8_t>(r);
        for (int c = 0; c < W; ++c) {
            float inv_val = inv_row[c];
            bool is_valid;
            if (config_.sky_mask_inv_thresh <= 0.0f) {
                // Sky masking disabled — all finite pixels are valid
                is_valid = std::isfinite(inv_val) && (inv_val > 0.0f);
            } else {
                is_valid = (inv_val >= config_.sky_mask_inv_thresh) && std::isfinite(inv_val);
            }
            sky_seg_row[c] = is_valid ? 0 : 1;
            d_row[c] = is_valid ? (1.0f / (inv_val + 1e-6f)) : config_.sky_fallback_depth;
        }
    }

    // First frame: just store state and return NaN (like Python)
    // No valid depth estimation possible without previous frame for optical flow
    if (frame_count_ == 0 || prev_img_.empty()) {
        // Skip segmentation on first frame - no depth estimation occurs
        result.seg_labels = cv::Mat::zeros(H, W, CV_32S);
        result.num_segments = 1;

        // Return NaN for z_refined - no valid depth on first frame
        result.z_refined = cv::Mat(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
        result.z_tri = cv::Mat(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
        result.confidence = cv::Mat::zeros(H, W, CV_32F);  // Zero confidence

        // Store current frame for next iteration (no depth state yet)
        prev_img_ = img_resized.clone();
        prev_inv_depth_ = inv_depth_f.clone();
        frame_count_++;

        return result;
    }

    // Update baseline EMA tracker (always, even if guard is disabled)
    baseline_state_.update(baseline);

    // Check baseline guard with adaptive threshold from EMA
    // should_disable() uses percentile-based threshold from history
    if (config_.use_baseline_guard && baseline_state_.should_disable(baseline, config_.min_baseline)) {
        result.tri_disabled = true;
        result.z_tri = cv::Mat(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));

        // Set dummy seg_labels (single segment) for early return
        result.seg_labels = cv::Mat::zeros(H, W, CV_32S);
        result.num_segments = 1;

        // If we have previous depth, return it; otherwise return NaN
        if (!prev_depth_.empty()) {
            prev_depth_.copyTo(result.z_refined);

            result.confidence = variance_to_confidence(prev_V_);

            // Update state (keep previous depth, increase variance)
            cv::Mat V_new;
            prev_V_.copyTo(V_new);
            V_new *= 1.1f;  // Increase variance

            update_state(img_resized, inv_depth_f, prev_depth_, V_new);
        } else {
            // No previous depth available, return NaN
            result.z_refined = cv::Mat(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
            result.confidence = cv::Mat::zeros(H, W, CV_32F);
            prev_img_ = img_resized.clone();
            prev_inv_depth_ = inv_depth_f.clone();
        }
        frame_count_++;
        clear_precomputed_flow();
        return result;
    }

    // ======== Run segmentation and optical flow in PARALLEL ========
    // These are independent operations, so we can execute them concurrently
    cv::Mat computed_seg_labels;
    int num_segments = 0;
    cv::Mat flow;
    cv::Mat flow_backward;  // Backward flow: curr → prev (for iterative refinement)

    // Check if flow was precomputed (via prepare_flow() for NPU/CPU overlap)
    bool used_precomputed_flow = false;
    if (flow_ready_ && !precomputed_flow_.empty()) {
        flow = precomputed_flow_;
        img_resized = precomputed_img_;  // Use the same image that flow was computed with
        precomputed_flow_ = cv::Mat();
        precomputed_img_ = cv::Mat();
        flow_ready_ = false;
        used_precomputed_flow = true;
    }

    double time_seg = 0, time_flow = 0, time_flow_back = 0;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            auto t0_seg = std::chrono::high_resolution_clock::now();
            // Compute EdgeAware segmentation
            if (seg_labels.empty()) {
                auto [labels, n_seg] = compute_segmentation(img_resized, depth_for_seg, sky_mask_for_seg);
                computed_seg_labels = labels;
                num_segments = n_seg;
            } else {
                computed_seg_labels = seg_labels;
                double minVal, maxVal;
                cv::minMaxLoc(seg_labels, &minVal, &maxVal);
                num_segments = static_cast<int>(maxVal) + 1;
            }
            auto t1_seg = std::chrono::high_resolution_clock::now();
            time_seg = std::chrono::duration<double, std::milli>(t1_seg - t0_seg).count();
        }

        #pragma omp section
        {
            auto t0_flow = std::chrono::high_resolution_clock::now();
            if (!used_precomputed_flow) {
                // Compute forward optical flow: prev → curr
                flow = compute_flow(prev_img_, img_resized);
            }
            auto t1_flow = std::chrono::high_resolution_clock::now();
            time_flow = std::chrono::duration<double, std::milli>(t1_flow - t0_flow).count();
        }

        #pragma omp section
        {
            // Skip backward flow if FB consistency check is disabled
            if (!config_.skip_fb_consistency) {
                auto t0_flow_back = std::chrono::high_resolution_clock::now();
                // Compute backward optical flow: curr → prev (for iterative refinement)
                // Uses separate DIS instance (dis_flow_backward_) for thread safety
                cv::Mat gray_curr_back, gray_prev_back;
                if (img_resized.channels() == 3) {
                    cv::cvtColor(img_resized, gray_curr_back, cv::COLOR_BGR2GRAY);
                } else {
                    gray_curr_back = img_resized;
                }
                if (prev_img_.channels() == 3) {
                    cv::cvtColor(prev_img_, gray_prev_back, cv::COLOR_BGR2GRAY);
                } else {
                    gray_prev_back = prev_img_;
                }
                dis_flow_backward_->calc(gray_curr_back, gray_prev_back, flow_backward);
                auto t1_flow_back = std::chrono::high_resolution_clock::now();
                time_flow_back = std::chrono::duration<double, std::milli>(t1_flow_back - t0_flow_back).count();
            }
        }
    }

    result.seg_labels = computed_seg_labels;
    result.num_segments = num_segments;

    // Build LabelIndex for efficient per-label operations (reused in solve_metric_from_rel)
    // Use -1 to auto-detect num_labels from actual label values (matches cv::Mat version behavior)
    LabelIndex label_index = build_label_index(computed_seg_labels, -1);

    if (config_.timing) {
        PR_TIMING_LOG("[Breakdown] seg=%.1fms flow=%.1fms back_flow=%.1fms", time_seg, time_flow, time_flow_back);
    }
    log_time("segmentation+optical_flow+backward_flow (parallel)");

    // ======== Forward-Backward Consistency Check ========
    // Computes mask of reliable flow pixels based on forward-backward consistency
    // consistency_error[x,y] = |forward_flow[x,y] + backward_flow[x+u, y+v]|
    // If error > threshold, mark as unreliable (occlusion/disocclusion boundary)
    cv::Mat flow_consistency_mask;
    int num_consistent = 0;
    int num_checked = 0;

    if (config_.skip_fb_consistency) {
        // Skip FB check: treat all pixels as valid
        flow_consistency_mask = cv::Mat(H, W, CV_8U, cv::Scalar(1));
        if (config_.timing) {
            PR_TIMING_LOG("[Timing] fb_consistency_check: SKIPPED");
        }
    } else {
        flow_consistency_mask = cv::Mat(H, W, CV_8U, cv::Scalar(0));
        const float fb_consistency_thresh_sq = 1.5f * 1.5f;  // squared threshold (avoids sqrt)

        auto t0_fbcheck = std::chrono::high_resolution_clock::now();

        for (int v = 0; v < H; ++v) {
            const float* flow_fwd_row = flow.ptr<float>(v);
            uint8_t* mask_row = flow_consistency_mask.ptr<uint8_t>(v);

            for (int u = 0; u < W; ++u) {
                // Forward flow at (u, v): points to (u + fu, v + fv) in frame t
                float fu = flow_fwd_row[u * 2 + 0];
                float fv = flow_fwd_row[u * 2 + 1];

                // Skip if flow is invalid
                if (!std::isfinite(fu) || !std::isfinite(fv)) continue;

                // Target position in frame t
                int u1 = static_cast<int>(std::round(u + fu));
                int v1 = static_cast<int>(std::round(v + fv));

                // Check bounds
                if (u1 < 0 || u1 >= W || v1 < 0 || v1 >= H) continue;

                // Backward flow at target position
                const float* flow_bwd_row = flow_backward.ptr<float>(v1);
                float bu = flow_bwd_row[u1 * 2 + 0];
                float bv = flow_bwd_row[u1 * 2 + 1];

                // Skip if backward flow is invalid
                if (!std::isfinite(bu) || !std::isfinite(bv)) continue;

                ++num_checked;

                // Forward-backward consistency error (squared, avoids sqrt)
                // If consistent: forward + backward ≈ 0 (returns to original position)
                float error_u = fu + bu;
                float error_v = fv + bv;
                float error_sq = error_u * error_u + error_v * error_v;

                if (error_sq <= fb_consistency_thresh_sq) {
                    mask_row[u] = 1;  // Consistent (reliable)
                    ++num_consistent;
                }
                // else: inconsistent (occlusion/disocclusion), mask stays 0
            }
        }

        auto t1_fbcheck = std::chrono::high_resolution_clock::now();
        double time_fbcheck = std::chrono::duration<double, std::milli>(t1_fbcheck - t0_fbcheck).count();

        if (config_.timing) {
            float consistency_ratio = num_checked > 0 ? 100.0f * num_consistent / num_checked : 0.0f;
            PR_TIMING_LOG("[Timing] fb_consistency: %.1fms (consistent %d/%d = %.0f%%)",
                         time_fbcheck, num_consistent, num_checked, consistency_ratio);
        }
    }
    log_time("fb_consistency_check");

    // ======== Iterative Refinement Loop ========
    // Iteration 0: use original inv_depth from monocular model
    // Iteration 1+: use refined depth from previous iteration
    const int total_iters = config_.enable_iterative_refinement ?
                            (1 + config_.iterative_refinement_iters) : 1;

    cv::Mat inv_depth_for_motion = prev_inv_depth_;  // Start with monocular prior
    cv::Mat z1_warp_flow, V_warp_flow;  // Warped prior from prev_depth_ (for iter 0)
    cv::Mat z_prior, V_prior;           // Prior for current iteration (updated each iter)
    FusionConfig fuse_cfg;
    fuse_cfg.chi2_hard = config_.fusion_chi2_hard;
    fuse_cfg.kcap_floor = config_.fusion_kcap_floor;
    // Baseline-proportional scaling: scale temporal parameters by motion amount
    // When camera is stationary (baseline ≈ 0), trust prior fully (no variance increase)
    // When camera moves normally (baseline = b_ref), behave as before
    // When camera moves a lot (baseline > b_ref), trust new observations more
    float b_ref = baseline_state_.b_ref();
    float baseline_ratio = (b_ref > 1e-6f) ? (baseline / b_ref) : 1.0f;

    fuse_cfg.lambda_forget = config_.fusion_lambda_forget * baseline_ratio;
    fuse_cfg.min_var = config_.fusion_var_floor;

    // Declare fusion state variables outside loop (visible after loop ends)
    cv::Mat V_post = cv::Mat(H, W, CV_32F, cv::Scalar(100.0f));
    cv::Mat z_fused_sparse;  // Sparse fusion result (before solve_metric_from_rel)

    // When skip_temporal_fusion is true, treat every frame as first frame (no prior)
    bool has_prev_depth = !prev_depth_.empty() && !config_.skip_temporal_fusion;

    // Pre-compute dense warp from prev_depth_ (for iteration 0)
    if (has_prev_depth) {
        warp_dense_combined(prev_depth_, prev_V_, flow, H, W,
                           fuse_cfg.lambda_forget, fuse_cfg.min_var,
                           z1_warp_flow, V_warp_flow);
    }
    log_time("warp_dense");

    // Save debug depth maps early (before any early returns)
    if (config_.debug) {
        if (!z1_warp_flow.empty()) {
            result.z_warp_flow = z1_warp_flow.clone();
        }
        if (!prev_depth_.empty()) {
            result.prev_depth_used = prev_depth_.clone();
        }
    }

    // Main iteration loop
    cv::Mat flow_for_motion = flow;  // Will be changed to reverse flow on iter > 0

    // Declare pose and coordinate variables outside loop (persist for final triangulation)
    Eigen::Matrix3d R_for_tri = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_for_tri = Eigen::Vector3d::Zero();
    std::vector<float> u0_tri, v0_tri, u1_tri, v1_tri;

    // Declare triangulation result outside loop (for final pass after pose-only iterations)
    TriResult tri_result;
    cv::Mat z1_warp_pose, V_warp_pose;
    cv::Mat V_obs(H, W, CV_32F);
    bool use_flow_warp = false;

    // Pre-build variance config (reused every iteration)
    DepthFusionConfig var_cfg;
    var_cfg.tau0_deg = config_.fusion_tau0_deg;
    var_cfg.sigma2_at_tau = config_.fusion_sigma2_at_tau;
    var_cfg.var_floor = config_.fusion_var_floor;
    var_cfg.var_cap = config_.fusion_var_cap;

    for (int iter = 0; iter < total_iters; ++iter) {
        // For iter > 0: use backward motion estimation with refined depth
        // This resolves forward motion degeneracy by using the refined depth
        // at frame t to constrain the motion estimation
        bool use_backward_estimation = (iter > 0) && !result.z_refined.empty();

        if (config_.debug && total_iters > 1) {
            std::cerr << "[IterRefine] Iteration " << iter << "/" << total_iters
                      << (use_backward_estimation ? " (backward)" : " (forward)") << std::endl;
        }

        // ======== Prepare flow and depth for motion estimation ========
        if (use_backward_estimation) {
            // Use actual backward flow (curr → prev) computed in parallel
            // This is more accurate than just flipping forward flow
            flow_for_motion = flow_backward;

            // Compute inv_depth from z_refined for backward motion estimation
            // At frame t, use refined depth to constrain backward motion (t → t-1)
            inv_depth_for_motion = cv::Mat(H, W, CV_32F);
            for (int r = 0; r < H; ++r) {
                const float* z_row = result.z_refined.ptr<float>(r);
                float* inv_row = inv_depth_for_motion.ptr<float>(r);
                for (int c = 0; c < W; ++c) {
                    float z = z_row[c];
                    if (std::isfinite(z) && z > 0.5f && z < 200.0f) {
                        inv_row[c] = 1.0f / z;
                    } else {
                        inv_row[c] = 0.0f;
                    }
                }
            }
            if (config_.debug) {
                std::cerr << "[IterRefine] Using actual backward flow and inv_depth from z_refined" << std::endl;
            }
        } else {
            flow_for_motion = flow;  // Use original forward flow
            // inv_depth_for_motion uses prev_inv_depth_ (previous frame's inv_depth)
            // This matches ipynb which uses img_prev's depth for flow prev→curr
        }

        // ======== Set prior for fusion ========
        // Iter 0: Use warped prev_depth_ as prior
        // Iter 1+: Use z_refined from previous iteration as prior (already in frame t coords)
        if (iter == 0) {
            z_prior = z1_warp_flow;   // Warped from prev_depth_
            V_prior = V_warp_flow;
            if (config_.debug && total_iters > 1) {
                std::cerr << "[IterRefine] Using warped prev_depth_ as prior" << std::endl;
            }
        } else {
            // iter > 0: z_prior/V_prior will be overridden by warp comparison block below
            // (pose warp or flow warp selected based on consistency check)
            // No clone needed here.
            if (config_.debug) {
                std::cerr << "[IterRefine] iter " << iter << ": prior will be set by warp comparison" << std::endl;
            }
        }

    // ======== Motion estimation ========
    // For iter 0 (forward): use forward-backward consistency mask to filter unreliable pixels
    // For iter > 0 (backward): consistency mask is in prev frame coords, not directly applicable
    // motion_field.cpp also filters by r_percentile_3 and isfinite checks internally
    cv::Mat motion_mask = use_backward_estimation ? cv::Mat() : flow_consistency_mask;

    // ======== GT Pose Direct Mode ========
    // When GT pose is enabled and provided, skip RANSAC entirely for massive speedup
    // Correspondences are extracted directly from optical flow
    bool use_gt_direct = config_.use_gt_pose_fallback &&
                         gt_R.has_value() && gt_t.has_value() &&
                         iter == 0 && !use_backward_estimation;

    MotionResult motion_result;

    if (use_gt_direct) {
        // Skip motion estimation entirely — extract correspondences from flow
        motion_result.success = true;
        motion_result.R = gt_R.value();
        motion_result.t = gt_t.value();

        // Generate correspondences from optical flow (subsample every 2 pixels)
        const int corr_step = 2;
        for (int v = 0; v < H; v += corr_step) {
            const float* flow_row = flow_for_motion.ptr<float>(v);
            for (int u = 0; u < W; u += corr_step) {
                float fx = flow_row[u * 2 + 0];
                float fy = flow_row[u * 2 + 1];
                if (!std::isfinite(fx) || !std::isfinite(fy)) continue;
                float mag_sq = fx * fx + fy * fy;
                if (mag_sq < config_.min_flow_px * config_.min_flow_px) continue;

                float u_curr = u + fx;
                float v_curr = v + fy;
                if (u_curr < 0 || u_curr >= W || v_curr < 0 || v_curr >= H) continue;

                motion_result.u0.push_back(static_cast<float>(u));
                motion_result.v0.push_back(static_cast<float>(v));
                motion_result.u1.push_back(u_curr);
                motion_result.v1.push_back(v_curr);
            }
        }
        motion_result.num_inliers = static_cast<int>(motion_result.u0.size());
        log_time("motion_estimation (SKIPPED - GT pose direct)");
    } else {
        // Normal motion estimation with RANSAC
        // use_gt_R mode: pass GT rotation to estimate only translation
        std::optional<Eigen::Vector3d> known_omega_for_motion = std::nullopt;
        if (config_.use_gt_R && gt_R.has_value()) {
            Eigen::Matrix3d R_gt_val = gt_R.value();
            bool R_valid = true;
            for (int i = 0; i < 3 && R_valid; ++i) {
                for (int j = 0; j < 3 && R_valid; ++j) {
                    if (!std::isfinite(R_gt_val(i, j))) R_valid = false;
                }
            }
            if (R_valid) {
                Eigen::Matrix3d R_cam = R_gt_val.transpose();
                Eigen::Vector3d omega = rotation_matrix_to_omega(R_cam);
                if (omega.allFinite() && omega.norm() < 10.0) {
                    known_omega_for_motion = omega;
                }
            }
        }

        motion_result = estimate_motion(flow_for_motion, inv_depth_for_motion, motion_mask, known_omega_for_motion);
        log_time(iter == 0 ? "motion_estimation" : "motion_estimation_iter");
    }

    // Compute rotation angle
    float rotation_angle = compute_rotation_angle_deg(motion_result.R);
    result.rotation_angle_deg = rotation_angle;

    // ======== GT Pose for Triangulation ========
    // use_gt_direct: always use GT pose (motion estimation was skipped)
    // fallback: use GT pose only when rotation exceeds threshold
    bool use_gt_pose_this_frame = use_gt_direct ||
                                   (config_.use_gt_pose_fallback &&
                                    gt_R.has_value() && gt_t.has_value() &&
                                    rotation_angle >= config_.gt_pose_rotation_threshold_deg &&
                                    motion_result.success &&
                                    iter == 0);

    if (use_gt_pose_this_frame) {
        result.used_gt_pose = true;

        // Use GT pose
        Eigen::Matrix3d R_gt = gt_R.value();
        Eigen::Vector3d t_gt_raw = gt_t.value();

        // Normalize GT translation to baseline (GT t may have different scale)
        double t_gt_norm = t_gt_raw.norm();
        Eigen::Vector3d t_gt_scaled = (t_gt_norm > 1e-8)
            ? t_gt_raw.normalized() * baseline
            : Eigen::Vector3d(0, 0, -baseline);

        result.R = R_gt;
        result.t = t_gt_scaled;
        result.num_matches = motion_result.num_inliers;  // Still report estimated inliers

        if (config_.debug) {
            std::cerr << "[GT Pose Fallback] rotation=" << rotation_angle
                      << " deg >= threshold=" << config_.gt_pose_rotation_threshold_deg
                      << " deg, using GT pose for triangulation" << std::endl;
            std::cerr << "[GT Pose Fallback] baseline=" << baseline
                      << ", |t_gt_raw|=" << t_gt_norm
                      << ", |t_gt_scaled|=" << t_gt_scaled.norm() << std::endl;
        }

        // Get correspondences from motion estimation (still use estimated correspondences)
        // These come from optical flow, which is independent of pose
        std::vector<float> u0_gt = std::move(motion_result.u0);
        std::vector<float> v0_gt = std::move(motion_result.v0);
        std::vector<float> u1_gt = std::move(motion_result.u1);
        std::vector<float> v1_gt = std::move(motion_result.v1);

        // Convert GT pose to triangulation coordinates
        // Motion field convention: p_curr = R_gt @ p_prev + t_gt (R transforms prev→curr)
        // Triangulation needs ray transform from curr→prev, which is R^T
        Eigen::Matrix3d R_for_tri_gt = R_gt.transpose();
        // Camera position in frame 0 (prev): C1 = -R^T @ t (now baseline-scaled)
        Eigen::Vector3d t_for_tri_gt = -R_for_tri_gt * t_gt_scaled;

        if (config_.debug) {
            std::cerr << "[GT Pose Fallback] t_for_tri_gt: " << t_for_tri_gt.transpose() << std::endl;
        }

        // Triangulate using GT pose - this gives FRESH metric depth from baseline
        TriResult tri_result_gt = triangulate(
            u0_gt, v0_gt,
            u1_gt, v1_gt,
            R_for_tri_gt, t_for_tri_gt
        );
        log_time("triangulation_gt_pose");

        result.z_tri = tri_result_gt.z1_tri;
        result.num_valid_tri = tri_result_gt.num_valid;

        // Also compute dense 3D warp for visualization/debugging
        // And compute z1_warp_pose using GT pose for fusion step
        if (has_prev_depth) {
            cv::Mat z_warp_gt = warp_depth_3d_dense(
                prev_depth_,
                R_gt, t_gt_scaled,
                config_.fx, config_.fy, config_.cx, config_.cy,
                H, W,
                0.5f, config_.max_depth
            );
            result.z_warp_gt = z_warp_gt.clone();

            // Dense 3D warp using GT pose for fusion
            warp_depth_3d_dense(
                prev_depth_, prev_V_,
                R_for_tri_gt, t_for_tri_gt,
                config_.fx, config_.fy, config_.cx, config_.cy,
                H, W,
                fuse_cfg.lambda_forget, fuse_cfg.min_var,
                z1_warp_pose, V_warp_pose
            );
        }

        // Now proceed with normal fusion pipeline using triangulated depth
        // Override the motion result variables for the rest of the pipeline
        R_for_tri = R_for_tri_gt;
        t_for_tri = t_for_tri_gt;
        u0_tri = u0_gt;
        v0_tri = v0_gt;
        u1_tri = u1_gt;
        v1_tri = v1_gt;

        // Set tri_result so fusion uses GT triangulation
        tri_result = tri_result_gt;

        // Continue to normal fusion pipeline (don't return early!)
        // The rest of the code will use tri_result for fusion
    }

    // Skip motion failure check and pose conversion if using GT pose
    // (GT pose already set up tri_result, R_for_tri, t_for_tri, etc.)
    if (!use_gt_pose_this_frame) {

    if (!motion_result.success) {
        // On first iteration: return failure but still provide R,t if available
        // On subsequent iterations: break and use previous iteration's result
        if (iter == 0) {
            result.tri_disabled = true;
            result.z_tri = cv::Mat(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));

            // Still provide R,t even if motion estimation "failed" (< 50 inliers)
            // The R,t may be inaccurate but user requested to always return them
            result.R = motion_result.R;
            result.t = motion_result.t;  // Already scaled by baseline in estimate_motion
            result.num_matches = motion_result.num_inliers;

            if (has_prev_depth) {
                prev_depth_.copyTo(result.z_refined);
                result.confidence = cv::Mat::ones(H, W, CV_32F) * 0.1f;
                update_state(img_resized, inv_depth_f, prev_depth_, prev_V_);
            } else {
                result.z_refined = cv::Mat(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
                result.confidence = cv::Mat::zeros(H, W, CV_32F);
                prev_img_ = img_resized.clone();
                prev_inv_depth_ = inv_depth_f.clone();
            }
            frame_count_++;
            return result;
        } else {
            // Use result from previous iteration
            if (config_.debug) {
                std::cerr << "[IterRefine] Motion failed at iteration " << iter << ", using previous result" << std::endl;
            }
            break;
        }
    }

    // ======== Process pose: invert if backward estimation ========
    // Variables R_for_tri, t_for_tri, u0_tri, v0_tri, u1_tri, v1_tri declared before loop

    if (use_backward_estimation) {
        // Backward estimation gave us R_rev, t_rev from frame t to frame t-1
        // Backward estimation: p_prev = R_rev @ p_curr + t_rev
        // After coordinate swap: frame 0 = prev, frame 1 = curr
        //
        // Triangulation needs R that transforms rays from frame 1 (curr) to frame 0 (prev)
        // R_rev transforms curr->prev, so we use R_rev directly
        //
        // Camera position: curr camera in prev frame = t_rev (from R_rev @ 0 + t_rev = t_rev)
        //
        // IMPORTANT: motion_result.t is NOT in metric scale - normalize to baseline
        Eigen::Matrix3d R_rev = motion_result.R;
        double t_norm = motion_result.t.norm();
        Eigen::Vector3d t_rev_scaled = (t_norm > 1e-8)
            ? motion_result.t.normalized() * baseline
            : Eigen::Vector3d(0, 0, -baseline);  // Default forward if t is zero

        // Triangulation now uses R directly, R_rev transforms curr->prev as needed
        R_for_tri = R_rev;

        // Camera 1 (curr) position in frame 0 (prev): C1 = t_rev (now baseline-scaled)
        t_for_tri = t_rev_scaled;

        // For output result: forward motion convention (p_curr = R_fwd @ p_prev + t_fwd)
        Eigen::Vector3d t_fwd = -R_rev.transpose() * t_rev_scaled;
        result.R = R_rev.transpose();  // R_fwd = R_rev^T
        result.t = t_fwd;  // Now baseline-scaled
        result.num_matches = motion_result.num_inliers;

        // For triangulation: swap backward's coordinates
        // Backward motion_result has:
        //   u0, v0 = source in frame t
        //   u1, v1 = target in frame t-1
        // For forward triangulation we need:
        //   u0_tri, v0_tri = frame t-1 coords
        //   u1_tri, v1_tri = frame t coords
        u0_tri = motion_result.u1;  // Backward's target (t-1) → forward's source
        v0_tri = motion_result.v1;
        u1_tri = motion_result.u0;  // Backward's source (t) → forward's target
        v1_tri = motion_result.v0;

        if (config_.debug) {
            std::cerr << "[IterRefine] Backward: t_cam = t_rev (convention conversion)" << std::endl;
            std::cerr << "[IterRefine] |t_rev_scaled|=" << t_rev_scaled.norm()
                      << ", |t_fwd|=" << t_fwd.norm()
                      << ", baseline=" << baseline << std::endl;
            std::cerr << "[IterRefine] t_for_tri: " << t_for_tri.transpose() << std::endl;
        }
    } else {
        // Forward estimation - use directly
        // IMPORTANT: motion_result.t is NOT in metric scale - normalize to baseline
        result.R = motion_result.R;
        double t_norm = motion_result.t.norm();
        Eigen::Vector3d t_scaled = (t_norm > 1e-8)
            ? motion_result.t.normalized() * baseline
            : Eigen::Vector3d(0, 0, -baseline);  // Default forward if t is zero
        result.t = t_scaled;  // Now baseline-scaled
        result.num_matches = motion_result.num_inliers;

        // Motion field: p_curr = R @ p_prev + t (R transforms prev→curr)
        // Triangulation needs ray transform from curr→prev, which is R^T
        // Triangulation code computes: R_input @ r1
        // To get R^T @ r1, we pass R^T
        R_for_tri = motion_result.R.transpose();

        // Camera position in frame 0 (prev): C1 = -R^T @ t_motion (now baseline-scaled)
        t_for_tri = -R_for_tri * t_scaled;

        // Use forward coordinates directly
        u0_tri = std::move(motion_result.u0);
        v0_tri = std::move(motion_result.v0);
        u1_tri = std::move(motion_result.u1);
        v1_tri = std::move(motion_result.v1);
    }

    // ======== For iter > 0: Save backward pose info but continue with full processing ========
    // iter 1+ will perform triangulation + fusion with improved pose
    if (iter > 0 && config_.debug) {
        result.R_backward = result.R;
        result.t_backward = result.t;
        std::cerr << "[IterRefine] Backward iteration " << iter << " complete, "
                  << "R improved, t_dir=" << result.t.transpose() << std::endl;
    }

    // ======== PARALLEL: triangulation || warp_pose (both need motion result) ========

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            // Triangulate using correct coordinates (swapped if backward estimation)
            tri_result = triangulate(
                u0_tri, v0_tri,
                u1_tri, v1_tri,
                R_for_tri, t_for_tri
            );
        }

        #pragma omp section
        {
            // Dense 3D pose-based warp (all pixels)
            if (has_prev_depth) {
                warp_depth_3d_dense(
                    prev_depth_, prev_V_,
                    R_for_tri, t_for_tri,
                    config_.fx, config_.fy, config_.cx, config_.cy,
                    H, W,
                    fuse_cfg.lambda_forget, fuse_cfg.min_var,
                    z1_warp_pose, V_warp_pose
                );
            }
        }
    }
    log_time("triangulation+warp_pose (parallel)");

    }  // End of if (!use_gt_pose_this_frame)

    result.z_tri = tri_result.z1_tri;
    result.num_valid_tri = tri_result.num_valid;

    // Save debug depth/pose fields
    if (config_.debug) {
        if (!z1_warp_pose.empty()) {
            result.z_warp_pose = z1_warp_pose.clone();
        }
        // Save forward pose only on iter 0 (iter > 0 overwrites result.R/t with backward)
        if (iter == 0) {
            result.R_forward = result.R;
            result.t_forward = result.t;
            if (!result.z_tri.empty()) {
                result.z_tri_forward = result.z_tri.clone();
            }
        }
    }

    // ========== Bayesian Fusion in S-space (like Python) ==========
    result.z_refined = cv::Mat(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    // Reset V_post for this iteration (declared outside loop)
    V_post.setTo(100.0f);

    constexpr bool USE_KALMAN_FUSION = true;

    // ===== PARALLEL: V_obs computation || Warp comparison (both independent) =====
    // V_obs, use_flow_warp declared before loop
    // When pose estimation is poor (median S-space error > 15%), fall back to flow warp
    use_flow_warp = false;

    if (!USE_KALMAN_FUSION || !has_prev_depth) {
        // First frame or fusion disabled: just compute V_obs, no warp comparison needed

        // Safety check for valid matrices
        if (tri_result.rpx_tri.empty() || result.z_tri.empty() ||
            tri_result.rpx_tri.rows != H || tri_result.rpx_tri.cols != W) {
            // Initialize with default values
            V_obs.setTo(100.0f);
            result.z_tri.copyTo(result.z_refined);
        } else {
            // Use rpx_to_variance_angle_aware for consistent variance computation
            // Paper formula: R = σ² × (r_ang / tau0)² where r_ang = rpx / f
            V_obs = rpx_to_variance_angle_aware(
                tri_result.rpx_tri,
                config_.fx, config_.fy,
                baseline,
                baseline_state_.b_ref(),
                var_cfg
            );
        }

        result.z_tri.copyTo(result.z_refined);
        V_obs.copyTo(V_post);
    } else {
        // Compute V_obs using rpx_to_variance_angle_aware
        // Paper formula: R = σ² × (r_ang / tau0)² where r_ang = rpx / f
        V_obs = rpx_to_variance_angle_aware(
            tri_result.rpx_tri,
            config_.fx, config_.fy,
            baseline,
            baseline_state_.b_ref(),
            var_cfg
        );

        // Warp comparison for consistency estimation
        {
            // ===== OPTIMIZED: Warp comparison with SAMPLING =====
            // Sample ~30K points instead of collecting all ~470K
            const int sample_step = 4;  // Sample every 4th pixel in each dimension
            std::vector<float> pct_errs;
            pct_errs.reserve(H * W / (sample_step * sample_step));

            for (int r = 0; r < H; r += sample_step) {
                const float* zp_row = z1_warp_pose.ptr<float>(r);
                const float* zf_row = z1_warp_flow.ptr<float>(r);
                const float* lam_row = inv_depth_f.ptr<float>(r);
                for (int c = 0; c < W; c += sample_step) {
                    float zp = zp_row[c];
                    float zf = zf_row[c];
                    float lam = lam_row[c];
                    if (std::isfinite(zp) && std::isfinite(zf) && std::isfinite(lam) && lam > 1e-8f) {
                        float sp = zp * lam;
                        float sf = zf * lam;
                        if (sp > 0 && sf > 0) {
                            float err = std::abs(sp - sf) / (sf + 1e-6f);
                            pct_errs.push_back(err);
                        }
                    }
                }
            }

            if (!pct_errs.empty()) {
                size_t mid = pct_errs.size() / 2;
                std::nth_element(pct_errs.begin(), pct_errs.begin() + mid, pct_errs.end());
                float med_err = pct_errs[mid];
                if (med_err > 0.15f) {
                    use_flow_warp = true;
                }
            }
        }

        // Select which depth to use as prior for fusion
        // Prefer 3D pose warp, but fall back to flow warp when pose is unreliable
        cv::Mat z1_warp;
        if (!z1_warp_pose.empty() && !use_flow_warp) {
            z1_warp = z1_warp_pose;
            if (!V_warp_pose.empty()) {
                V_prior = V_warp_pose;
            }
        } else if (!z1_warp_flow.empty()) {
            // Flow warp: pose estimation unreliable or pose warp unavailable
            z1_warp = z1_warp_flow;
            if (!V_warp_flow.empty()) {
                V_prior = V_warp_flow;
            }
        }

        // Debug: check warp statistics
        if (config_.debug) {
            int valid_warp = 0, valid_tri = 0;
            double sum_warp = 0, sum_tri = 0;
            for (int r = 0; r < H; ++r) {
                const float* zw = z1_warp.ptr<float>(r);
                const float* zt = result.z_tri.ptr<float>(r);
                for (int c = 0; c < W; ++c) {
                    if (std::isfinite(zw[c]) && zw[c] > 0) { valid_warp++; sum_warp += zw[c]; }
                    if (std::isfinite(zt[c]) && zt[c] > 0) { valid_tri++; sum_tri += zt[c]; }
                }
            }
            std::cerr << "[Fusion] z1_warp: valid=" << valid_warp
                      << ", mean=" << (valid_warp > 0 ? sum_warp/valid_warp : 0) << std::endl;
            std::cerr << "[Fusion] z_tri: valid=" << valid_tri
                      << ", mean=" << (valid_tri > 0 ? sum_tri/valid_tri : 0) << std::endl;

            // Debug: check prev_depth statistics
            int valid_prev = 0;
            double sum_prev = 0;
            for (int r = 0; r < H; ++r) {
                const float* pd = prev_depth_.ptr<float>(r);
                for (int c = 0; c < W; ++c) {
                    if (std::isfinite(pd[c]) && pd[c] > 0) { valid_prev++; sum_prev += pd[c]; }
                }
            }
            std::cerr << "[Fusion] prev_depth: valid=" << valid_prev
                      << ", mean=" << (valid_prev > 0 ? sum_prev/valid_prev : 0) << std::endl;
            std::cerr << "[Fusion] baseline=" << baseline << ", t_norm=" << t_for_tri.norm() << std::endl;
        }

        // ===== S-space fusion (like Python) =====
        // S = z * inv_depth (scale-normalized space)

        // ===== Paper Eq. 9: Adaptive variance inflation =====
        // V_prior *= (1 + (r_ang / tau0)^2)
        // r_ang = median(rpx) / f_eff  (frame-level reprojection angle)
        // When rotation is large, rpx is large → variance inflates → K increases
        // → triangulation gets more weight in fusion
        if (!tri_result.rpx_tri.empty()) {
            // Collect valid rpx values (sample every 4th pixel for speed)
            std::vector<float> rpx_vals;
            rpx_vals.reserve(H * W / 16);
            for (int r = 0; r < H; r += 4) {
                const float* rpx_row = tri_result.rpx_tri.ptr<float>(r);
                for (int c = 0; c < W; c += 4) {
                    float rpx = rpx_row[c];
                    if (std::isfinite(rpx) && rpx > 0) {
                        rpx_vals.push_back(rpx);
                    }
                }
            }

            if (rpx_vals.size() > 10) {
                // Compute median rpx
                size_t mid = rpx_vals.size() / 2;
                std::nth_element(rpx_vals.begin(), rpx_vals.begin() + mid, rpx_vals.end());
                float median_rpx = rpx_vals[mid];

                // Convert to angle: r_ang = rpx / f_eff
                float f_eff = std::sqrt(config_.fx * config_.fy);
                float r_ang = median_rpx / f_eff;

                // Reference angle tau0 (same as variance computation)
                float tau0_deg = config_.fusion_tau0_deg;
                if (tau0_deg < 0) tau0_deg = 0.1f;  // Default
                float tau0_ang = tau0_deg * static_cast<float>(M_PI) / 180.0f;

                // Eq. 9: inflation = 1 + (r_ang / tau0)^2
                float ratio = r_ang / tau0_ang;
                float inflation = 1.0f + ratio * ratio;

                // Apply to V_prior (all pixels)
                if (inflation > 1.001f) {
                    V_prior *= inflation;

                    if (config_.debug) {
                        std::cerr << "[Eq9] median_rpx=" << median_rpx
                                  << ", r_ang_deg=" << (r_ang * 180.0f / M_PI)
                                  << ", inflation=" << inflation << std::endl;
                    }
                }
            }
        }

        // ===== OPTIMIZED: Kappa estimation with SAMPLING =====
        // Sample every 4th pixel instead of collecting all ~470K values
        const int kappa_sample_step = 4;
        std::vector<float> rel_diff_vals;
        rel_diff_vals.reserve(H * W / (kappa_sample_step * kappa_sample_step));

        for (int r = 0; r < H; r += kappa_sample_step) {
            const float* z_warp_row = z1_warp.ptr<float>(r);
            const float* z_tri_row = result.z_tri.ptr<float>(r);
            const float* lam_row = inv_depth_f.ptr<float>(r);
            for (int c = 0; c < W; c += kappa_sample_step) {
                float lam = lam_row[c];
                if (std::isfinite(lam) && lam > 1e-8f) {
                    float zw = z_warp_row[c];
                    float zt = z_tri_row[c];
                    if (std::isfinite(zw) && zw > 1e-8f && std::isfinite(zt) && zt > 1e-8f) {
                        float sp = zw * lam;
                        float so = zt * lam;
                        float rd = std::abs(sp - so) / std::max(so, 1e-6f);
                        rel_diff_vals.push_back(rd);
                    }
                }
            }
        }

        // Update kappa using MAD of relative differences
        float kappa_raw = kappa_state_;
        if (!rel_diff_vals.empty()) {
            size_t n = rel_diff_vals.size();
            size_t mid = n / 2;
            std::nth_element(rel_diff_vals.begin(), rel_diff_vals.begin() + mid, rel_diff_vals.end());
            float median = rel_diff_vals[mid];

            for (auto& v : rel_diff_vals) {
                v = std::abs(v - median);
            }
            std::nth_element(rel_diff_vals.begin(), rel_diff_vals.begin() + mid, rel_diff_vals.end());
            float mad = rel_diff_vals[mid];

            kappa_raw = std::max(1.4826f * mad * fuse_cfg.diff_mad_mul, fuse_cfg.kappa_min);
        }

        // EMA update for kappa (baseline-proportional)
        float kappa_beta_eff = std::min(fuse_cfg.kappa_ema_beta * baseline_ratio, 1.0f);
        kappa_state_ = (1.0f - kappa_beta_eff) * kappa_state_ +
                       kappa_beta_eff * kappa_raw;
        float kappa_use = std::clamp(kappa_state_, fuse_cfg.kappa_min, fuse_cfg.kappa_max);

        // ===== Scale Jump Detection =====
        // Detect sudden global scale changes (e.g., rotation causing triangulation failure)
        // Compare current triangulation's global scale with EMA of previous scales
        bool scale_jump_rejected = false;
        float curr_global_scale = 1.0f;

        if (fuse_cfg.scale_jump_reject_enable) {
            // Compute current frame's global scale: median(z_tri / z_rel)
            // where z_rel = 1 / inv_depth
            std::vector<float> scale_ratios;
            scale_ratios.reserve(H * W / 16);  // Sample every 4th pixel

            for (int r = 0; r < H; r += 4) {
                const float* z_tri_row = result.z_tri.ptr<float>(r);
                const float* lam_row = inv_depth_f.ptr<float>(r);
                for (int c = 0; c < W; c += 4) {
                    float zt = z_tri_row[c];
                    float lam = lam_row[c];
                    // z_rel = 1/lam, so scale = z_tri * lam
                    if (std::isfinite(zt) && zt > 0.5f && zt < config_.max_depth &&
                        std::isfinite(lam) && lam > 1e-8f) {
                        scale_ratios.push_back(zt * lam);
                    }
                }
            }

            if (scale_ratios.size() > 100) {
                // Compute median scale
                size_t mid = scale_ratios.size() / 2;
                std::nth_element(scale_ratios.begin(), scale_ratios.begin() + mid, scale_ratios.end());
                curr_global_scale = scale_ratios[mid];

                if (scale_state_initialized_) {
                    // Check for scale jump
                    float scale_ratio = curr_global_scale / scale_state_;
                    float scale_change = std::abs(scale_ratio - 1.0f);

                    if (scale_change > fuse_cfg.scale_jump_threshold) {
                        scale_jump_rejected = true;
                        if (config_.debug) {
                            std::cerr << "[ScaleJump] REJECTED: curr=" << curr_global_scale
                                      << ", prev=" << scale_state_
                                      << ", ratio=" << scale_ratio
                                      << ", change=" << scale_change
                                      << " > threshold=" << fuse_cfg.scale_jump_threshold << std::endl;
                        }
                    } else {
                        // Update EMA only if not rejected
                        // Asymmetric EMA: fast recovery (low beta), slow corruption (high beta)
                        // Baseline-proportional: scale beta by motion amount
                        float ema_beta;
                        if (curr_global_scale > scale_state_) {
                            // Scale increasing (recovery): use lower beta for faster update
                            ema_beta = std::min(0.5f * baseline_ratio, 1.0f);
                        } else {
                            // Scale decreasing (potential corruption): use higher beta for slower update
                            ema_beta = std::min(fuse_cfg.scale_ema_beta * baseline_ratio, 1.0f);
                        }
                        scale_state_ = ema_beta * scale_state_ +
                                      (1.0f - ema_beta) * curr_global_scale;
                    }
                } else {
                    // First valid frame: initialize
                    scale_state_ = curr_global_scale;
                    scale_state_initialized_ = true;
                    if (config_.debug) {
                        std::cerr << "[ScaleJump] Initialized scale_state=" << scale_state_ << std::endl;
                    }
                }
            }
        }

        // ===== Frame Reject Check (m2-based) =====
        // Count pixels where m2 > chi2_hard to decide if frame should be rejected
        bool frame_rejected = false;
        if (fuse_cfg.frame_reject_enable) {
            int n_valid = 0;
            int n_bad = 0;
            const int fr_step = 4;  // Sample every 4th pixel (1/16 of total)

            for (int r = 0; r < H; r += fr_step) {
                const float* z_warp_row = z1_warp.ptr<float>(r);
                const float* z_tri_row = result.z_tri.ptr<float>(r);
                const float* vp_row = V_prior.ptr<float>(r);
                const float* vo_row = V_obs.ptr<float>(r);
                const float* lam_row = inv_depth_f.ptr<float>(r);

                for (int c = 0; c < W; c += fr_step) {
                    float lam = lam_row[c];
                    float vp = vp_row[c];
                    float vo = vo_row[c];

                    float sp = -1.0f, so = -1.0f;
                    if (std::isfinite(lam) && lam > 1e-8f) {
                        float zw = z_warp_row[c];
                        float zt = z_tri_row[c];
                        if (std::isfinite(zw) && zw > 1e-8f) sp = zw * lam;
                        if (std::isfinite(zt) && zt > 1e-8f) so = zt * lam;
                    }

                    if (sp > 0 && so > 0) {
                        bool vp_valid = std::isfinite(vp) && vp > 0;
                        bool vo_valid = std::isfinite(vo) && vo > 0;

                        float V_prior = std::max(vp_valid ? vp : 10.0f, fuse_cfg.min_var);
                        float V_obs = std::max(vo, fuse_cfg.min_var);

                        float innov = so - sp;
                        float V_innov = V_prior + V_obs;
                        float m2 = (innov * innov) / V_innov;

                        float C = 0.0f;
                        float rd = std::abs(sp - so) / std::max(so, 1e-6f);
                        float x = rd / kappa_use;
                        float x2 = x * x;
                        if (x2 < 9.0f) {
                            float x4 = x2 * x2;
                            C = 1.0f / (1.0f + x2 + 0.5f * x4 + x4 * x2 / 6.0f);
                        }

                        float chi2_hard_eff = fuse_cfg.chi2_hard * (1.0f + 0.7f * fuse_cfg.gate_loosen * (1.0f - C));

                        n_valid++;
                        if (m2 > chi2_hard_eff) {
                            n_bad++;
                        }
                    }
                }
            }

            // Frame reject decision (threshold scaled for sampling)
            if (n_valid >= fuse_cfg.frame_reject_min_valid / (fr_step * fr_step)) {
                float bad_frac = static_cast<float>(n_bad) / n_valid;
                if (bad_frac >= fuse_cfg.frame_reject_bad_frac) {
                    frame_rejected = true;
                    if (config_.debug) {
                        std::cerr << "[FrameReject] bad_frac=" << bad_frac
                                  << " >= " << fuse_cfg.frame_reject_bad_frac
                                  << " (n_bad=" << n_bad << ", n_valid=" << n_valid << ")"
                                  << std::endl;
                    }
                }
            }
        }

        // Combine frame rejection from m2-based check and scale jump detection
        frame_rejected = frame_rejected || scale_jump_rejected;

        // If frame rejected, use prior depth and skip fusion
        if (frame_rejected) {
            // Keep R and t from forward motion (even if inaccurate, preserve for trajectory)
            // result.R and result.t already set from forward motion
            // Don't zero out t - preserve forward motion's t

            // Use warped prior as refined depth
            z1_warp.copyTo(result.z_refined);

            // Increase variance
            cv::Mat V_increased;
            V_prior.copyTo(V_increased);
            V_increased *= 1.5f;

            // Compute confidence
            result.confidence = variance_to_confidence(V_increased);

            update_state(img_resized, inv_depth_f, result.z_refined, V_increased);

            if (config_.debug) {
                std::cerr << "[FrameReject] Using prior depth, R preserved, t=0" << std::endl;
            }

            frame_count_++;
            return result;
        }

        // Pass 2: Compute S values + Kalman update in one loop (no temp matrices)
        for (int r = 0; r < H; ++r) {
            const float* z_warp_row = z1_warp.ptr<float>(r);
            const float* z_tri_row = result.z_tri.ptr<float>(r);
            const float* vp_row = V_prior.ptr<float>(r);
            const float* vo_row = V_obs.ptr<float>(r);
            const float* lam_row = inv_depth_f.ptr<float>(r);
            float* z_ref_row = result.z_refined.ptr<float>(r);
            float* v_post_row = V_post.ptr<float>(r);

            for (int c = 0; c < W; ++c) {
                float lam = lam_row[c];
                float vp = vp_row[c];
                float vo = vo_row[c];

                // Compute S values on-the-fly (no temp storage)
                float sp = -1.0f, so = -1.0f;
                if (std::isfinite(lam) && lam > 1e-8f) {
                    float zw = z_warp_row[c];
                    float zt = z_tri_row[c];
                    if (std::isfinite(zw) && zw > 1e-8f) sp = zw * lam;
                    if (std::isfinite(zt) && zt > 1e-8f) so = zt * lam;
                }

                // Compute consistency C with fast Gaussian approximation
                float C = 0.0f;
                if (sp > 0 && so > 1e-8f) {
                    float rd = std::abs(sp - so) / std::max(so, 1e-6f);
                    float x = rd / kappa_use;
                    float x2 = x * x;
                    // Fast exp(-x^2): Pade approximation accurate to <1% for |x|<3
                    // exp(-x^2) ≈ 1 / (1 + x^2 + 0.5*x^4 + x^6/6)
                    if (x2 < 9.0f) {
                        float x4 = x2 * x2;
                        C = 1.0f / (1.0f + x2 + 0.5f * x4 + x4 * x2 / 6.0f);
                    }
                }

                // Use > 0 check since S_prior/S_obs initialized to -1
                bool sp_valid = sp > 0;
                bool so_valid = so > 0;
                bool vp_valid = std::isfinite(vp) && vp > 0;
                bool vo_valid = std::isfinite(vo) && vo > 0;
                bool lam_valid = std::isfinite(lam) && lam > 1e-8f;

                if (!sp_valid && !so_valid) {
                    // Neither valid
                    z_ref_row[c] = std::numeric_limits<float>::quiet_NaN();
                    v_post_row[c] = 100.0f;
                } else if (!sp_valid) {
                    // Only observation valid
                    v_post_row[c] = vo_valid ? vo : 100.0f;
                    z_ref_row[c] = lam_valid ? (so / lam) : std::numeric_limits<float>::quiet_NaN();
                } else if (!so_valid) {
                    // Only prior valid - pass through with warp-inflated variance
                    // (warp already applied V *= (1 + lambda_forget) + min_var)
                    v_post_row[c] = std::max(vp_valid ? vp : 10.0f, fuse_cfg.min_var);
                    z_ref_row[c] = lam_valid ? (sp / lam) : std::numeric_limits<float>::quiet_NaN();
                } else {
                    // Both valid - Do Kalman fusion in scale-space
                    // Variable naming convention (matches paper):
                    //   scale = z * lambda (state variable in scale-space)
                    //   V = variance of scale
                    //   sp = prior scale (from warped previous depth)
                    //   so = observed scale (from triangulation)
                    //   V_prior, V_obs = variances
                    //   V_innov = innovation variance = V_prior + V_obs

                    float V_prior = std::max(vp_valid ? vp : 10.0f, fuse_cfg.min_var);
                    float V_obs = std::max(vo, fuse_cfg.min_var);

                    // Innovation (difference) and Mahalanobis distance
                    float innov = so - sp;  // scale_obs - scale_prior
                    float V_innov = V_prior + V_obs;  // innovation variance
                    float m2 = (innov * innov) / V_innov;

                    // Chi2 gates with loosening based on consistency
                    float chi2_hard_eff = fuse_cfg.chi2_hard * (1.0f + 0.7f * fuse_cfg.gate_loosen * (1.0f - C));

                    // K_cap based on pixel-level consistency C
                    // K_cap = kcap_floor + (1 - kcap_floor) * C  (per-pixel)
                    float K_cap = fuse_cfg.kcap_floor + (1.0f - fuse_cfg.kcap_floor) * C;

                    // Kalman gain: K = V_prior / V_innov
                    float K_raw = V_prior / V_innov;
                    float K_eff = std::min(K_raw, K_cap);

                    float scale_post, V_post;

                    // Chi2 hard gate
                    if (m2 > chi2_hard_eff) {
                        // Hard rejection - choose based on variance
                        if (V_prior < V_obs) {
                            scale_post = sp;
                            V_post = V_prior;
                        } else {
                            scale_post = so;
                            V_post = V_obs;
                        }
                    } else {
                        // Normal Kalman update: scale_post = scale_prior + K * innov
                        scale_post = sp + K_eff * innov;
                        V_post = std::max((1.0f - K_eff) * (1.0f - K_eff) * V_prior + K_eff * K_eff * V_obs,
                                           fuse_cfg.min_var);
                    }

                    v_post_row[c] = V_post;

                    // Convert scale back to z: z = scale / inv_depth
                    if (lam_valid) {
                        float z_result = scale_post / lam;
                        // Clamp to max depth
                        z_ref_row[c] = std::min(z_result, config_.max_depth);
                    } else {
                        z_ref_row[c] = std::numeric_limits<float>::quiet_NaN();
                    }
                }
            }
        }

        // Debug: count valid z_refined after fusion
        if (config_.debug) {
            int valid_refined = 0, valid_warp = 0, valid_tri = 0;
            for (int r = 0; r < H; ++r) {
                const float* zr = result.z_refined.ptr<float>(r);
                const float* zw = z1_warp.ptr<float>(r);
                const float* zt = result.z_tri.ptr<float>(r);
                for (int c = 0; c < W; ++c) {
                    if (std::isfinite(zr[c]) && zr[c] > 0) valid_refined++;
                    if (std::isfinite(zw[c]) && zw[c] > 0) valid_warp++;
                    if (std::isfinite(zt[c]) && zt[c] > 0) valid_tri++;
                }
            }
            std::cerr << "[Fusion] After fusion: z_refined valid=" << valid_refined
                      << ", z_warp valid=" << valid_warp
                      << ", z_tri valid=" << valid_tri << std::endl;
        }

        if (config_.debug) {
            if (!V_prior.empty()) {
                result.V_prior = V_prior.clone();
            }
            if (!V_post.empty()) {
                result.V_post = V_post.clone();
            }
        }

        log_time("kalman_fusion");
    }

    // ===== Apply metric scale estimation (optional) =====
    // NOTE: This runs for BOTH cases (first frame / skip_temporal_fusion AND kalman fusion)
    // metric_scale_mode: 0=off, 1=global, 2=per-segment
    // Legacy: use_metric_scale=true -> mode 2, false -> mode 0
    int effective_mode = config_.metric_scale_mode;
    if (effective_mode == 2 && !config_.use_metric_scale) {
        effective_mode = 0;  // Legacy override
    }

    if (effective_mode > 0) {
        // Compute rel_depth and valid_mask in one pass
            cv::Mat rel_depth(H, W, CV_32F);
            cv::Mat valid_mask(H, W, CV_8U);
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < H; ++r) {
                const float* lam_row = inv_depth_f.ptr<float>(r);
                const float* zr = result.z_refined.ptr<float>(r);
                float* rd_row = rel_depth.ptr<float>(r);
                uint8_t* m_row = valid_mask.ptr<uint8_t>(r);
                for (int c = 0; c < W; ++c) {
                    float lam = lam_row[c];
                    if (std::isfinite(lam) && lam > 1e-8f) {
                        rd_row[c] = 1.0f / lam;
                    } else {
                        rd_row[c] = std::numeric_limits<float>::quiet_NaN();
                    }
                    m_row[c] = (std::isfinite(zr[c]) && zr[c] > 0) ? 1 : 0;
                }
            }

            // Save sparse fusion result before solve_metric_from_rel (debug only)
            if (config_.debug) {
                z_fused_sparse = result.z_refined.clone();
            }

            // Apply metric scale estimation
            MetricScaleConfig metric_cfg;
            metric_cfg.min_pts_per_label = 10;
            metric_cfg.min_pts_ratio = 0.001f;
            metric_cfg.single_med_thr = 0.12f;
            metric_cfg.single_p90_thr = 0.25f;
            metric_cfg.var_floor = fuse_cfg.min_var;
            metric_cfg.var_cap = 100.0f;

            // mode 1: global scale (empty LabelIndex)
            // mode 2: per-segment scale (use pre-built label_index)
            const LabelIndex& li_for_metric = (effective_mode == 1) ? LabelIndex() : label_index;

            MetricScaleResult metric_result = solve_metric_from_rel(
                rel_depth, result.z_refined, valid_mask,
                li_for_metric, V_post, metric_cfg
            );

            if (config_.debug) {
                std::cerr << "[Fusion] metric_scale_mode=" << effective_mode
                          << ", global_scale=" << metric_result.global_scale << std::endl;
            }

            // Use scale-corrected depth
            result.z_refined = metric_result.z_out;
            V_post = metric_result.V_out;
            log_time("solve_metric_from_rel");

            if (config_.debug) {
                std::cerr << "[Fusion] After solve_metric_from_rel: global_scale="
                          << metric_result.global_scale << std::endl;
            }
        }

    // ======== End of iteration ========
    if (config_.debug && iter < total_iters - 1) {
        std::cerr << "[IterRefine] Iteration " << iter << " complete" << std::endl;
    }

    // Save iteration debug info
    if (config_.debug) {
        IterationDebugInfo iter_info;
        iter_info.iter = iter;
        iter_info.is_backward = use_backward_estimation;
        iter_info.R = R_for_tri;
        iter_info.t = t_for_tri;
        iter_info.num_inliers = result.num_matches;
        iter_info.num_valid_tri = tri_result.num_valid;
        iter_info.metric_scale = 1.0f;

        if (!tri_result.z1_tri.empty()) {
            iter_info.z_tri = tri_result.z1_tri.clone();
        }
        if (!z_fused_sparse.empty()) {
            iter_info.z_fused_sparse = z_fused_sparse.clone();
        }
        if (!result.z_refined.empty()) {
            iter_info.z_refined = result.z_refined.clone();
        }

        result.iteration_info.push_back(std::move(iter_info));
    }

    }  // End of iteration loop

    // Compute confidence from variance
    result.confidence = variance_to_confidence(V_post);

    // Store optical flow for visualization
    result.flow = flow;

    // Update state for next frame
    if (!config_.skip_temporal_fusion) {
        // Full update: save everything for temporal fusion
        update_state(img_resized, inv_depth_f, result.z_refined, V_post);
    } else {
        // skip_temporal_fusion mode: still need prev_img_ and prev_inv_depth_ for
        // optical flow and motion estimation on next frame, but skip depth/variance
        img_resized.copyTo(prev_img_);
        inv_depth_f.copyTo(prev_inv_depth_);
        // Don't update prev_depth_, prev_V_ - each frame is independent
    }
    frame_count_++;

    if (config_.timing) {
        auto t_end = Clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        PR_TIMING_LOG("[Timing] TOTAL: %.1f ms", total_ms);
    }

    return result;
}

}  // namespace pr_depth
