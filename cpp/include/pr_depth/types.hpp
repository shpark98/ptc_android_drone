#pragma once
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <string>
#include <stdexcept>
#include <cmath>  // FIX: for std::isfinite

namespace pr_depth {

// Camera intrinsics
struct CameraIntrinsics {
    int H, W;
    float fx, fy, cx, cy;

    CameraIntrinsics() : H(0), W(0), fx(0), fy(0), cx(0), cy(0) {}
    CameraIntrinsics(int h, int w, float fx_, float fy_, float cx_, float cy_)
        : H(h), W(w), fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}
};

// Minimal state for temporal fusion
// Paper: Section 3.4, Eq. (14) outputs
// Only {S_post, V_post} required; Z_post included only for warping
struct PrDepthState {
    cv::Mat S_post;      // Posterior scale field (HxW, float32)
    cv::Mat V_post;      // Posterior variance (HxW, float32)
    cv::Mat Z_post;      // Final depth (HxW, float32) - for warping only

    bool empty() const {
        return S_post.empty() || V_post.empty();
    }

    void clear() {
        S_post.release();
        V_post.release();
        Z_post.release();
    }
};

// Configuration for optical flow
// FIX: Use OpenCV enum directly, default to MEDIUM (value = 2)
struct OpticalFlowConfig {
    std::string method = "dis";         // "dis" only (RAFT not in scope)
    int preset = 2;                     // cv::DISOpticalFlow::PRESET_MEDIUM
    int finest_scale = 0;               // Downscale control (0 = no downscale)
    bool use_spatial_propagation = true;
};

// Utility: fail-fast on missing displacement
// Paper constraint: displacement ALWAYS provided externally
inline void require_displacement(float displacement) {
    if (!std::isfinite(displacement) || displacement <= 0.0f) {
        throw std::invalid_argument(
            "Displacement scalar must be provided externally and > 0. "
            "No fallback to GPS/odometry. Got: " + std::to_string(displacement)
        );
    }
}

} // namespace pr_depth
