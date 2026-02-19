#pragma once
#include <opencv2/core.hpp>
#include <cmath>
#include <algorithm>
#include <vector>

namespace pr_depth {

/**
 * LabelIndex: Cache structure for efficient per-label pixel access.
 *
 * Instead of std::vector<std::vector<int>> (poor cache locality),
 * uses contiguous memory with offset-based indexing:
 *   - flat_pixels: all pixel indices sorted by label
 *   - offsets[L]: start index for label L in flat_pixels
 *   - Label L's pixels: flat_pixels[offsets[L]..offsets[L+1])
 *
 * Benefits:
 *   - Single contiguous allocation (cache-friendly)
 *   - O(1) access to any label's pixel list
 *   - No per-label dynamic allocation overhead
 */
struct LabelIndex {
    int num_labels = 0;
    std::vector<int> offsets;      // size: num_labels+1
    std::vector<int> flat_pixels;  // size: total_pixels (H*W for dense)

    // Get pixel indices for label L
    inline const int* begin(int L) const { return flat_pixels.data() + offsets[L]; }
    inline const int* end(int L) const { return flat_pixels.data() + offsets[L + 1]; }
    inline int count(int L) const { return offsets[L + 1] - offsets[L]; }
    inline bool empty() const { return num_labels == 0; }
};

/**
 * Build LabelIndex from a label map.
 *
 * @param labels Label map (HxW, CV_32S)
 * @param num_labels Number of labels (if known, pass 0 to auto-detect)
 * @return LabelIndex with contiguous pixel storage
 */
LabelIndex build_label_index(const cv::Mat& labels, int num_labels = 0);

// Configuration for depth fusion operations
struct DepthFusionConfig {
    // rpx_to_variance parameters
    float tau0_deg = -1.0f;         // Reference angle in degrees (-1 = use default 0.1 deg)
    float sigma2_at_tau = 1.0f;     // Variance scale at reference angle
    float var_floor = 1e-3f;        // Minimum variance (prevents overconfidence)
    float var_cap = 1e+3f;          // Maximum variance (prevents numerical issues)
};

/**
 * Convert Sampson error (pixel-based reprojection error) to observation variance.
 *
 * Paper: Section 3.4 - Temporal Fusion
 * The observation variance σ²_obs is derived from Sampson error, which measures
 * geometric consistency of point correspondences. Lower error = higher confidence.
 *
 * Formula:
 *   rpx_ang = rpx / f_eff                    [convert to radians]
 *   norm_err = rpx_ang / tau0_eff            [normalize by reference angle]
 *   variance = norm_err² × sigma2_at_tau     [quadratic mapping]
 *
 * @param rpx        Sampson error map (HxW, float32)
 * @param fx, fy     Camera focal lengths
 * @param baseline   Current frame displacement (translation magnitude)
 * @param b_ref_auto Reference baseline for auto-scaling
 * @param config     Fusion configuration
 * @return Variance map (HxW, float32)
 */
cv::Mat rpx_to_variance_angle_aware(
    const cv::Mat& rpx,
    float fx, float fy,
    float baseline,
    float b_ref_auto,
    const DepthFusionConfig& config = DepthFusionConfig()
);

// Configuration for robust scale matching
struct RobustScaleConfig {
    float max_depth = 80.0f;       // Maximum valid depth
    int min_overlap = 2000;        // Minimum overlapping pixels
    float tol_median = 0.3f;       // Median relative error tolerance
};

// Result of robust scale matching
struct RobustScaleResult {
    float scale;                   // Estimated scale factor
    bool scale_ok;                 // Whether scale is reliable
    int overlap;                   // Number of pixels used
    float median_relerr;           // Median relative error
};

/**
 * Robustly estimate scale between triangulated and reference depth.
 *
 * Finds scale s such that z_ref ≈ s * z_tri using:
 * 1. Median of ratios (initial estimate)
 * 2. MAD-based outlier rejection
 * 3. Quality check via median relative error
 *
 * @param z_tri Triangulated depth (HxW, float32)
 * @param z_ref Reference depth (HxW, float32)
 * @param mask Valid pixel mask (HxW, uint8, optional)
 * @param config Configuration parameters
 * @return Scale factor and diagnostics
 */
RobustScaleResult robust_scale_match(
    const cv::Mat& z_tri,
    const cv::Mat& z_ref,
    const cv::Mat& mask = cv::Mat(),
    const RobustScaleConfig& config = RobustScaleConfig()
);

/**
 * Fast map warping using nearest-neighbor splatting.
 *
 * Given point correspondences (u0,v0) -> (u1,v1) and a scalar map M0,
 * splat values from source locations to target locations.
 *
 * @param u0, v0 Source coordinates (N elements)
 * @param u1, v1 Target coordinates (N elements)
 * @param M0 Source scalar map (HxW, float32)
 * @param fill Fill value for unmapped pixels (default NaN)
 * @return Warped map (HxW, float32)
 */
cv::Mat warp_map_fast(
    const std::vector<float>& u0, const std::vector<float>& v0,
    const std::vector<float>& u1, const std::vector<float>& v1,
    const cv::Mat& M0,
    float fill = std::numeric_limits<float>::quiet_NaN()
);

/**
 * Compute per-label median of a 2D map using histogram approximation.
 *
 * @param map2d Scalar map (HxW, float32)
 * @param labels Label map (HxW, int32)
 * @param min_pts Minimum points per label (default 20)
 * @return Vector of medians indexed by label
 */
std::vector<float> aggregate_label_median(
    const cv::Mat& map2d,
    const cv::Mat& labels,
    int min_pts = 20
);

/**
 * Baseline auto-state tracker using EMA.
 *
 * Tracks camera baseline magnitude over time with exponential moving average.
 * Used to set variance scaling for triangulation-based depth.
 */
class BaselineAutoState {
public:
    explicit BaselineAutoState(float ema_beta = 0.5f, int hist_len = 100);

    void update(float b);
    float b_ref() const;
    float guard_threshold() const;
    bool should_disable(float baseline, float extra_min = 0.0f) const;

private:
    float b_ema_;
    bool has_ema_;
    std::vector<float> hist_;  // Fixed-size ring buffer
    int hist_head_;            // Next write position
    int hist_count_;           // Current number of valid entries
    float beta_;
    int hist_len_;
};

// Configuration for solve_metric_from_rel
struct MetricScaleConfig {
    int min_pts_per_label = 10;     // Minimum absolute points per segment
    float min_pts_ratio = 0.001f;   // Minimum ratio: valid_pts / total_label_pts (0.1%)
    float single_med_thr = 0.12f;   // Single-scale acceptance threshold (median)
    float single_p90_thr = 0.25f;   // Single-scale acceptance threshold (p90)
    float global_trim_k = 4.5f;     // MAD-based outlier trimming factor
    float var_floor = 1e-3f;        // Minimum variance
    float var_cap = 1e+3f;          // Maximum variance
};

// Result of solve_metric_from_rel
struct MetricScaleResult {
    cv::Mat z_out;      // Metric depth (HxW, float32)
    cv::Mat S_map;      // Scale map (HxW, float32)
    cv::Mat V_out;      // Variance map (HxW, float32)
    float global_scale = 1.0f; // Global scale factor
    float global_med_rel = 1.0f; // Global median relative error (1.0 = invalid/skip)
    float global_p90_rel = 1.0f; // Global p90 relative error (1.0 = invalid/skip)
};

/**
 * Solve metric scale from relative depth (outdoor, single-scale mode).
 *
 * Estimates per-segment scale factors to convert relative depth to metric.
 * Uses robust median estimation with MAD-based outlier rejection.
 *
 * @param rel_depth Relative depth map (HxW, float32)
 * @param z_obs Observed metric depth from triangulation (HxW, float32)
 * @param mask Valid pixel mask (HxW, uint8, optional)
 * @param labels Segment labels (HxW, int32, optional)
 * @param v_px Previous variance map (HxW, float32, optional)
 * @param config Configuration parameters
 * @return Metric depth, scale map, variance map, and diagnostics
 */
MetricScaleResult solve_metric_from_rel(
    const cv::Mat& rel_depth,
    const cv::Mat& z_obs,
    const cv::Mat& mask = cv::Mat(),
    const cv::Mat& labels = cv::Mat(),
    const cv::Mat& v_px = cv::Mat(),
    const MetricScaleConfig& config = MetricScaleConfig()
);

/**
 * Solve metric scale using pre-built LabelIndex (faster version).
 *
 * Use this when LabelIndex is already available from segmentation.
 * Avoids redundant label map traversal.
 */
MetricScaleResult solve_metric_from_rel(
    const cv::Mat& rel_depth,
    const cv::Mat& z_obs,
    const cv::Mat& mask,
    const LabelIndex& label_index,
    const cv::Mat& v_px = cv::Mat(),
    const MetricScaleConfig& config = MetricScaleConfig()
);

}  // namespace pr_depth
