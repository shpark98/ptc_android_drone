#include "pr_depth/depth_fusion.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>
#include <set>
#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace pr_depth {

// ============================================================================
// LabelIndex implementation
// ============================================================================

LabelIndex build_label_index(const cv::Mat& labels, int num_labels) {
    LabelIndex idx;

    if (labels.empty()) {
        return idx;
    }

    const int H = labels.rows;
    const int W = labels.cols;

    // Auto-detect num_labels if not provided
    if (num_labels <= 0) {
        int maxL = 0;
        for (int r = 0; r < H; ++r) {
            const int32_t* row = labels.ptr<int32_t>(r);
            for (int c = 0; c < W; ++c) {
                maxL = std::max(maxL, row[c]);
            }
        }
        num_labels = maxL + 1;
    }

    idx.num_labels = num_labels;
    idx.offsets.resize(num_labels + 1, 0);

    // Pass 1: Count pixels per label
    for (int r = 0; r < H; ++r) {
        const int32_t* row = labels.ptr<int32_t>(r);
        for (int c = 0; c < W; ++c) {
            int L = row[c];
            if (L >= 0 && L < num_labels) {
                idx.offsets[L + 1]++;
            }
        }
    }

    // Convert counts to cumulative offsets
    for (int L = 1; L <= num_labels; ++L) {
        idx.offsets[L] += idx.offsets[L - 1];
    }

    // Allocate flat array
    idx.flat_pixels.resize(idx.offsets[num_labels]);

    // Pass 2: Fill pixel indices (use write positions)
    std::vector<int> write_pos(num_labels);
    for (int L = 0; L < num_labels; ++L) {
        write_pos[L] = idx.offsets[L];
    }

    for (int r = 0; r < H; ++r) {
        const int32_t* row = labels.ptr<int32_t>(r);
        for (int c = 0; c < W; ++c) {
            int L = row[c];
            if (L >= 0 && L < num_labels) {
                idx.flat_pixels[write_pos[L]++] = r * W + c;
            }
        }
    }

    return idx;
}

// ============================================================================
// Helper functions
// ============================================================================

// Helper: compute median of a vector
static double compute_median(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    size_t n = v.size();
    size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    if (n % 2 == 0) {
        double a = v[mid];
        std::nth_element(v.begin(), v.begin() + mid - 1, v.end());
        return (a + v[mid - 1]) / 2.0;
    }
    return v[mid];
}

// Helper: compute percentile using nth_element (O(n) instead of O(n log n))
static double compute_percentile(std::vector<double>& v, double pctl) {
    if (v.empty()) return 0.0;
    double idx = (pctl / 100.0) * (v.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));
    if (lo >= v.size()) lo = v.size() - 1;
    if (hi >= v.size()) hi = v.size() - 1;

    // Use nth_element for O(n) complexity
    std::nth_element(v.begin(), v.begin() + lo, v.end());
    double val_lo = v[lo];

    if (lo == hi) return val_lo;

    // Get hi value (already partitioned around lo)
    std::nth_element(v.begin() + lo + 1, v.begin() + hi, v.end());
    double val_hi = v[hi];

    double frac = idx - lo;
    return val_lo * (1.0 - frac) + val_hi * frac;
}

cv::Mat rpx_to_variance_angle_aware(
    const cv::Mat& rpx,
    float fx, float fy,
    float baseline,
    float b_ref_auto,
    const DepthFusionConfig& config
) {
    // Input validation
    if (rpx.empty()) {
        return cv::Mat();
    }
    if (rpx.type() != CV_32F) {
        throw std::runtime_error("rpx must be CV_32F");
    }

    const int H = rpx.rows;
    const int W = rpx.cols;

    // Effective focal length (geometric mean)
    const float f_eff = std::sqrt(std::max(fx, 1e-6f) * std::max(fy, 1e-6f));

    // Reference angle tau0 (baseline-scaled)
    float tau0_ang_base;
    if (config.tau0_deg > 0) {
        tau0_ang_base = config.tau0_deg * static_cast<float>(M_PI) / 180.0f;
    } else {
        tau0_ang_base = 1.75e-3f;  // ~0.1 deg default
    }

    // Scale tau0 by baseline ratio
    float B = (std::isfinite(baseline) && baseline > 0) ? baseline : b_ref_auto;
    float bref = (std::isfinite(b_ref_auto) && b_ref_auto > 0) ? b_ref_auto : 1.0f;
    float tau0_ang_eff = tau0_ang_base * (B / bref);

    // Denominator for normalization (avoid division by zero)
    float denom = std::max(tau0_ang_eff, 1e-8f);

    // Output variance map
    cv::Mat variance(H, W, CV_32F);

    // Pre-compute constants
    const float inv_f_eff = 1.0f / f_eff;
    const float inv_denom = 1.0f / denom;
    const float scale = config.sigma2_at_tau * inv_denom * inv_denom * inv_f_eff * inv_f_eff;

    // Process each pixel with OpenMP parallelization
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < H; ++r) {
        const float* rpx_row = rpx.ptr<float>(r);
        float* var_row = variance.ptr<float>(r);

        for (int c = 0; c < W; ++c) {
            float rpx_val = std::max(rpx_row[c], 0.0f);
            // Combined: v = (rpx / f_eff / denom)^2 * sigma2_at_tau
            float v = rpx_val * rpx_val * scale;
            var_row[c] = std::max(config.var_floor, std::min(config.var_cap, v));
        }
    }

    return variance;
}

RobustScaleResult robust_scale_match(
    const cv::Mat& z_tri,
    const cv::Mat& z_ref,
    const cv::Mat& mask,
    const RobustScaleConfig& config
) {
    RobustScaleResult result;
    result.scale = 1.0f;
    result.scale_ok = false;
    result.overlap = 0;
    result.median_relerr = std::numeric_limits<float>::infinity();

    if (z_tri.empty() || z_ref.empty()) {
        return result;
    }

    const int H = z_tri.rows;
    const int W = z_tri.cols;
    const float max_depth = config.max_depth;

    // Build valid mask and collect valid pixels
    std::vector<double> x_vals, y_vals;
    x_vals.reserve(H * W / 4);
    y_vals.reserve(H * W / 4);

    bool has_mask = !mask.empty();

    for (int r = 0; r < H; ++r) {
        const float* tri_row = z_tri.ptr<float>(r);
        const float* ref_row = z_ref.ptr<float>(r);
        const uint8_t* mask_row = has_mask ? mask.ptr<uint8_t>(r) : nullptr;

        for (int c = 0; c < W; ++c) {
            float t = tri_row[c];
            float f = ref_row[c];

            // Check validity
            if (!std::isfinite(t) || !std::isfinite(f)) continue;
            if (t <= 0 || f <= 0) continue;
            if (t > max_depth || f > max_depth) continue;
            if (has_mask && mask_row[c] == 0) continue;

            x_vals.push_back(static_cast<double>(t));
            y_vals.push_back(static_cast<double>(f));
        }
    }

    result.overlap = static_cast<int>(x_vals.size());

    if (result.overlap < config.min_overlap) {
        return result;
    }

    // Step 1: Compute ratios y/x
    std::vector<double> ratios(x_vals.size());
    for (size_t i = 0; i < x_vals.size(); ++i) {
        ratios[i] = y_vals[i] / std::max(x_vals[i], 1e-12);
    }

    // Percentile cutoff (1-99%)
    std::vector<double> ratios_copy = ratios;
    double lo = compute_percentile(ratios_copy, 1.0);

    // Filter to good ratios
    std::vector<double> good_ratios;
    std::vector<double> good_x, good_y;
    for (size_t i = 0; i < ratios.size(); ++i) {
        if (ratios[i] >= lo) {
            good_ratios.push_back(ratios[i]);
            good_x.push_back(x_vals[i]);
            good_y.push_back(y_vals[i]);
        }
    }

    if (static_cast<int>(good_ratios.size()) < config.min_overlap) {
        return result;
    }

    // Initial scale estimate (median of ratios)
    double s0 = compute_median(good_ratios);

    // Step 2: MAD-based refinement
    std::vector<double> errors(good_x.size());
    for (size_t i = 0; i < good_x.size(); ++i) {
        double predicted = s0 * good_x[i];
        double err = std::abs(good_y[i] - predicted) / std::max(good_y[i], 1e-6);
        errors[i] = err;
    }

    double med_err = compute_median(errors);
    std::vector<double> err_dev(errors.size());
    for (size_t i = 0; i < errors.size(); ++i) {
        err_dev[i] = std::abs(errors[i] - med_err);
    }
    double mad = compute_median(err_dev) + 1e-12;
    double thr = 3.5 * mad;

    // Collect inliers
    std::vector<double> inlier_ratios;
    for (size_t i = 0; i < errors.size(); ++i) {
        if (errors[i] <= thr) {
            inlier_ratios.push_back(good_y[i] / std::max(good_x[i], 1e-12));
        }
    }

    double s;
    if (static_cast<int>(inlier_ratios.size()) >= std::max(config.min_overlap,
            static_cast<int>(0.5 * good_ratios.size()))) {
        s = compute_median(inlier_ratios);
    } else {
        s = s0;
    }

    // Step 3: Quality evaluation
    std::vector<double> rel_errors;
    for (size_t i = 0; i < x_vals.size(); ++i) {
        double z_tri_s = s * x_vals[i];
        double rel = std::abs(y_vals[i] - z_tri_s) / std::max(y_vals[i], 1e-6);
        rel_errors.push_back(rel);
    }

    double med_rel = compute_median(rel_errors);

    result.scale = static_cast<float>(s);
    result.scale_ok = (med_rel <= config.tol_median);
    result.median_relerr = static_cast<float>(med_rel);

    return result;
}

cv::Mat warp_map_fast(
    const std::vector<float>& u0, const std::vector<float>& v0,
    const std::vector<float>& u1, const std::vector<float>& v1,
    const cv::Mat& M0,
    float fill
) {
    if (M0.empty()) {
        return cv::Mat();
    }

    const int H = M0.rows;
    const int W = M0.cols;
    const int N = static_cast<int>(u0.size());

    // Initialize output with fill value
    cv::Mat out(H, W, CV_32F, cv::Scalar(fill));

    for (int i = 0; i < N; ++i) {
        // Round source coordinates
        int ui0 = static_cast<int>(std::round(u0[i]));
        int vi0 = static_cast<int>(std::round(v0[i]));

        // Check source bounds
        if (ui0 < 0 || ui0 >= W || vi0 < 0 || vi0 >= H) continue;

        // Get value from source map
        float val = M0.ptr<float>(vi0)[ui0];
        if (!std::isfinite(val)) continue;

        // Round target coordinates
        int u1r = static_cast<int>(std::round(u1[i]));
        int v1r = static_cast<int>(std::round(v1[i]));

        // Check target bounds
        if (u1r < 0 || u1r >= W || v1r < 0 || v1r >= H) continue;

        // Splat value
        out.ptr<float>(v1r)[u1r] = val;
    }

    return out;
}

std::vector<float> aggregate_label_median(
    const cv::Mat& map2d,
    const cv::Mat& labels,
    int min_pts
) {
    if (map2d.empty() || labels.empty()) {
        return {};
    }

    const int H = map2d.rows;
    const int W = map2d.cols;

    // Find max label
    int maxL = 0;
    for (int r = 0; r < H; ++r) {
        const int32_t* lbl_row = labels.ptr<int32_t>(r);
        for (int c = 0; c < W; ++c) {
            maxL = std::max(maxL, lbl_row[c]);
        }
    }
    maxL += 1;  // Number of labels

    // Collect valid values per label
    std::vector<std::vector<float>> label_vals(maxL);

    for (int r = 0; r < H; ++r) {
        const float* map_row = map2d.ptr<float>(r);
        const int32_t* lbl_row = labels.ptr<int32_t>(r);
        for (int c = 0; c < W; ++c) {
            float v = map_row[c];
            if (std::isfinite(v)) {
                int L = lbl_row[c];
                if (L >= 0 && L < maxL) {
                    label_vals[L].push_back(v);
                }
            }
        }
    }

    // Compute median per label
    std::vector<float> result(maxL, 0.0f);
    for (int i = 0; i < maxL; ++i) {
        if (static_cast<int>(label_vals[i].size()) >= min_pts) {
            // Convert to double for median computation
            std::vector<double> vals_d(label_vals[i].begin(), label_vals[i].end());
            result[i] = static_cast<float>(compute_median(vals_d));
        }
    }

    return result;
}

// BaselineAutoState implementation
BaselineAutoState::BaselineAutoState(float ema_beta, int hist_len)
    : b_ema_(0.0f), has_ema_(false), hist_(hist_len), hist_head_(0), hist_count_(0),
      beta_(ema_beta), hist_len_(hist_len) {}

void BaselineAutoState::update(float b) {
    if (!std::isfinite(b)) return;

    if (!has_ema_) {
        b_ema_ = b;
        has_ema_ = true;
    } else {
        b_ema_ = beta_ * b_ema_ + (1.0f - beta_) * b;
    }

    // Ring buffer: overwrite oldest entry, O(1)
    hist_[hist_head_] = b;
    hist_head_ = (hist_head_ + 1) % hist_len_;
    if (hist_count_ < hist_len_) {
        hist_count_++;
    }
}

// Copy valid ring buffer entries into a contiguous vector for percentile/median ops
static std::vector<double> ring_to_vec(const std::vector<float>& buf, int head, int count) {
    std::vector<double> out(count);
    int len = static_cast<int>(buf.size());
    int start = (head - count + len) % len;
    for (int i = 0; i < count; ++i) {
        out[i] = buf[(start + i) % len];
    }
    return out;
}

float BaselineAutoState::b_ref() const {
    if (has_ema_) {
        return b_ema_;
    }
    if (hist_count_ > 0) {
        auto hist_d = ring_to_vec(hist_, hist_head_, hist_count_);
        return static_cast<float>(compute_median(hist_d));
    }
    return 0.1f;
}

float BaselineAutoState::guard_threshold() const {
    if (hist_count_ == 0) {
        return 0.01f;
    }
    // Compute 15th percentile
    auto hist_d = ring_to_vec(hist_, hist_head_, hist_count_);
    float b_p15 = static_cast<float>(compute_percentile(hist_d, 15.0));
    return std::max(0.01f, 0.5f * b_p15);
}

bool BaselineAutoState::should_disable(float baseline, float extra_min) const {
    float thr_auto = guard_threshold();
    float thr = std::max(extra_min, thr_auto);
    return !std::isfinite(baseline) || baseline < thr;
}

// Helper: compute MAD (Median Absolute Deviation)
static double compute_mad(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double med = compute_median(v);
    std::vector<double> abs_dev(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        abs_dev[i] = std::abs(v[i] - med);
    }
    return compute_median(abs_dev);
}

// cv::Mat labels overload: build LabelIndex and delegate to the faster version
MetricScaleResult solve_metric_from_rel(
    const cv::Mat& rel_depth,
    const cv::Mat& z_obs,
    const cv::Mat& mask,
    const cv::Mat& labels,
    const cv::Mat& v_px,
    const MetricScaleConfig& config
) {
    LabelIndex label_index;
    if (!labels.empty()) {
        label_index = build_label_index(labels);
    }
    return solve_metric_from_rel(rel_depth, z_obs, mask, label_index, v_px, config);
}

// ============================================================================
// solve_metric_from_rel with pre-built LabelIndex (faster version)
// ============================================================================

MetricScaleResult solve_metric_from_rel(
    const cv::Mat& rel_depth,
    const cv::Mat& z_obs,
    const cv::Mat& mask,
    const LabelIndex& label_index,
    const cv::Mat& v_px,
    const MetricScaleConfig& config
) {
    MetricScaleResult result;
    const int H = rel_depth.rows;
    const int W = rel_depth.cols;
    const int total_pixels = H * W;

    result.z_out = cv::Mat(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    result.S_map = cv::Mat(H, W, CV_32F, cv::Scalar(1.0f));
    result.V_out = cv::Mat(H, W, CV_32F, cv::Scalar(1e-2f));  // Match original initialization
    result.global_scale = 1.0f;
    result.global_med_rel = 0.0f;
    result.global_p90_rel = 0.0f;

    bool has_mask = !mask.empty();
    bool has_vpx = !v_px.empty();

    const float* rd_data = rel_depth.ptr<float>(0);
    const float* zo_data = z_obs.ptr<float>(0);
    const uint8_t* mask_data = has_mask ? mask.ptr<uint8_t>(0) : nullptr;
    const float* vpx_data = has_vpx ? v_px.ptr<float>(0) : nullptr;

    // Global scale estimation - use all pixels for accuracy
    // (sampling caused bias in some sequences like 0013)
    const int sample_step = 1;
    std::vector<double> global_ratios;
    std::vector<double> var_vals;
    global_ratios.reserve(total_pixels / (sample_step * sample_step));
    if (has_vpx) var_vals.reserve(total_pixels / (sample_step * sample_step));

    for (int r = 0; r < H; r += sample_step) {
        int row_offset = r * W;
        for (int c = 0; c < W; c += sample_step) {
            int i = row_offset + c;
            float rd = rd_data[i];
            float zo = zo_data[i];
            if (std::isfinite(rd) && std::isfinite(zo) &&
                rd > 0 && zo > 0 &&
                (!has_mask || mask_data[i] != 0)) {
                global_ratios.push_back(static_cast<double>(zo) / static_cast<double>(rd));
            }
            if (has_vpx) {
                float v = vpx_data[i];
                if (std::isfinite(v)) var_vals.push_back(v);
            }
        }
    }

    if (global_ratios.empty()) {
        // No valid pixels for scale estimation
        // Fill z_out with whatever is available: z_obs > rel_depth > 1.0
        float* z_out_data = result.z_out.ptr<float>(0);
        for (int i = 0; i < total_pixels; ++i) {
            float zo = zo_data[i];
            float rd = rd_data[i];

            if (std::isfinite(zo) && zo > 0) {
                z_out_data[i] = zo;
            } else if (std::isfinite(rd) && rd > 0) {
                z_out_data[i] = rd;  // Use rel_depth as-is (scale=1.0)
            } else {
                z_out_data[i] = 1.0f;  // Fallback constant
            }
        }
        result.global_scale = 1.0f;
        result.global_med_rel = 1.0f;
        result.global_p90_rel = 1.0f;
        return result;
    }

    // MAD-based inlier selection (match cv::Mat version exactly)
    std::vector<double> ratios_copy = global_ratios;
    double med = compute_median(ratios_copy);
    double mad = compute_mad(ratios_copy);

    std::vector<double> inlier_ratios;
    if (mad > 0) {
        for (double r : global_ratios) {
            if (std::abs(r - med) <= config.global_trim_k * mad) {
                inlier_ratios.push_back(r);
            }
        }
    } else {
        inlier_ratios = global_ratios;
    }

    if (inlier_ratios.empty()) {
        inlier_ratios = global_ratios;
    }

    // Global scale estimation
    double s_global = compute_median(inlier_ratios);

    // Refine with MAD-based second pass
    std::vector<double> refined_ratios;
    double s_mad = compute_mad(inlier_ratios);
    if (s_mad > 0) {
        for (double r : inlier_ratios) {
            if (std::abs(r - s_global) <= 3.5 * s_mad) {
                refined_ratios.push_back(r);
            }
        }
        if (!refined_ratios.empty()) {
            s_global = compute_median(refined_ratios);
        }
    }

    // Sanity check: clamp scale to reasonable range
    const double MIN_SCALE = 0.1;
    const double MAX_SCALE = 10.0;
    s_global = std::max(MIN_SCALE, std::min(MAX_SCALE, s_global));

    result.global_scale = static_cast<float>(s_global);

    // Global variance from sampled v_px
    float v_global = 1e-2f;  // Match original initialization
    if (has_vpx && !var_vals.empty()) {
        v_global = static_cast<float>(std::max(
            static_cast<double>(config.var_floor),
            std::min(static_cast<double>(config.var_cap), compute_median(var_vals))));
    }

    // Global error metrics from sample
    std::vector<double> rel_errors;
    rel_errors.reserve(global_ratios.size());
    for (int r = 0; r < H; r += sample_step) {
        int row_offset = r * W;
        for (int c = 0; c < W; c += sample_step) {
            int i = row_offset + c;
            float rd = rd_data[i];
            float zo = zo_data[i];
            if (std::isfinite(rd) && std::isfinite(zo) &&
                rd > 0 && zo > 0 &&
                (!has_mask || mask_data[i] != 0)) {
                double z_pred = s_global * rd;
                rel_errors.push_back(std::abs(z_pred - zo) / std::max(static_cast<double>(zo), 1e-6));
            }
        }
    }

    if (!rel_errors.empty()) {
        std::vector<double> errors_copy = rel_errors;
        result.global_med_rel = static_cast<float>(compute_median(errors_copy));
        result.global_p90_rel = static_cast<float>(compute_percentile(errors_copy, 90.0));
    }

    // Per-label processing using LabelIndex
    if (!label_index.empty()) {
        float* z_out_data = result.z_out.ptr<float>(0);
        float* s_map_data = result.S_map.ptr<float>(0);
        float* v_out_data = result.V_out.ptr<float>(0);

        const int maxL = label_index.num_labels;

        #pragma omp parallel for schedule(dynamic)
        for (int L = 0; L < maxL; ++L) {
            const int n_pixels = label_index.count(L);
            if (n_pixels == 0) continue;

            const int* px_begin = label_index.begin(L);
            const int* px_end = label_index.end(L);

            // ===== OPTIMIZED: Avoid LabelPixelData struct, use indices directly =====
            // Pass 1: Collect ratios and variance for valid pixels
            std::vector<double> label_ratios;
            label_ratios.reserve(n_pixels);
            std::vector<double> var_values;
            if (has_vpx) var_values.reserve(n_pixels);

            for (const int* px = px_begin; px != px_end; ++px) {
                int idx = *px;
                float rd = rd_data[idx];
                float zo = zo_data[idx];
                if (std::isfinite(rd) && std::isfinite(zo) &&
                    rd > 0 && zo > 0 &&
                    (!has_mask || mask_data[idx] != 0)) {
                    label_ratios.push_back(static_cast<double>(zo) / static_cast<double>(rd));
                }
                if (has_vpx) {
                    float v = vpx_data[idx];
                    if (std::isfinite(v)) var_values.push_back(v);
                }
            }

            // Decide scale for this label
            float scale_L = static_cast<float>(s_global);
            float var_L = v_global;

            int valid_pts = static_cast<int>(label_ratios.size());
            float ratio_valid = (n_pixels > 0) ? static_cast<float>(valid_pts) / n_pixels : 0.0f;

            if (valid_pts >= config.min_pts_per_label && ratio_valid >= config.min_pts_ratio) {
                double s_L = compute_median(label_ratios);
                double mad_L = compute_mad(label_ratios);

                if (mad_L > 0) {
                    std::vector<double> inl;
                    inl.reserve(label_ratios.size());
                    for (double r : label_ratios) {
                        if (std::abs(r - s_L) <= 3.5 * mad_L) {
                            inl.push_back(r);
                        }
                    }
                    if (!inl.empty()) {
                        s_L = compute_median(inl);
                    }
                }

                // Compute errors directly from data arrays
                std::vector<double> errs;
                errs.reserve(valid_pts);
                for (const int* px = px_begin; px != px_end; ++px) {
                    int idx = *px;
                    float rd = rd_data[idx];
                    float zo = zo_data[idx];
                    if (std::isfinite(rd) && std::isfinite(zo) &&
                        rd > 0 && zo > 0 &&
                        (!has_mask || mask_data[idx] != 0)) {
                        double z_pred = s_L * rd;
                        errs.push_back(std::abs(z_pred - zo) / std::max(static_cast<double>(zo), 1e-6));
                    }
                }

                if (!errs.empty()) {
                    size_t mid = errs.size() / 2;
                    std::nth_element(errs.begin(), errs.begin() + mid, errs.end());
                    float med_err = static_cast<float>(errs[mid]);
                    float p90_err = static_cast<float>(compute_percentile(errs, 90.0));

                    if (med_err <= config.single_med_thr && p90_err <= config.single_p90_thr) {
                        scale_L = static_cast<float>(s_L);
                        if (!var_values.empty()) {
                            var_L = static_cast<float>(std::max(
                                config.var_floor, std::min(config.var_cap,
                                    static_cast<float>(compute_median(var_values)))));
                        }
                    }
                }
            }

            // Pass 2: Apply scale to all pixels in this label
            for (const int* px = px_begin; px != px_end; ++px) {
                int idx = *px;
                float rd = rd_data[idx];
                z_out_data[idx] = scale_L * rd;
                s_map_data[idx] = scale_L;
                v_out_data[idx] = var_L;
            }
        }
    } else {
        // No labels: apply global scale everywhere
        const float s_global_f = static_cast<float>(s_global);
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < H; ++r) {
            const float* rd_row = rel_depth.ptr<float>(r);
            float* z_row = result.z_out.ptr<float>(r);
            float* s_row = result.S_map.ptr<float>(r);
            float* v_row = result.V_out.ptr<float>(r);

            for (int c = 0; c < W; ++c) {
                float rd = rd_row[c];
                if (std::isfinite(rd) && rd > 0) {
                    z_row[c] = s_global_f * rd;
                    s_row[c] = s_global_f;
                    v_row[c] = v_global;
                }
            }
        }
    }

    // Fill remaining NaN/invalid with z_obs or median
    float z_med = 1.0f;
    {
        std::vector<double> valid_z;
        valid_z.reserve(H * W / 4);
        for (int r = 0; r < H; ++r) {
            const float* z_row = result.z_out.ptr<float>(r);
            for (int c = 0; c < W; ++c) {
                if (std::isfinite(z_row[c]) && z_row[c] > 0) {
                    valid_z.push_back(z_row[c]);
                }
            }
        }
        if (!valid_z.empty()) {
            z_med = static_cast<float>(compute_median(valid_z));
        }
    }

    for (int r = 0; r < H; ++r) {
        float* z_row = result.z_out.ptr<float>(r);
        const float* zo_row = z_obs.ptr<float>(r);

        for (int c = 0; c < W; ++c) {
            if (!std::isfinite(z_row[c]) || z_row[c] <= 0) {
                if (std::isfinite(zo_row[c]) && zo_row[c] > 0) {
                    z_row[c] = zo_row[c];
                } else {
                    z_row[c] = z_med;
                }
            }
        }
    }

    return result;
}

}  // namespace pr_depth
