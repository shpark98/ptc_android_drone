/**
 * @file motion_field.cpp
 * @brief Motion field estimation with full RANSAC implementation
 */
#include "pr_depth/motion_field.hpp"
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <set>
#include <iostream>
#include <sstream>
#include <omp.h>

namespace pr_depth {

// ============================================================================
// RANSAC Utility Functions
// ============================================================================

/**
 * Compute Huber weights for robust estimation
 */
std::vector<double> compute_huber_weights(const std::vector<double>& residuals, double delta) {
    std::vector<double> weights(residuals.size());
    for (size_t i = 0; i < residuals.size(); ++i) {
        if (residuals[i] <= delta) {
            weights[i] = 1.0;
        } else {
            weights[i] = delta / (residuals[i] + 1e-12);
        }
    }
    return weights;
}

/**
 * Compute angle error between predicted and observed flow (in degrees)
 */
std::vector<double> compute_angle_errors(
    const std::vector<Eigen::Vector2d>& pred_px,
    const std::vector<Eigen::Vector2d>& obs_px) {

    std::vector<double> errors(pred_px.size());
    for (size_t i = 0; i < pred_px.size(); ++i) {
        double pm = pred_px[i].norm() + 1e-12;
        double om = obs_px[i].norm() + 1e-12;
        double dot = pred_px[i].dot(obs_px[i]) / (pm * om);
        dot = std::max(-1.0, std::min(1.0, dot));
        errors[i] = std::acos(dot) * 180.0 / M_PI;
    }
    return errors;
}

/**
 * Compute MAD (Median Absolute Deviation) threshold
 */
double compute_mad_threshold(const std::vector<double>& values, double k = 3.5) {
    if (values.empty()) return 1e6;

    std::vector<double> sorted = values;
    std::nth_element(sorted.begin(), sorted.begin() + sorted.size()/2, sorted.end());
    double median = sorted[sorted.size()/2];

    std::vector<double> abs_dev(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        abs_dev[i] = std::abs(values[i] - median);
    }

    std::nth_element(abs_dev.begin(), abs_dev.begin() + abs_dev.size()/2, abs_dev.end());
    double mad = abs_dev[abs_dev.size()/2];

    return median + k * mad + 1e-6;
}

/**
 * Cell coverage score (number of cells + bonus for spatial distribution)
 */
double compute_coverage_score(const std::vector<bool>& inlier_mask,
                              const std::vector<int>& cell_ids,
                              int max_cells,
                              double lambda_cell = 1.0) {
    if (inlier_mask.empty()) return 0.0;

    std::vector<int> counts(max_cells, 0);
    int total = 0;
    for (size_t i = 0; i < inlier_mask.size(); ++i) {
        if (inlier_mask[i]) {
            counts[cell_ids[i]]++;
            total++;
        }
    }

    int occupied_cells = 0;
    for (int c : counts) {
        if (c > 0) occupied_cells++;
    }

    return total + lambda_cell * occupied_cells;
}

// ============================================================================
// Translation-only estimation (when rotation is known)
// ============================================================================

/**
 * Solve for translation only, given known rotation (omega).
 * Motion field: flow = B @ omega + r * A @ t
 * With known omega: flow_trans = flow - B @ omega = r * A @ t
 * This is a 3-DOF linear system for t.
 */
Eigen::Vector3d solve_translation_only(
    const std::vector<Eigen::Vector2d>& flow_normalized,
    const std::vector<Eigen::Vector2d>& coords_normalized,
    const std::vector<double>& inv_depths,
    const Eigen::Vector3d& known_omega) {

    int n_points = static_cast<int>(flow_normalized.size());
    if (n_points < 3) {
        throw std::runtime_error("Need at least 3 points for translation estimation");
    }

    // Build linear system: flow_trans = r * A @ t
    // where flow_trans = flow - B @ omega
    Eigen::MatrixXd M(2 * n_points, 3);
    Eigen::VectorXd b(2 * n_points);

    for (int i = 0; i < n_points; ++i) {
        double x = coords_normalized[i](0);
        double y = coords_normalized[i](1);
        double r = inv_depths[i];

        Eigen::Matrix<double, 2, 3> B, A;
        compute_motion_matrices(x, y, B, A);

        // Subtract rotation component from flow
        Eigen::Vector2d flow_rot = B * known_omega;
        Eigen::Vector2d flow_trans = flow_normalized[i] - flow_rot;

        // Equation: flow_trans = r * A @ t
        M.block<1, 3>(2*i, 0) = A.row(0) * r;
        M.block<1, 3>(2*i+1, 0) = A.row(1) * r;

        b(2*i) = flow_trans(0);
        b(2*i+1) = flow_trans(1);
    }

    // Solve using SVD
    Eigen::Vector3d t = M.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    return t;
}

/**
 * RANSAC estimation for translation only (given known rotation).
 * Returns translation vector t (camera frame).
 */
Eigen::Vector3d ransac_estimate_translation_only(
    const std::vector<Eigen::Vector2d>& flow_normalized,
    const std::vector<Eigen::Vector2d>& coords_normalized,
    const std::vector<double>& inv_depths,
    const std::vector<double>& flow_magnitudes,
    const std::vector<int>& cell_ids,
    const CameraIntrinsics& intrinsics,
    int max_cells,
    const Eigen::Vector3d& known_omega,
    const MotionFieldConfig& cfg) {

    // Validate known_omega
    if (!known_omega.allFinite()) {
        return Eigen::Vector3d(0, 0, -1);  // Return default forward direction
    }

    int n_points = static_cast<int>(flow_normalized.size());
    if (n_points < 3) {
        return Eigen::Vector3d(0, 0, -1);  // Not enough points, return default
    }

    // Safety check: all vectors should have same size
    if (static_cast<int>(coords_normalized.size()) != n_points ||
        static_cast<int>(inv_depths.size()) != n_points ||
        static_cast<int>(flow_magnitudes.size()) != n_points ||
        static_cast<int>(cell_ids.size()) != n_points) {
        return Eigen::Vector3d(0, 0, -1);  // Size mismatch, return default
    }

    double fx = intrinsics.fx;
    double fy = intrinsics.fy;

    // Pre-compute B, A matrices and flow_trans (with rotation subtracted)
    std::vector<Eigen::Matrix<double, 2, 3>> B_all(n_points);
    std::vector<Eigen::Matrix<double, 2, 3>> A_all(n_points);
    std::vector<Eigen::Vector2d> flow_trans_all(n_points);

    for (int i = 0; i < n_points; ++i) {
        double x = coords_normalized[i](0);
        double y = coords_normalized[i](1);
        compute_motion_matrices(x, y, B_all[i], A_all[i]);
        flow_trans_all[i] = flow_normalized[i] - B_all[i] * known_omega;
    }

    // Build cell-to-indices map
    std::unordered_map<int, std::vector<int>> cell_to_idx;
    for (int i = 0; i < n_points; ++i) {
        cell_to_idx[cell_ids[i]].push_back(i);
    }

    std::vector<int> unique_cells;
    for (const auto& pair : cell_to_idx) {
        unique_cells.push_back(pair.first);
    }

    if (unique_cells.empty()) {
        throw std::runtime_error("No valid cells for translation RANSAC");
    }

    // RANSAC loop
    Eigen::Vector3d best_t = Eigen::Vector3d(0, 0, -1);  // Default: forward direction
    double best_score = -1.0;

    std::mt19937 gen(cfg.seed == 0 ? std::random_device{}() : cfg.seed);

    // Use fewer iterations for translation-only (3 DOF vs 6 DOF)
    int max_iters = std::min(cfg.ransac_max_iters, 100);

    for (int iter = 0; iter < max_iters; ++iter) {
        // Sample 3 cells (minimum for translation)
        int s_eff = std::min(3, static_cast<int>(unique_cells.size()));
        if (s_eff < 3) {
            // Not enough cells - use all points from available cells
            s_eff = static_cast<int>(unique_cells.size());
        }

        std::vector<int> sampled_cells;
        std::sample(unique_cells.begin(), unique_cells.end(),
                   std::back_inserter(sampled_cells), s_eff, gen);

        // Pick one point from each cell
        std::vector<int> sample_indices;
        for (int cell : sampled_cells) {
            auto it = cell_to_idx.find(cell);
            if (it == cell_to_idx.end() || it->second.empty()) continue;
            const auto& indices = it->second;
            std::uniform_int_distribution<> dis(0, static_cast<int>(indices.size()) - 1);
            sample_indices.push_back(indices[dis(gen)]);
        }

        if (sample_indices.size() < 3) continue;

        // Build sample data
        std::vector<Eigen::Vector2d> flow_sample, coords_sample;
        std::vector<double> inv_depths_sample;
        for (int idx : sample_indices) {
            if (idx < 0 || idx >= n_points) continue;
            flow_sample.push_back(flow_normalized[idx]);
            coords_sample.push_back(coords_normalized[idx]);
            inv_depths_sample.push_back(inv_depths[idx]);
        }

        if (flow_sample.size() < 3) continue;

        // Solve for translation
        Eigen::Vector3d t_s;
        try {
            t_s = solve_translation_only(flow_sample, coords_sample, inv_depths_sample, known_omega);
        } catch (...) {
            continue;
        }

        // Check for valid result
        if (!t_s.allFinite()) continue;

        // Score: compute inliers using MAGSAC++ or MAD-based scoring
        double score_s = 0.0;

        for (int i = 0; i < n_points; ++i) {
            double r = inv_depths[i];
            const auto& A = A_all[i];

            // Predicted translation flow
            Eigen::Vector2d pred_trans = A * t_s * r;
            Eigen::Vector2d err = flow_trans_all[i] - pred_trans;

            double ex = err(0) * fx, ey = err(1) * fy;
            double res_px = std::sqrt(ex * ex + ey * ey);
            double flow_mag = flow_magnitudes[i];
            if (flow_mag < 1e-6) flow_mag = 1e-6;  // Avoid division by zero
            double rel_res = res_px / flow_mag;

            if (cfg.use_magsac_scoring) {
                double weight = std::max(0.0, 1.0 - rel_res / cfg.magsac_rel_sigma_max);
                score_s += weight;
            } else {
                if (rel_res < cfg.ransac_thresh_ratio * 0.2) {
                    score_s += 1.0;
                }
            }
        }

        if (score_s > best_score) {
            best_t = t_s;
            best_score = score_s;
        }
    }

    if (best_score < 0 || !best_t.allFinite()) {
        // Return default forward direction if RANSAC failed
        return Eigen::Vector3d(0, 0, -1);
    }

    return best_t;
}

// ============================================================================
// MotionFieldEstimator Implementation
// ============================================================================

MotionFieldEstimator::MotionFieldEstimator(const MotionFieldConfig& cfg)
    : cfg_(cfg) {}

Eigen::VectorXd MotionFieldEstimator::solve_least_squares(
    const std::vector<Eigen::Vector2d>& flow_normalized,
    const std::vector<Eigen::Vector2d>& coords_normalized,
    const std::vector<double>& inv_depths) {

    int n_points = flow_normalized.size();

    // Determine number of variables based on depth_scale_mode
    // mode 0: 6 vars (omega[3], t[3])
    // mode 1: 7 vars - same as mode 0 (s*t, normalize later)
    // mode 2: 7 vars CONSTRAINED (omega[3], t[3], alpha[1])
    //         flow = B @ omega + (r + alpha) * A @ t
    //         This constrains t_offset = alpha * t (same direction as t)

    int n_vars = 6;  // default: omega(3) + t(3)
    if (cfg_.depth_scale_mode == 2) {
        n_vars = 7;  // omega(3) + t(3) + alpha(1) - CONSTRAINED
    }

    if (n_points < (n_vars + 1) / 2) {
        throw std::runtime_error("Need more points for motion field estimation");
    }

    if (cfg_.depth_scale_mode == 2) {
        // Constrained Mode 2: flow = B @ omega + (r + alpha) * A @ t
        // This is nonlinear in (t, alpha), so we use alternating optimization:
        // 1. Initialize with mode 0 solution (alpha = 0)
        // 2. Fix t direction, solve for omega, |t|, alpha
        // 3. Iterate if needed

        // Step 1: Solve mode 0 to get initial t direction
        Eigen::MatrixXd M0(2 * n_points, 6);
        Eigen::VectorXd b0(2 * n_points);

        for (int i = 0; i < n_points; ++i) {
            double x = coords_normalized[i](0);
            double y = coords_normalized[i](1);
            double r = inv_depths[i];

            Eigen::Matrix<double, 2, 3> B, A;
            compute_motion_matrices(x, y, B, A);

            M0.block<1, 3>(2*i, 0) = B.row(0);
            M0.block<1, 3>(2*i+1, 0) = B.row(1);
            M0.block<1, 3>(2*i, 3) = A.row(0) * r;
            M0.block<1, 3>(2*i+1, 3) = A.row(1) * r;

            b0(2*i) = flow_normalized[i](0);
            b0(2*i+1) = flow_normalized[i](1);
        }

        Eigen::VectorXd theta0 = M0.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b0);
        Eigen::Vector3d omega_init = theta0.head<3>();
        Eigen::Vector3d t_init = theta0.tail<3>();
        double t_norm = t_init.norm();
        if (t_norm < 1e-10) {
            // Degenerate case: return mode 0 result with alpha=0
            Eigen::VectorXd theta(7);
            theta.head<6>() = theta0;
            theta(6) = 0.0;
            return theta;
        }
        Eigen::Vector3d t_dir = t_init / t_norm;

        // Step 2: Fix t direction, solve linear system for [omega(3), scale(1), alpha(1)]
        // flow = B @ omega + (r + alpha) * scale * A @ t_dir
        //      = B @ omega + r * scale * A @ t_dir + alpha * scale * A @ t_dir
        // Let s = scale, a = alpha * scale
        // flow = B @ omega + r * s * (A @ t_dir) + a * (A @ t_dir)
        // This is linear in [omega, s, a] (5 variables)

        Eigen::MatrixXd M2(2 * n_points, 5);
        Eigen::VectorXd b2(2 * n_points);

        for (int i = 0; i < n_points; ++i) {
            double x = coords_normalized[i](0);
            double y = coords_normalized[i](1);
            double r = inv_depths[i];

            Eigen::Matrix<double, 2, 3> B, A;
            compute_motion_matrices(x, y, B, A);

            // A @ t_dir (2x1)
            Eigen::Vector2d At = A * t_dir;

            M2.block<1, 3>(2*i, 0) = B.row(0);
            M2.block<1, 3>(2*i+1, 0) = B.row(1);
            M2(2*i, 3) = At(0) * r;      // r * s * (A @ t_dir)
            M2(2*i+1, 3) = At(1) * r;
            M2(2*i, 4) = At(0);          // a * (A @ t_dir) where a = alpha * s
            M2(2*i+1, 4) = At(1);

            b2(2*i) = flow_normalized[i](0);
            b2(2*i+1) = flow_normalized[i](1);
        }

        Eigen::VectorXd theta2 = M2.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b2);

        // Extract: omega(3), s(1), a(1) where a = alpha * s
        Eigen::Vector3d omega = theta2.head<3>();
        double s = theta2(3);  // scale
        double a = theta2(4);  // alpha * scale

        // Recover t = s * t_dir, alpha = a / s
        Eigen::Vector3d t = s * t_dir;
        double alpha = (std::abs(s) > 1e-10) ? (a / s) : 0.0;

        // Return theta = [omega(3), t(3), alpha(1)]
        Eigen::VectorXd theta(7);
        theta.head<3>() = omega;
        theta.segment<3>(3) = t;
        theta(6) = alpha;
        return theta;

    } else {
        // Standard mode (mode 0/1): flow = B @ omega + r * A @ t
        Eigen::MatrixXd M(2 * n_points, n_vars);
        Eigen::VectorXd b(2 * n_points);

        for (int i = 0; i < n_points; ++i) {
            double x = coords_normalized[i](0);
            double y = coords_normalized[i](1);
            double r = inv_depths[i];

            Eigen::Matrix<double, 2, 3> B, A;
            compute_motion_matrices(x, y, B, A);

            M.block<1, 3>(2*i, 0) = B.row(0);
            M.block<1, 3>(2*i+1, 0) = B.row(1);
            M.block<1, 3>(2*i, 3) = A.row(0) * r;
            M.block<1, 3>(2*i+1, 3) = A.row(1) * r;

            b(2*i) = flow_normalized[i](0);
            b(2*i+1) = flow_normalized[i](1);
        }

        // Solve using SVD
        Eigen::VectorXd theta = M.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
        return theta;
    }
}

Eigen::VectorXd MotionFieldEstimator::refine_irls(
    const std::vector<Eigen::Vector2d>& flow_normalized,
    const std::vector<Eigen::Vector2d>& coords_normalized,
    const std::vector<double>& inv_depths,
    const Eigen::VectorXd& theta_init,
    const CameraIntrinsics& intrinsics) {

    int n_points = flow_normalized.size();
    Eigen::VectorXd theta = theta_init;

    double fx = intrinsics.fx;
    double fy = intrinsics.fy;
    double delta = cfg_.huber_delta_rel;

    // OPTIMIZATION: Pre-compute B, A matrices once for all points
    std::vector<Eigen::Matrix<double, 2, 3>> B_irls(n_points);
    std::vector<Eigen::Matrix<double, 2, 3>> A_irls(n_points);
    for (int i = 0; i < n_points; ++i) {
        double x = coords_normalized[i](0);
        double y = coords_normalized[i](1);
        compute_motion_matrices(x, y, B_irls[i], A_irls[i]);
    }

    // Determine number of variables based on depth_scale_mode
    int n_vars = 6;
    if (cfg_.depth_scale_mode == 2) {
        n_vars = 7;  // CONSTRAINED: omega(3) + t(3) + alpha(1)
    }

    for (int iter = 0; iter < cfg_.lo_irls_iters; ++iter) {
        // Compute residuals and weights
        std::vector<double> weights(n_points);
        Eigen::Vector3d omega = theta.head<3>();
        Eigen::Vector3d t;
        double alpha = 0.0;

        if (cfg_.depth_scale_mode == 2) {
            // Constrained mode: theta = [omega(3), t(3), alpha(1)]
            t = theta.segment<3>(3);
            alpha = theta(6);
        } else {
            t = theta.tail<3>();
        }

        for (int i = 0; i < n_points; ++i) {
            double r = inv_depths[i];
            const auto& B = B_irls[i];
            const auto& A = A_irls[i];

            // Predicted flow (normalized)
            // Constrained mode: flow = B @ omega + (r + alpha) * A @ t
            Eigen::Vector2d pred_n;
            double r_eff = (cfg_.depth_scale_mode == 2) ? (r + alpha) : r;
            pred_n(0) = B.row(0).dot(omega) + A.row(0).dot(t) * r_eff;
            pred_n(1) = B.row(1).dot(omega) + A.row(1).dot(t) * r_eff;

            // Residual in pixels
            Eigen::Vector2d err = flow_normalized[i] - pred_n;
            double ex = err(0) * fx, ey = err(1) * fy;
            double res_px = std::sqrt(ex * ex + ey * ey);

            // Flow magnitude for relative normalization
            double fmx = flow_normalized[i](0) * fx, fmy = flow_normalized[i](1) * fy;
            double flow_mag = std::sqrt(fmx * fmx + fmy * fmy);
            double scale = std::max(flow_mag, static_cast<double>(cfg_.min_flow_px));

            // Relative residual
            double r_rel = res_px / scale;

            // Huber weight
            weights[i] = (r_rel <= delta) ? 1.0 : (delta / (r_rel + 1e-12));
        }

        // Weighted least squares with constrained formulation
        if (cfg_.depth_scale_mode == 2) {
            // Constrained mode: flow = B @ omega + (r + alpha) * A @ t
            // Use alternating optimization: fix t direction, solve for [omega, scale, alpha*scale]

            // Get current t direction
            double t_norm = t.norm();
            if (t_norm < 1e-10) {
                // Degenerate: keep current theta
                break;
            }
            Eigen::Vector3d t_dir = t / t_norm;

            // Build weighted system for [omega(3), s(1), a(1)] where t = s*t_dir, t_offset = a*t_dir
            Eigen::MatrixXd M_weighted(2 * n_points, 5);
            Eigen::VectorXd b_weighted(2 * n_points);

            for (int i = 0; i < n_points; ++i) {
                double r = inv_depths[i];
                double w = std::sqrt(weights[i]);
                const auto& B = B_irls[i];
                const auto& A = A_irls[i];

                Eigen::Vector2d At = A * t_dir;

                M_weighted.block<1, 3>(2*i, 0) = B.row(0) * w;
                M_weighted.block<1, 3>(2*i+1, 0) = B.row(1) * w;
                M_weighted(2*i, 3) = At(0) * r * w;      // r * s * (A @ t_dir)
                M_weighted(2*i+1, 3) = At(1) * r * w;
                M_weighted(2*i, 4) = At(0) * w;          // a * (A @ t_dir)
                M_weighted(2*i+1, 4) = At(1) * w;

                b_weighted(2*i) = flow_normalized[i](0) * w;
                b_weighted(2*i+1) = flow_normalized[i](1) * w;
            }

            Eigen::VectorXd theta5 = M_weighted.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b_weighted);

            // Extract and convert back to 7-var theta
            Eigen::Vector3d omega_new = theta5.head<3>();
            double s_new = theta5(3);
            double a_new = theta5(4);

            Eigen::Vector3d t_new = s_new * t_dir;
            double alpha_new = (std::abs(s_new) > 1e-10) ? (a_new / s_new) : 0.0;

            Eigen::VectorXd theta_new(7);
            theta_new.head<3>() = omega_new;
            theta_new.segment<3>(3) = t_new;
            theta_new(6) = alpha_new;

            // Check convergence
            if ((theta_new - theta).norm() <= 1e-8 * (theta.norm() + 1e-12)) {
                theta = theta_new;
                break;
            }

            theta = theta_new;

        } else {
            // Standard mode (mode 0/1)
            Eigen::MatrixXd M_weighted(2 * n_points, n_vars);
            Eigen::VectorXd b_weighted(2 * n_points);

            for (int i = 0; i < n_points; ++i) {
                double r = inv_depths[i];
                double w = std::sqrt(weights[i]);
                const auto& B = B_irls[i];
                const auto& A = A_irls[i];

                M_weighted.block<1, 3>(2*i, 0) = B.row(0) * w;
                M_weighted.block<1, 3>(2*i+1, 0) = B.row(1) * w;
                M_weighted.block<1, 3>(2*i, 3) = A.row(0) * r * w;
                M_weighted.block<1, 3>(2*i+1, 3) = A.row(1) * r * w;

                b_weighted(2*i) = flow_normalized[i](0) * w;
                b_weighted(2*i+1) = flow_normalized[i](1) * w;
            }

            Eigen::VectorXd theta_new = M_weighted.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b_weighted);

            // Check convergence
            if ((theta_new - theta).norm() <= 1e-8 * (theta.norm() + 1e-12)) {
                theta = theta_new;
                break;
            }

            theta = theta_new;
        }
    }

    return theta;
}

/**
 * Full RANSAC estimation with cell-based sampling and adaptive thresholds
 * OPTIMIZED: Pre-compute B, A matrices, reuse memory
 */
Eigen::VectorXd MotionFieldEstimator::ransac_estimate(
    const std::vector<Eigen::Vector2d>& flow_normalized,
    const std::vector<Eigen::Vector2d>& coords_normalized,
    const std::vector<double>& inv_depths,
    const std::vector<double>& flow_magnitudes,
    const std::vector<int>& cell_ids,
    const CameraIntrinsics& intrinsics,
    int max_cells) {

    int n_points = flow_normalized.size();
    if (n_points < cfg_.ransac_min_sample) {
        throw std::runtime_error("Not enough points for RANSAC");
    }

    double fx = intrinsics.fx;
    double fy = intrinsics.fy;

    // Determine number of variables based on depth_scale_mode
    int n_vars = 6;
    if (cfg_.depth_scale_mode == 2) {
        n_vars = 7;  // CONSTRAINED: omega(3) + t(3) + alpha(1)
    }

    // ===== OPTIMIZATION: Pre-compute B, A matrices for all points =====
    // These are constant throughout RANSAC, so compute once
    std::vector<Eigen::Matrix<double, 2, 3>> B_all(n_points);
    std::vector<Eigen::Matrix<double, 2, 3>> A_all(n_points);

    for (int i = 0; i < n_points; ++i) {
        double x = coords_normalized[i](0);
        double y = coords_normalized[i](1);
        compute_motion_matrices(x, y, B_all[i], A_all[i]);
    }

    // Build cell-to-indices map
    std::unordered_map<int, std::vector<int>> cell_to_idx;
    for (int i = 0; i < n_points; ++i) {
        cell_to_idx[cell_ids[i]].push_back(i);
    }

    std::vector<int> unique_cells;
    unique_cells.reserve(cell_to_idx.size());
    for (const auto& pair : cell_to_idx) {
        unique_cells.push_back(pair.first);
    }

    if (unique_cells.empty()) {
        throw std::runtime_error("No valid cells for RANSAC");
    }

    // Initialize threshold with robust estimate
    double thr_eval = cfg_.ransac_thresh_ratio;

    // Seed threshold using initial subset
    {
        int seed_take = std::min(256, n_points);
        std::vector<Eigen::Vector2d> flow_seed(flow_normalized.begin(), flow_normalized.begin() + seed_take);
        std::vector<Eigen::Vector2d> coords_seed(coords_normalized.begin(), coords_normalized.begin() + seed_take);
        std::vector<double> inv_depths_seed(inv_depths.begin(), inv_depths.begin() + seed_take);

        try {
            Eigen::VectorXd theta_seed = solve_least_squares(flow_seed, coords_seed, inv_depths_seed);

            // Compute residuals on full set
            std::vector<double> res_ratios;
            Eigen::Vector3d omega_seed = theta_seed.head<3>();
            Eigen::Vector3d t_seed = theta_seed.tail<3>();
            for (int i = 0; i < n_points; ++i) {
                double r = inv_depths[i];
                const auto& B = B_all[i];
                const auto& A = A_all[i];

                Eigen::Vector2d pred_n;
                pred_n(0) = B.row(0).dot(omega_seed) + A.row(0).dot(t_seed) * r;
                pred_n(1) = B.row(1).dot(omega_seed) + A.row(1).dot(t_seed) * r;

                Eigen::Vector2d err = flow_normalized[i] - pred_n;
                double ex = err(0) * fx, ey = err(1) * fy;
                double res_px = std::sqrt(ex * ex + ey * ey);
                res_ratios.push_back(res_px / flow_magnitudes[i]);
            }

            thr_eval = compute_mad_threshold(res_ratios, 3.5);
            thr_eval = std::max(cfg_.ransac_thresh_ratio * 0.25,
                              std::min(thr_eval, cfg_.ransac_thresh_ratio * 2.0));
        } catch (...) {
            // Use default if seed fails
        }
    }

    // RANSAC loop (sequential - parallel overhead too high for 100 iterations)
    Eigen::VectorXd best_theta;
    double best_score = -1.0;
    int no_improve = 0;
    int patience = std::max(10, cfg_.ransac_max_iters / 10);

    std::mt19937 gen;
    if (cfg_.seed == 0) {
        std::random_device rd;
        gen.seed(rd());
    } else {
        gen.seed(cfg_.seed);
    }

    // Pre-allocate reusable buffers for RANSAC loop (avoid per-iteration heap allocation)
    std::vector<bool> inlier_mask(n_points);
    std::vector<double> weights_magsac(n_points);
    std::vector<Eigen::Vector2d> pred_px_iter(n_points);
    std::vector<Eigen::Vector2d> obs_px_iter(n_points);
    std::vector<double> rel_residuals(n_points);

    for (int iter = 0; iter < cfg_.ransac_max_iters; ++iter) {
        // Sample cells
        int s_eff = std::min(cfg_.ransac_min_sample, static_cast<int>(unique_cells.size()));

        std::vector<int> sampled_cells;
        std::sample(unique_cells.begin(), unique_cells.end(),
                   std::back_inserter(sampled_cells), s_eff, gen);

        // Pick one point from each cell
        std::vector<int> sample_indices;
        for (int cell : sampled_cells) {
            const auto& indices = cell_to_idx[cell];
            if (indices.size() == 1) {
                sample_indices.push_back(indices[0]);
            } else {
                std::uniform_int_distribution<> dis(0, indices.size() - 1);
                sample_indices.push_back(indices[dis(gen)]);
            }
        }

        // Build sample data
        std::vector<Eigen::Vector2d> flow_sample, coords_sample;
        std::vector<double> inv_depths_sample;
        for (int idx : sample_indices) {
            flow_sample.push_back(flow_normalized[idx]);
            coords_sample.push_back(coords_normalized[idx]);
            inv_depths_sample.push_back(inv_depths[idx]);
        }

        // Check condition number - use pre-computed B, A matrices
        // For constrained mode 2, we check condition of the 6-var system (mode 0)
        // since the actual solve uses mode 0 first, then constrains
        int n_vars_check = (cfg_.depth_scale_mode == 2) ? 6 : n_vars;
        Eigen::MatrixXd M_sample(2 * sample_indices.size(), n_vars_check);
        for (size_t i = 0; i < sample_indices.size(); ++i) {
            int idx = sample_indices[i];
            double r = inv_depths_sample[i];
            const auto& B = B_all[idx];
            const auto& A = A_all[idx];

            M_sample.block<1, 3>(2*i, 0) = B.row(0);
            M_sample.block<1, 3>(2*i+1, 0) = B.row(1);
            // Both mode 0 and constrained mode 2 use same structure for condition check
            M_sample.block<1, 3>(2*i, 3) = A.row(0) * r;
            M_sample.block<1, 3>(2*i+1, 3) = A.row(1) * r;
        }

        Eigen::MatrixXd G = M_sample.transpose() * M_sample;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(G);
        Eigen::VectorXd eigenvalues = es.eigenvalues();

        double min_eval = eigenvalues.minCoeff();
        double max_eval = eigenvalues.maxCoeff();
        double cond_ratio = (max_eval > 1e-12) ? (std::sqrt(min_eval) / std::sqrt(max_eval)) : 0.0;

        if (cond_ratio < 1e-4) {
            no_improve++;
            if (no_improve >= patience) break;
            continue;
        }

        // Solve for this sample
        Eigen::VectorXd theta_s;
        try {
            theta_s = solve_least_squares(flow_sample, coords_sample, inv_depths_sample);
        } catch (...) {
            no_improve++;
            if (no_improve >= patience) break;
            continue;
        }
        // Evaluate inliers with MAGSAC++ scoring using pre-computed B, A matrices

        // Extract theta components for faster access
        Eigen::Vector3d omega_s = theta_s.head<3>();
        Eigen::Vector3d t_s;
        double alpha_s = 0.0;
        if (cfg_.depth_scale_mode == 2) {
            // Constrained mode: theta = [omega(3), t(3), alpha(1)]
            t_s = theta_s.segment<3>(3);
            alpha_s = theta_s(6);
        } else {
            t_s = theta_s.tail<3>();
        }

        // First pass: compute all predictions
        for (int i = 0; i < n_points; ++i) {
            double r = inv_depths[i];
            const auto& B = B_all[i];
            const auto& A = A_all[i];

            // Constrained mode: flow = B @ omega + (r + alpha) * A @ t
            double r_eff = (cfg_.depth_scale_mode == 2) ? (r + alpha_s) : r;
            Eigen::Vector2d pred_n;
            pred_n(0) = B.row(0).dot(omega_s) + A.row(0).dot(t_s) * r_eff;
            pred_n(1) = B.row(1).dot(omega_s) + A.row(1).dot(t_s) * r_eff;

            pred_px_iter[i] = Eigen::Vector2d(pred_n(0) * fx, pred_n(1) * fy);
            obs_px_iter[i] = Eigen::Vector2d(flow_normalized[i](0) * fx, flow_normalized[i](1) * fy);
        }
        // Scoring: MAGSAC++ (soft weighting) or MAD-based (binary threshold)
        double score_s = 0.0;
        int inlier_count = 0;

        // First compute all relative residuals (buffer pre-allocated)
        for (int i = 0; i < n_points; ++i) {
            Eigen::Vector2d err = obs_px_iter[i] - pred_px_iter[i];
            double res_px = err.norm();
            rel_residuals[i] = res_px / flow_magnitudes[i];
        }

        if (cfg_.use_magsac_scoring) {
            // MAGSAC++ scoring: weight = max(0, 1 - rel_res / rel_sigma_max)
            double rel_sigma_max = cfg_.magsac_rel_sigma_max;
            double inlier_weight_thresh = cfg_.magsac_inlier_weight;

            for (int i = 0; i < n_points; ++i) {
                double rel_res = rel_residuals[i];
                double weight = std::max(0.0, 1.0 - rel_res / rel_sigma_max);
                weights_magsac[i] = weight;
                score_s += weight;

                inlier_mask[i] = (weight >= inlier_weight_thresh);
                if (inlier_mask[i]) {
                    inlier_count++;
                }
            }
        } else {
            // MAD-based scoring (paper method): binary threshold from median + k*MAD
            // Plus direction check when use_direction_check is enabled (Supplementary 2.3)
            std::vector<double> sorted_res = rel_residuals;
            std::nth_element(sorted_res.begin(), sorted_res.begin() + n_points/2, sorted_res.end());
            double median_res = sorted_res[n_points/2];

            std::vector<double> abs_dev(n_points);
            for (int i = 0; i < n_points; ++i) {
                abs_dev[i] = std::abs(rel_residuals[i] - median_res);
            }
            std::nth_element(abs_dev.begin(), abs_dev.begin() + n_points/2, abs_dev.end());
            double mad = abs_dev[n_points/2] + 1e-8;

            // Threshold from MAD (using ransac_thresh_ratio as k)
            double thresh = median_res + cfg_.ransac_thresh_ratio * mad;
            thresh = std::max(0.05, std::min(thresh, 0.5));  // Clamp to reasonable range

            for (int i = 0; i < n_points; ++i) {
                bool is_inlier = (rel_residuals[i] <= thresh);

                // Direction check (paper Supplementary 2.3): reject if direction deviation too large
                if (is_inlier && cfg_.use_direction_check) {
                    double pred_mag = pred_px_iter[i].norm() + 1e-12;
                    double obs_mag = obs_px_iter[i].norm() + 1e-12;
                    double cos_sim = pred_px_iter[i].dot(obs_px_iter[i]) / (pred_mag * obs_mag);
                    // cos_sim_thresh = 0.5 means reject if angle > 60 degrees
                    if (cos_sim < cfg_.cos_sim_thresh) {
                        is_inlier = false;
                    }
                }

                inlier_mask[i] = is_inlier;
                if (inlier_mask[i]) {
                    inlier_count++;
                    score_s += 1.0;  // Binary: each inlier contributes 1
                }
                weights_magsac[i] = inlier_mask[i] ? 1.0 : 0.0;
            }
        }

        // Add cell coverage bonus
        double coverage_bonus = compute_coverage_score(inlier_mask, cell_ids, max_cells, 1.0) - inlier_count;
        score_s += coverage_bonus;

        // NOTE: Per paper, IRLS refinement is done only AFTER RANSAC selects best hypothesis
        // "After RANSAC selects the best hypothesis, we refine the parameters using
        //  a small number of IRLS iterations" - Supplementary Material Section 2.4

        // Update best
        if (score_s > best_score) {
            best_theta = theta_s;
            best_score = score_s;
            no_improve = 0;
        } else {
            no_improve++;
            if (no_improve >= patience) break;
        }
    }

    if (best_score < 0) {
        throw std::runtime_error("RANSAC failed to find valid solution");
    }

    // Final IRLS refinement on all inliers - use pre-computed B, A matrices
    std::vector<bool> final_inlier_mask(n_points);
    Eigen::Vector3d omega_final = best_theta.head<3>();
    Eigen::Vector3d t_final;
    double alpha_final = 0.0;
    if (cfg_.depth_scale_mode == 2) {
        // Constrained mode: theta = [omega(3), t(3), alpha(1)]
        t_final = best_theta.segment<3>(3);
        alpha_final = best_theta(6);
    } else {
        t_final = best_theta.tail<3>();
    }

    // Compute residuals for final inlier selection
    // Also cache pred_px/obs_px for direction check reuse
    std::vector<double> final_rel_residuals(n_points);
    for (int i = 0; i < n_points; ++i) {
        double r = inv_depths[i];
        const auto& B = B_all[i];
        const auto& A = A_all[i];

        double r_eff = (cfg_.depth_scale_mode == 2) ? (r + alpha_final) : r;
        Eigen::Vector2d pred_n;
        pred_n(0) = B.row(0).dot(omega_final) + A.row(0).dot(t_final) * r_eff;
        pred_n(1) = B.row(1).dot(omega_final) + A.row(1).dot(t_final) * r_eff;

        pred_px_iter[i] = Eigen::Vector2d(pred_n(0) * fx, pred_n(1) * fy);
        obs_px_iter[i] = Eigen::Vector2d(flow_normalized[i](0) * fx, flow_normalized[i](1) * fy);

        Eigen::Vector2d err = obs_px_iter[i] - pred_px_iter[i];
        double res_px = err.norm();
        final_rel_residuals[i] = res_px / flow_magnitudes[i];
    }

    // Final inlier selection using same method as RANSAC scoring
    if (cfg_.use_magsac_scoring) {
        double rel_sigma_max = cfg_.magsac_rel_sigma_max;
        double inlier_weight_thresh = cfg_.magsac_inlier_weight;
        for (int i = 0; i < n_points; ++i) {
            double weight = std::max(0.0, 1.0 - final_rel_residuals[i] / rel_sigma_max);
            final_inlier_mask[i] = (weight >= inlier_weight_thresh);
        }
    } else {
        // MAD-based threshold with direction check
        std::vector<double> sorted_res = final_rel_residuals;
        std::nth_element(sorted_res.begin(), sorted_res.begin() + n_points/2, sorted_res.end());
        double median_res = sorted_res[n_points/2];

        std::vector<double> abs_dev(n_points);
        for (int i = 0; i < n_points; ++i) {
            abs_dev[i] = std::abs(final_rel_residuals[i] - median_res);
        }
        std::nth_element(abs_dev.begin(), abs_dev.begin() + n_points/2, abs_dev.end());
        double mad = abs_dev[n_points/2] + 1e-8;

        double thresh = median_res + cfg_.ransac_thresh_ratio * mad;
        thresh = std::max(0.05, std::min(thresh, 0.5));

        for (int i = 0; i < n_points; ++i) {
            bool is_inlier = (final_rel_residuals[i] <= thresh);

            // Direction check for final inlier selection (reuse cached pred/obs)
            if (is_inlier && cfg_.use_direction_check) {
                double pred_mag = pred_px_iter[i].norm() + 1e-12;
                double obs_mag = obs_px_iter[i].norm() + 1e-12;
                double cos_sim = pred_px_iter[i].dot(obs_px_iter[i]) / (pred_mag * obs_mag);
                if (cos_sim < cfg_.cos_sim_thresh) {
                    is_inlier = false;
                }
            }

            final_inlier_mask[i] = is_inlier;
        }
    }

    std::vector<Eigen::Vector2d> flow_final, coords_final;
    std::vector<double> inv_depths_final;

    for (int i = 0; i < n_points; ++i) {
        if (final_inlier_mask[i]) {
            flow_final.push_back(flow_normalized[i]);
            coords_final.push_back(coords_normalized[i]);
            inv_depths_final.push_back(inv_depths[i]);
        }
    }

    int min_inliers_for_refine = (cfg_.depth_scale_mode == 2) ? 7 : 6;  // 7 for constrained mode
    if (static_cast<int>(flow_final.size()) >= min_inliers_for_refine) {
        best_theta = refine_irls(flow_final, coords_final, inv_depths_final, best_theta, intrinsics);
    }

    return best_theta;
}

MotionFieldResult MotionFieldEstimator::estimate(
    const cv::Mat& flow,
    const cv::Mat& inv_depth,
    const CameraIntrinsics& intrinsics,
    const cv::Mat& mask,
    const std::optional<Eigen::Vector3d>& known_omega) {

    if (flow.type() != CV_32FC2) {
        throw std::runtime_error("Flow must be CV_32FC2");
    }
    if (inv_depth.type() != CV_32F) {
        throw std::runtime_error("Inverse depth must be CV_32F");
    }
    if (flow.size() != inv_depth.size()) {
        throw std::runtime_error("Flow and inverse depth must have same size");
    }

    // Translation-only mode: when known_omega is provided
    const bool translation_only = known_omega.has_value();

    int H = flow.rows;
    int W = flow.cols;

    // Compute 3rd percentile for filtering
    float r_percentile_3 = 0.0f;
    {
        std::vector<float> valid_inv_depths;
        for (int y = 0; y < H; ++y) {
            const float* row = inv_depth.ptr<float>(y);
            for (int x = 0; x < W; ++x) {
                float r = row[x];
                if (std::isfinite(r)) {
                    valid_inv_depths.push_back(r);
                }
            }
        }
        if (!valid_inv_depths.empty()) {
            std::nth_element(valid_inv_depths.begin(),
                           valid_inv_depths.begin() + valid_inv_depths.size() * 3 / 100,
                           valid_inv_depths.end());
            r_percentile_3 = valid_inv_depths[valid_inv_depths.size() * 3 / 100];
        }
    }

    // Assign cells (grid)
    int grid_cols = 6;
    int grid_rows = 4;
    int max_cells = grid_rows * grid_cols;

    // Collect ALL valid points with cell IDs (for inlier computation after RANSAC)
    // Reserve capacity to avoid reallocations (estimate ~50% valid)
    int estimated_valid = H * W / 2;
    std::vector<Eigen::Vector2d> all_flow_normalized;
    std::vector<Eigen::Vector2d> all_coords_normalized;
    std::vector<double> all_inv_depths_vec;
    std::vector<int> all_cell_ids;
    std::vector<float> all_pixel_x, all_pixel_y;
    std::vector<float> all_flow_u, all_flow_v;
    all_flow_normalized.reserve(estimated_valid);
    all_coords_normalized.reserve(estimated_valid);
    all_inv_depths_vec.reserve(estimated_valid);
    all_cell_ids.reserve(estimated_valid);
    all_pixel_x.reserve(estimated_valid);
    all_pixel_y.reserve(estimated_valid);
    all_flow_u.reserve(estimated_valid);
    all_flow_v.reserve(estimated_valid);

    // Pre-compute constants
    double inv_fx = 1.0 / intrinsics.fx;
    double inv_fy = 1.0 / intrinsics.fy;
    double cx_d = intrinsics.cx;
    double cy_d = intrinsics.cy;
    float min_flow_px_base = cfg_.min_flow_px;
    float adaptive_scale = cfg_.adaptive_flow_depth_scale;

    // Margin boundaries (exclude edge pixels)
    int margin_x = static_cast<int>(W * cfg_.margin_x_pct);
    int margin_y = static_cast<int>(H * cfg_.margin_y_pct);
    int x_min = margin_x;
    int x_max = W - margin_x;
    int y_min = margin_y;
    int y_max = H - margin_y;

    // Use raw pointer access for cv::Mat (faster than .at<>())
    const float* inv_depth_ptr = inv_depth.ptr<float>();
    const cv::Vec2f* flow_ptr = flow.ptr<cv::Vec2f>();
    const uint8_t* mask_ptr = mask.empty() ? nullptr : mask.ptr<uint8_t>();

    for (int y = y_min; y < y_max; ++y) {
        int row_offset = y * W;
        for (int x = x_min; x < x_max; ++x) {
            int idx = row_offset + x;

            if (mask_ptr && mask_ptr[idx] == 0) continue;

            float r = inv_depth_ptr[idx];
            const cv::Vec2f& f = flow_ptr[idx];

            if (!std::isfinite(r) || r <= r_percentile_3) continue;
            if (!std::isfinite(f[0]) || !std::isfinite(f[1])) continue;

            // Adaptive min_flow: far objects (low inv_depth) allow smaller flow
            // near objects (high inv_depth) require larger flow
            float adaptive_min_flow = min_flow_px_base / (1.0f + adaptive_scale * r);
            float adaptive_min_flow_sq = adaptive_min_flow * adaptive_min_flow;

            float flow_mag_sq = f[0] * f[0] + f[1] * f[1];
            if (flow_mag_sq < adaptive_min_flow_sq) continue;

            // Cell ID
            int cx = std::min(x * grid_cols / W, grid_cols - 1);
            int cy_grid = std::min(y * grid_rows / H, grid_rows - 1);
            int cell_id = cy_grid * grid_cols + cx;

            double x_norm = (x - cx_d) * inv_fx;
            double y_norm = (y - cy_d) * inv_fy;
            double u_norm = f[0] * inv_fx;
            double v_norm = f[1] * inv_fy;

            // Store in ALL vectors (for inlier computation)
            all_coords_normalized.emplace_back(x_norm, y_norm);
            all_flow_normalized.emplace_back(u_norm, v_norm);
            all_inv_depths_vec.push_back(static_cast<double>(r));
            all_cell_ids.push_back(cell_id);
            all_pixel_x.push_back(static_cast<float>(x));
            all_pixel_y.push_back(static_cast<float>(y));
            all_flow_u.push_back(f[0]);
            all_flow_v.push_back(f[1]);
        }
    }

    if (all_coords_normalized.size() < 6) {
        std::ostringstream oss;
        oss << "Not enough valid points for motion estimation: "
            << all_coords_normalized.size() << " points (need >= 6)"
            << ", r_percentile_3=" << r_percentile_3
            << ", min_flow_px=" << cfg_.min_flow_px
            << ", adaptive_flow_depth_scale=" << cfg_.adaptive_flow_depth_scale;
        throw std::runtime_error(oss.str());
    }

    // Subsampled vectors (only populated if needed, avoids ~12MB copy when not subsampling)
    std::vector<Eigen::Vector2d> flow_sub, coords_sub;
    std::vector<double> inv_depths_sub;
    std::vector<int> cell_ids_sub;
    bool need_subsample = all_coords_normalized.size() > static_cast<size_t>(cfg_.max_points);

    if (need_subsample) {
        std::mt19937 gen_shuffle;
        if (cfg_.seed == 0) {
            std::random_device rd;
            gen_shuffle.seed(rd());
        } else {
            gen_shuffle.seed(cfg_.seed + 1);
        }

        int n_total = static_cast<int>(all_coords_normalized.size());
        int n_depth_bins = cfg_.depth_bins;

        // Compute depth percentiles for binning (2nd and 98th percentile)
        // Use nth_element O(n) instead of full sort O(n log n)
        std::vector<double> depth_pct = all_inv_depths_vec;
        int p2_idx = std::max(0, static_cast<int>(n_total * 0.02));
        int p98_idx = std::min(n_total - 1, static_cast<int>(n_total * 0.98));
        std::nth_element(depth_pct.begin(), depth_pct.begin() + p98_idx, depth_pct.end());
        double depth_hi = depth_pct[p98_idx];
        std::nth_element(depth_pct.begin(), depth_pct.begin() + p2_idx, depth_pct.begin() + p98_idx);
        double depth_lo = depth_pct[p2_idx];

        if (depth_hi <= depth_lo) {
            depth_hi = depth_lo + 1e-6;
        }
        double inv_span = static_cast<double>(n_depth_bins) / (depth_hi - depth_lo);

        // Assign depth bin to each point
        std::vector<int> depth_bin_ids(n_total);
        for (int i = 0; i < n_total; ++i) {
            double d_clipped = std::max(depth_lo, std::min(depth_hi, all_inv_depths_vec[i]));
            int bin = static_cast<int>((d_clipped - depth_lo) * inv_span);
            depth_bin_ids[i] = std::max(0, std::min(n_depth_bins - 1, bin));
        }

        // Create composite bucket key: depth_bin * n_cells + cell_id
        int n_buckets = n_depth_bins * max_cells;
        std::vector<std::vector<int>> bucket_indices(n_buckets);

        for (int i = 0; i < n_total; ++i) {
            int bucket_key = depth_bin_ids[i] * max_cells + all_cell_ids[i];
            bucket_indices[bucket_key].push_back(i);
        }

        // Count non-empty buckets and their sizes
        std::vector<int> non_empty_buckets;
        std::vector<int> bucket_sizes;
        for (int b = 0; b < n_buckets; ++b) {
            if (!bucket_indices[b].empty()) {
                non_empty_buckets.push_back(b);
                bucket_sizes.push_back(static_cast<int>(bucket_indices[b].size()));
            }
        }

        int n_non_empty = static_cast<int>(non_empty_buckets.size());
        if (n_non_empty == 0) {
            throw std::runtime_error("No valid buckets for stratified sampling");
        }

        // Allocate quota to each bucket proportionally
        std::vector<int> quota(n_non_empty);
        int total_points = 0;
        for (int i = 0; i < n_non_empty; ++i) {
            total_points += bucket_sizes[i];
        }

        int remaining = cfg_.max_points;
        for (int i = 0; i < n_non_empty; ++i) {
            // Proportional allocation with at least 1 per bucket
            int alloc = std::max(1, static_cast<int>(
                static_cast<double>(bucket_sizes[i]) / total_points * cfg_.max_points));
            alloc = std::min(alloc, bucket_sizes[i]);
            alloc = std::min(alloc, remaining);
            quota[i] = alloc;
            remaining -= alloc;
        }

        // Distribute remaining quota
        while (remaining > 0) {
            for (int i = 0; i < n_non_empty && remaining > 0; ++i) {
                if (quota[i] < bucket_sizes[i]) {
                    quota[i]++;
                    remaining--;
                }
            }
            // Break if no bucket can accept more
            bool can_add = false;
            for (int i = 0; i < n_non_empty; ++i) {
                if (quota[i] < bucket_sizes[i]) {
                    can_add = true;
                    break;
                }
            }
            if (!can_add) break;
        }

        // Sample from each bucket
        std::vector<int> sampled_indices;
        sampled_indices.reserve(cfg_.max_points);

        for (int i = 0; i < n_non_empty; ++i) {
            int bucket_id = non_empty_buckets[i];
            auto& indices = bucket_indices[bucket_id];
            int q = quota[i];

            if (q >= static_cast<int>(indices.size())) {
                // Take all
                for (int idx : indices) {
                    sampled_indices.push_back(idx);
                }
            } else {
                // Random sample without replacement
                std::vector<int> shuffled = indices;
                std::shuffle(shuffled.begin(), shuffled.end(), gen_shuffle);
                for (int j = 0; j < q; ++j) {
                    sampled_indices.push_back(shuffled[j]);
                }
            }
        }

        // Build sampled vectors (only the 4 needed for RANSAC)
        std::vector<Eigen::Vector2d> flow_sampled, coords_sampled;
        std::vector<double> inv_depths_sampled;
        std::vector<int> cell_ids_sampled;

        flow_sampled.reserve(sampled_indices.size());
        coords_sampled.reserve(sampled_indices.size());
        inv_depths_sampled.reserve(sampled_indices.size());
        cell_ids_sampled.reserve(sampled_indices.size());

        for (int idx : sampled_indices) {
            flow_sampled.push_back(all_flow_normalized[idx]);
            coords_sampled.push_back(all_coords_normalized[idx]);
            inv_depths_sampled.push_back(all_inv_depths_vec[idx]);
            cell_ids_sampled.push_back(all_cell_ids[idx]);
        }

        flow_sub = std::move(flow_sampled);
        coords_sub = std::move(coords_sampled);
        inv_depths_sub = std::move(inv_depths_sampled);
        cell_ids_sub = std::move(cell_ids_sampled);
    }

    // References to vectors for RANSAC (avoids ~12MB copy when no subsampling needed)
    const auto& flow_normalized = need_subsample ? flow_sub : all_flow_normalized;
    const auto& coords_normalized = need_subsample ? coords_sub : all_coords_normalized;
    const auto& inv_depths_vec = need_subsample ? inv_depths_sub : all_inv_depths_vec;
    const auto& cell_ids = need_subsample ? cell_ids_sub : all_cell_ids;

    // Compute flow magnitudes for RANSAC
    double fx = intrinsics.fx;
    double fy = intrinsics.fy;
    std::vector<double> flow_mags(flow_normalized.size());
    for (size_t i = 0; i < flow_normalized.size(); ++i) {
        double fmx = flow_normalized[i](0) * fx, fmy = flow_normalized[i](1) * fy;
        flow_mags[i] = std::sqrt(fmx * fmx + fmy * fmy);
        flow_mags[i] = std::max(flow_mags[i], static_cast<double>(cfg_.min_flow_px));
    }

    // Extract rotation and translation
    Eigen::Vector3d omega;
    Eigen::Vector3d t_cam;
    double depth_scale = 1.0;
    double depth_offset = 0.0;

    if (translation_only) {
        // ===== Translation-only mode: use known rotation =====
        omega = known_omega.value();

        // Run translation-only RANSAC
        t_cam = ransac_estimate_translation_only(
            flow_normalized, coords_normalized, inv_depths_vec,
            flow_mags, cell_ids, intrinsics, max_cells,
            omega, cfg_);

    } else {
        // ===== Full estimation: joint rotation + translation =====
        // Run RANSAC
        Eigen::VectorXd theta = ransac_estimate(
            flow_normalized, coords_normalized, inv_depths_vec,
            flow_mags, cell_ids, intrinsics, max_cells);

        // Extract rotation and translation based on depth_scale_mode
        omega = theta.head<3>();

        if (cfg_.depth_scale_mode == 2) {
            // Constrained mode: theta = [omega(3), t(3), alpha(1)]
            // flow = B @ omega + (r + alpha) * A @ t
            // This models: r' = s * r + o where r' = (r + alpha), so:
            //   - depth_scale s = ||t|| (implicit in t magnitude)
            //   - depth_offset o = alpha (additive offset to inverse depth)
            // t_cam is the full translation vector (includes scale)
            t_cam = theta.segment<3>(3);
            double alpha = theta(6);

            // For output, normalize t to unit direction and record scale
            double t_norm = t_cam.norm();
            if (t_norm > 1e-8) {
                depth_scale = t_norm;
                t_cam = t_cam / t_norm;  // Normalize to unit direction
            }
            depth_offset = alpha;  // Direct offset on inverse depth
        } else {
            // Standard mode
            t_cam = theta.tail<3>();
        }
    }

    // Convert rotation vector to matrix (camera rotation)
    double angle = omega.norm();
    Eigen::Matrix3d R_cam;
    if (angle < 1e-8) {
        R_cam = Eigen::Matrix3d::Identity();
    } else {
        Eigen::Vector3d axis = omega / angle;
        Eigen::Matrix3d K;
        K <<      0,  -axis(2),   axis(1),
              axis(2),        0,  -axis(0),
             -axis(1),   axis(0),        0;

        R_cam = Eigen::Matrix3d::Identity() + std::sin(angle) * K + (1.0 - std::cos(angle)) * K * K;
    }

    // Convert from camera motion to standard point transformation convention
    // Standard: p_curr = R @ p_prev + t (transforms points from prev to curr frame)
    // Motion field gives camera motion (R_cam, t_cam) where:
    //   - R_cam: camera rotation from frame 0 to frame 1
    //   - t_cam: camera translation direction in frame 0 coords
    // Point transformation: R = R_cam^T, t = -R_cam^T @ t_cam
    Eigen::Matrix3d R = R_cam.transpose();
    Eigen::Vector3d t = -(R_cam.transpose() * t_cam);


    // ========== OPTIMIZED: Compute threshold using SAMPLING, then collect inliers from ALL ==========
    int n_all = static_cast<int>(all_coords_normalized.size());
    // Note: Use camera motion (omega, t_cam) for motion field residual computation
    // omega and t_cam are already set from either translation-only or full estimation above
    double omega0 = omega(0), omega1 = omega(1), omega2 = omega(2);

    // Use t_cam directly for residual computation
    // Note: For constrained depth_scale_mode == 2 (full estimation only), alpha was already
    // incorporated into the estimation. For translation-only mode, alpha_est = 0.
    double alpha_est = depth_offset;  // depth_offset = alpha for constrained mode, 0 otherwise
    double t0 = t_cam(0), t1 = t_cam(1), t2 = t_cam(2);

    // Lambda to compute residual and angle error for a single point (inline)
    auto compute_residual_angle = [&](int i, double& out_residual, double& out_angle, double& out_obs_mag) {
        double x = all_coords_normalized[i](0);
        double y = all_coords_normalized[i](1);
        double r = all_inv_depths_vec[i];
        double x2 = x * x, y2 = y * y, xy = x * y;

        // Constrained mode: flow = B * omega + (r + alpha) * A * t
        // A = [[-1, 0, x], [0, -1, y]]
        double r_eff = (cfg_.depth_scale_mode == 2) ? (r + alpha_est) : r;
        double A_t_0 = -t0 + x * t2;
        double A_t_1 = -t1 + y * t2;

        double pred_n0 = xy * omega0 - (1.0 + x2) * omega1 + y * omega2 + r_eff * A_t_0;
        double pred_n1 = (1.0 + y2) * omega0 - xy * omega1 - x * omega2 + r_eff * A_t_1;

        double obs_n0 = all_flow_normalized[i](0);
        double obs_n1 = all_flow_normalized[i](1);

        double err0 = pred_n0 - obs_n0;
        double err1 = pred_n1 - obs_n1;
        out_residual = std::sqrt(err0 * err0 + err1 * err1);

        // Pixel space for angle
        double pred_px0 = pred_n0 * fx, pred_px1 = pred_n1 * fy;
        double obs_px0 = obs_n0 * fx, obs_px1 = obs_n1 * fy;
        out_obs_mag = std::sqrt(obs_px0 * obs_px0 + obs_px1 * obs_px1);

        if (out_obs_mag > cfg_.min_flow_px) {
            double pred_mag = std::sqrt(pred_px0 * pred_px0 + pred_px1 * pred_px1) + 1e-12;
            double dot = (pred_px0 * obs_px0 + pred_px1 * obs_px1) / (pred_mag * out_obs_mag);
            dot = std::max(-1.0, std::min(1.0, dot));
            out_angle = std::acos(dot) * 180.0 / M_PI;
        } else {
            out_angle = 0.0;
        }
    };

    // ===== Step 1: Sample points for threshold estimation =====
    const int threshold_sample_size = std::min(50000, n_all);
    std::vector<int> sample_indices(n_all);
    std::iota(sample_indices.begin(), sample_indices.end(), 0);

    if (n_all > threshold_sample_size) {
        std::mt19937 gen_sample(cfg_.seed == 0 ? 12345 : cfg_.seed + 100);
        std::shuffle(sample_indices.begin(), sample_indices.end(), gen_sample);
        sample_indices.resize(threshold_sample_size);
    }

    // Compute residuals and angle errors on sampled points only
    std::vector<double> sample_residuals(sample_indices.size());
    std::vector<double> sample_angle_errors;
    sample_angle_errors.reserve(sample_indices.size());

    for (size_t s = 0; s < sample_indices.size(); ++s) {
        int i = sample_indices[s];
        double residual, angle, obs_mag;
        compute_residual_angle(i, residual, angle, obs_mag);
        sample_residuals[s] = residual;
        if (obs_mag > cfg_.min_flow_px) {
            sample_angle_errors.push_back(angle);
        }
    }

    // Compute adaptive threshold from samples
    int n_sample = static_cast<int>(sample_residuals.size());
    std::nth_element(sample_residuals.begin(), sample_residuals.begin() + n_sample / 2, sample_residuals.end());
    double median_res = sample_residuals[n_sample / 2];

    std::vector<double> abs_dev(n_sample);
    for (int s = 0; s < n_sample; ++s) {
        abs_dev[s] = std::abs(sample_residuals[s] - median_res);
    }
    std::nth_element(abs_dev.begin(), abs_dev.begin() + n_sample / 2, abs_dev.end());
    double mad = abs_dev[n_sample / 2] + 1e-8;
    double thresh = median_res + cfg_.mad_scale * mad;

    // Compute angle threshold from samples
    double angle_thresh_final = 45.0;
    if (sample_angle_errors.size() > 10) {
        angle_thresh_final = compute_mad_threshold(sample_angle_errors, 3.0);
        angle_thresh_final = std::max(15.0, std::min(angle_thresh_final, 60.0));
    }

    // Prepare result
    MotionFieldResult result;
    result.R = R;
    result.t = t;
    result.omega = -omega;  // Negated to match R (R = exp(-omega_cam))
    result.num_points_used = static_cast<int>(coords_normalized.size());  // Sampled points used for RANSAC
    result.inlier_mask = cv::Mat::zeros(H, W, CV_8U);
    result.flow_refined = flow.clone();
    result.depth_scale = depth_scale;
    result.depth_offset = depth_offset;

    // Collect inlier matches from ALL valid pixels (with angle gate)
    // Flow fusion (Supplementary 2.5): outliers are replaced with motion-field prediction
    int num_inliers = 0;
    double inlier_res_sum = 0.0;

    // Reserve space for expected inliers (~80% typical)
    result.u0.reserve(n_all * 8 / 10);
    result.v0.reserve(n_all * 8 / 10);
    result.u1.reserve(n_all * 8 / 10);
    result.v1.reserve(n_all * 8 / 10);

    // Lambda to compute motion-field predicted flow for a pixel
    auto compute_predicted_flow = [&](int i, float& pred_u, float& pred_v) {
        double x = all_coords_normalized[i](0);
        double y = all_coords_normalized[i](1);
        double r = all_inv_depths_vec[i];
        double x2 = x * x, y2 = y * y, xy = x * y;

        // Constrained mode: flow = B * omega + (r + alpha) * A * t
        double r_eff = (cfg_.depth_scale_mode == 2) ? (r + alpha_est) : r;
        double A_t_0 = -t0 + x * t2;
        double A_t_1 = -t1 + y * t2;

        double pred_n0 = xy * omega0 - (1.0 + x2) * omega1 + y * omega2 + r_eff * A_t_0;
        double pred_n1 = (1.0 + y2) * omega0 - xy * omega1 - x * omega2 + r_eff * A_t_1;

        // Convert to pixel flow
        pred_u = static_cast<float>(pred_n0 * fx);
        pred_v = static_cast<float>(pred_n1 * fy);
    };

    for (int i = 0; i < n_all; ++i) {
        double residual, angle, obs_mag;
        compute_residual_angle(i, residual, angle, obs_mag);

        // Check residual threshold
        bool res_ok = (residual <= thresh);

        // Check angle gate for flows with sufficient magnitude
        bool angle_ok = true;
        if (obs_mag > cfg_.min_flow_px) {
            angle_ok = (angle <= angle_thresh_final);
        }

        int px = static_cast<int>(all_pixel_x[i]);
        int py = static_cast<int>(all_pixel_y[i]);

        if (res_ok && angle_ok) {
            // Inlier: keep observed flow
            result.u0.push_back(all_pixel_x[i]);
            result.v0.push_back(all_pixel_y[i]);
            result.u1.push_back(all_pixel_x[i] + all_flow_u[i]);
            result.v1.push_back(all_pixel_y[i] + all_flow_v[i]);
            num_inliers++;
            inlier_res_sum += residual;

            if (px >= 0 && px < W && py >= 0 && py < H) {
                result.inlier_mask.ptr<uint8_t>(py)[px] = 1;
            }
        } else {
            // Outlier: replace with motion-field prediction (Supplementary 2.5)
            if (px >= 0 && px < W && py >= 0 && py < H) {
                float pred_u, pred_v;
                compute_predicted_flow(i, pred_u, pred_v);
                result.flow_refined.ptr<cv::Vec2f>(py)[px] = cv::Vec2f(pred_u, pred_v);
            }
        }
    }

    result.num_inliers = num_inliers;
    result.mean_residual = num_inliers > 0 ?
        static_cast<float>(inlier_res_sum / num_inliers) : 0.0f;

    return result;
}

} // namespace pr_depth
