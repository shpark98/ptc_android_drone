#include "pr_depth/triangulation.hpp"
#include <cmath>
#include <limits>
#include <iostream>

namespace pr_depth {

Triangulator::Triangulator(
    int H, int W, float fx, float fy, float cx, float cy,
    const TriangulationConfig& config
) : H_(H), W_(W), fx_(fx), fy_(fy), cx_(cx), cy_(cy), config_(config) {
    // Build intrinsic matrix
    K_ << fx, 0, cx,
          0, fy, cy,
          0,  0,  1;
    K_inv_ = K_.inverse();
}

// Helper: skew-symmetric matrix from vector
static Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S << 0, -v(2), v(1),
         v(2), 0, -v(0),
         -v(1), v(0), 0;
    return S;
}

// Compute fundamental matrix F = K^{-T} [t]_x R K^{-1}
Eigen::Matrix3d compute_fundamental(
    const Eigen::Matrix3d& K_inv,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t
) {
    return K_inv.transpose() * skew(t) * R * K_inv;
}

// Compute Sampson error (squared) for a batch of point correspondences
void sampson_error_batch(
    const std::vector<float>& u0,
    const std::vector<float>& v0,
    const std::vector<float>& u1,
    const std::vector<float>& v1,
    const Eigen::Matrix3d& F,
    std::vector<double>& errors
) {
    int N = u0.size();
    errors.resize(N);

    for (int i = 0; i < N; ++i) {
        // Homogeneous coordinates
        Eigen::Vector3d x0(u0[i], v0[i], 1.0);
        Eigen::Vector3d x1(u1[i], v1[i], 1.0);

        // Fx0 and F^T x1
        Eigen::Vector3d Fx0 = F * x0;
        Eigen::Vector3d Ftx1 = F.transpose() * x1;

        // Numerator: (x1^T F x0)^2
        double d = x1.dot(Fx0);
        double num = d * d;

        // Denominator: ||Fx0||^2_[0:2] + ||F^T x1||^2_[0:2]
        double denom = Fx0(0) * Fx0(0) + Fx0(1) * Fx0(1) +
                       Ftx1(0) * Ftx1(0) + Ftx1(1) * Ftx1(1);

        // Sampson error (squared)
        errors[i] = num / std::max(denom, 1e-12);
    }
}

void Triangulator::triangulate_core(
    const Eigen::MatrixXd& r0,
    const Eigen::MatrixXd& r1,
    const Eigen::Vector3d& dt,
    const Eigen::Matrix3d& dR,
    const std::vector<float>& u1,
    const std::vector<float>& v1,
    std::vector<double>& rho0,
    std::vector<double>& rho1,
    std::vector<bool>& valid
) {
    int N = r0.rows();
    rho0.resize(N);
    rho1.resize(N);
    valid.resize(N);

    const double eps = 1e-9;

    for (int i = 0; i < N; ++i) {
        // Get rays
        Eigen::Vector3d r0_i = r0.row(i).transpose();
        Eigen::Vector3d r1_i = r1.row(i).transpose();

        // Compute coefficients for ray-ray intersection
        // min || r0 * rho0 - (r1 * rho1 + dt) ||^2
        double a = r0_i.squaredNorm();
        double d = r1_i.squaredNorm();
        double r01 = r0_i.dot(r1_i);
        double b = -r01;
        double c = -r01;
        double e0 = r0_i.dot(dt);
        double e1 = -r1_i.dot(dt);

        double det = a * d - r01 * r01;

        // Check determinant
        if (std::abs(det) <= eps) {
            rho0[i] = std::numeric_limits<double>::quiet_NaN();
            rho1[i] = std::numeric_limits<double>::quiet_NaN();
            valid[i] = false;
            continue;
        }

        // Solve 2x2 system
        double rho0_val = (d * e0 - b * e1) / det;
        double rho1_val = (-c * e0 + a * e1) / det;

        // Check validity
        bool is_valid = (
            std::isfinite(rho0_val) &&
            std::isfinite(rho1_val) &&
            rho0_val > 0 &&
            rho1_val > 0
        );

        // Check if projection lands in image
        if (is_valid) {
            int ui = static_cast<int>(std::round(u1[i]));
            int vi = static_cast<int>(std::round(v1[i]));
            if (ui < 0 || ui >= W_ || vi < 0 || vi >= H_) {
                is_valid = false;
            }
        }

        rho0[i] = rho0_val;
        rho1[i] = rho1_val;
        valid[i] = is_valid;
    }
}

void Triangulator::project_points(
    const Eigen::MatrixXd& P0,
    std::vector<float>& u,
    std::vector<float>& v
) {
    int N = P0.rows();
    u.resize(N);
    v.resize(N);

    for (int i = 0; i < N; ++i) {
        double Z = P0(i, 2);
        double invZ = 1.0 / std::max(Z, 1e-12);
        u[i] = static_cast<float>(fx_ * P0(i, 0) * invZ + cx_);
        v[i] = static_cast<float>(fy_ * P0(i, 1) * invZ + cy_);
    }
}

void Triangulator::splat_zbuffer(
    const Eigen::MatrixXd& P0,
    const Eigen::Vector3d& dt,
    const Eigen::Matrix3d& dR,
    const std::vector<float>& u0,
    const std::vector<float>& v0,
    const std::vector<float>& u1,
    const std::vector<float>& v1,
    const std::vector<bool>& valid,
    cv::Mat& z_out,
    cv::Mat& e_out
) {
    int N = P0.rows();

    // Initialize output maps
    z_out = cv::Mat(H_, W_, CV_32F, cv::Scalar(config_.fill_value));
    e_out = cv::Mat(H_, W_, CV_32F, cv::Scalar(config_.fill_value));

    // Compute fundamental matrix for Sampson error
    Eigen::Matrix3d F = compute_fundamental(K_inv_, dR, dt);

    // Compute Sampson errors for all correspondences
    std::vector<double> sampson_errors;
    sampson_error_batch(u0, v0, u1, v1, F, sampson_errors);

    // Transform points to frame 1
    Eigen::MatrixXd P1 = (dR * (P0.transpose().colwise() - dt)).transpose();

    // Two-pass Z-buffer (matches Python's np.maximum.at)
    struct SplatPoint {
        int flat_idx;
        float invZ;
        float depth;
        float error;
    };
    std::vector<SplatPoint> points;
    points.reserve(N);

    // Pass 1: Collect all valid points
    for (int i = 0; i < N; ++i) {
        if (!valid[i]) continue;

        double Z1 = P1(i, 2);
        if (!std::isfinite(Z1) || Z1 <= 1e-8) continue;

        int ui = static_cast<int>(std::round(u1[i]));
        int vi = static_cast<int>(std::round(v1[i]));

        if (ui < 0 || ui >= W_ || vi < 0 || vi >= H_) continue;

        float invZ = static_cast<float>(1.0 / Z1);
        float err = static_cast<float>(std::sqrt(sampson_errors[i]));
        int flat_idx = vi * W_ + ui;

        points.push_back({flat_idx, invZ, static_cast<float>(Z1), err});
    }

    if (points.empty()) return;

    // Pass 2: Find maximum invZ for each pixel (like np.maximum.at)
    std::vector<float> max_invZ(H_ * W_, -std::numeric_limits<float>::infinity());
    for (const auto& p : points) {
        max_invZ[p.flat_idx] = std::max(max_invZ[p.flat_idx], p.invZ);
    }

    // Pass 3: Splat only points with maximum invZ (>= for tie-breaking, keeps last)
    for (const auto& p : points) {
        if (p.invZ >= max_invZ[p.flat_idx]) {
            int r = p.flat_idx / W_;
            int c = p.flat_idx % W_;
            z_out.ptr<float>(r)[c] = p.depth;
            e_out.ptr<float>(r)[c] = p.error;
        }
    }
}

TriangulationResult Triangulator::triangulate(
    const cv::Mat& u0, const cv::Mat& v0,
    const cv::Mat& u1, const cv::Mat& v1,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t
) {
    TriangulationResult result;

    // Check inputs
    if (u0.size() != v0.size() || u0.size() != u1.size() || u0.size() != v1.size()) {
        throw std::runtime_error("Input coordinate arrays must have same size");
    }

    if (u0.type() != CV_32F || v0.type() != CV_32F ||
        u1.type() != CV_32F || v1.type() != CV_32F) {
        throw std::runtime_error("Input coordinates must be float32");
    }

    int N = u0.rows * u0.cols;

    // Flatten input coordinates (row-major order)
    std::vector<float> u0_flat, v0_flat, u1_flat, v1_flat;
    u0_flat.reserve(N);
    v0_flat.reserve(N);
    u1_flat.reserve(N);
    v1_flat.reserve(N);

    for (int r = 0; r < u0.rows; ++r) {
        const float* u0_row = u0.ptr<float>(r);
        const float* v0_row = v0.ptr<float>(r);
        const float* u1_row = u1.ptr<float>(r);
        const float* v1_row = v1.ptr<float>(r);

        for (int c = 0; c < u0.cols; ++c) {
            u0_flat.push_back(u0_row[c]);
            v0_flat.push_back(v0_row[c]);
            u1_flat.push_back(u1_row[c]);
            v1_flat.push_back(v1_row[c]);
        }
    }

    // Compute ray directions in frame 0 from (u0, v0)
    Eigen::MatrixXd r0(N, 3);
    for (int i = 0; i < N; ++i) {
        r0(i, 0) = (u0_flat[i] - cx_) / fx_;
        r0(i, 1) = (v0_flat[i] - cy_) / fy_;
        r0(i, 2) = 1.0;
    }

    // Compute ray directions in frame 1 from (u1, v1), then rotate by R^T
    // Python: r1 = np.stack([(u1-cx)/fx, (v1-cy)/fy, 1]) @ R.T
    Eigen::MatrixXd r1_base(N, 3);
    for (int i = 0; i < N; ++i) {
        r1_base(i, 0) = (u1_flat[i] - cx_) / fx_;
        r1_base(i, 1) = (v1_flat[i] - cy_) / fy_;
        r1_base(i, 2) = 1.0;
    }
    Eigen::MatrixXd r1 = (r1_base * R.transpose()).eval();  // r1 = r1_base @ R^T

    // Perform triangulation
    std::vector<double> rho0, rho1;
    std::vector<bool> valid;
    triangulate_core(r0, r1, t, R, u1_flat, v1_flat, rho0, rho1, valid);

    // Build 3D points in frame 0
    std::vector<int> valid_indices;
    for (int i = 0; i < N; ++i) {
        if (valid[i]) {
            valid_indices.push_back(i);
        }
    }

    int num_valid = valid_indices.size();
    result.num_valid = num_valid;

    if (num_valid == 0) {
        // No valid points
        result.z1_tri = cv::Mat(H_, W_, CV_32F, cv::Scalar(config_.fill_value));
        result.rpx_tri = cv::Mat(H_, W_, CV_32F, cv::Scalar(config_.fill_value));
        return result;
    }

    // Extract valid points
    Eigen::MatrixXd P0_valid(num_valid, 3);
    std::vector<float> u0_valid(num_valid), v0_valid(num_valid);
    std::vector<float> u1_valid(num_valid), v1_valid(num_valid);
    std::vector<bool> valid_subset(num_valid, true);

    for (int i = 0; i < num_valid; ++i) {
        int idx = valid_indices[i];
        P0_valid.row(i) = r0.row(idx) * rho0[idx];
        u0_valid[i] = u0_flat[idx];
        v0_valid[i] = v0_flat[idx];
        u1_valid[i] = u1_flat[idx];
        v1_valid[i] = v1_flat[idx];
    }

    // Splat to depth map using Z-buffer
    splat_zbuffer(P0_valid, t, R, u0_valid, v0_valid, u1_valid, v1_valid,
                  valid_subset, result.z1_tri, result.rpx_tri);

    return result;
}

// ===== OPTIMIZED: Vector-based triangulation (skips cv::Mat conversion) =====
TriangulationResult Triangulator::triangulate_vec(
    const std::vector<float>& u0,
    const std::vector<float>& v0,
    const std::vector<float>& u1,
    const std::vector<float>& v1,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t
) {
    TriangulationResult result;
    int N = static_cast<int>(u0.size());

    if (N == 0) {
        result.z1_tri = cv::Mat(H_, W_, CV_32F, cv::Scalar(config_.fill_value));
        result.rpx_tri = cv::Mat(H_, W_, CV_32F, cv::Scalar(config_.fill_value));
        return result;
    }

    // Ray transformation: r1_transformed = R @ r1_base
    // Motion field convention: p_curr = R @ p_prev + t
    // We compute R @ [x, y, 1]^T by extracting elements as follows:
    // r1x = R(0,0)*x + R(0,1)*y + R(0,2)  <- row 0 of R
    // r1y = R(1,0)*x + R(1,1)*y + R(1,2)  <- row 1 of R
    // r1z = R(2,0)*x + R(2,1)*y + R(2,2)  <- row 2 of R
    //
    // The computation pattern is: r1i = rX0*x + rX1*y + rX2
    // So we map: r00=R(0,0), r10=R(0,1), r20=R(0,2) for r1x
    //            r01=R(1,0), r11=R(1,1), r21=R(1,2) for r1y
    //            r02=R(2,0), r12=R(2,1), r22=R(2,2) for r1z
    double r00 = R(0, 0), r10 = R(0, 1), r20 = R(0, 2);
    double r01 = R(1, 0), r11 = R(1, 1), r21 = R(1, 2);
    double r02 = R(2, 0), r12 = R(2, 1), r22 = R(2, 2);
    double tx = t(0), ty = t(1), tz = t(2);
    float fx_inv = 1.0f / fx_, fy_inv = 1.0f / fy_;

    // Compute fundamental matrix for Sampson error
    Eigen::Matrix3d F = compute_fundamental(K_inv_, R, t);

    // ===== MERGED LOOP: ray computation + triangulation + validity check =====
    // Store results compactly for valid points only
    struct ValidPoint {
        float u1_px, v1_px;  // Target pixel coords
        float depth;         // Depth in frame 1
        float sampson_err;   // Sampson error
    };
    std::vector<ValidPoint> valid_points;
    valid_points.reserve(N);

    const double eps = 1e-9;

    for (int i = 0; i < N; ++i) {
        // Compute ray directions inline
        double x0 = (u0[i] - cx_) * fx_inv;
        double y0 = (v0[i] - cy_) * fy_inv;
        // r0 = [x0, y0, 1]

        double x1_base = (u1[i] - cx_) * fx_inv;
        double y1_base = (v1[i] - cy_) * fy_inv;
        // r1_base = [x1_base, y1_base, 1], then r1 = r1_base @ R
        // For row vector [x,y,1] @ R: result[j] = x*R(0,j) + y*R(1,j) + R(2,j)
        double r1x = x1_base * r00 + y1_base * r10 + r20;
        double r1y = x1_base * r01 + y1_base * r11 + r21;
        double r1z = x1_base * r02 + y1_base * r12 + r22;

        // Ray-ray intersection: min || r0 * rho0 - (r1 * rho1 + t) ||^2
        double a = x0 * x0 + y0 * y0 + 1.0;  // r0.squaredNorm()
        double d = r1x * r1x + r1y * r1y + r1z * r1z;  // r1.squaredNorm()
        double r01_dot = x0 * r1x + y0 * r1y + r1z;  // r0.dot(r1)
        double e0 = x0 * tx + y0 * ty + tz;  // r0.dot(t)
        double e1 = -(r1x * tx + r1y * ty + r1z * tz);  // -r1.dot(t)

        double det = a * d - r01_dot * r01_dot;
        if (std::abs(det) <= eps) continue;

        double rho0 = (d * e0 + r01_dot * e1) / det;
        double rho1 = (r01_dot * e0 + a * e1) / det;

        if (!std::isfinite(rho0) || !std::isfinite(rho1) || rho0 <= 0 || rho1 <= 0) continue;

        // Check target pixel bounds
        int ui = static_cast<int>(std::round(u1[i]));
        int vi = static_cast<int>(std::round(v1[i]));
        if (ui < 0 || ui >= W_ || vi < 0 || vi >= H_) continue;

        // Compute 3D point in frame 0, then transform to frame 1
        double P0x = x0 * rho0, P0y = y0 * rho0, P0z = rho0;
        // P1 = R * (P0 - t)
        double dx = P0x - tx, dy = P0y - ty, dz = P0z - tz;
        double P1z = R(2, 0) * dx + R(2, 1) * dy + R(2, 2) * dz;

        if (!std::isfinite(P1z) || P1z <= 1e-8) continue;

        // Compute Sampson error inline
        Eigen::Vector3d pt0(u0[i], v0[i], 1.0);
        Eigen::Vector3d pt1(u1[i], v1[i], 1.0);
        Eigen::Vector3d Fx0 = F * pt0;
        Eigen::Vector3d Ftx1 = F.transpose() * pt1;
        double d_epipolar = pt1.dot(Fx0);
        double denom = Fx0(0) * Fx0(0) + Fx0(1) * Fx0(1) + Ftx1(0) * Ftx1(0) + Ftx1(1) * Ftx1(1);
        float sampson = static_cast<float>(std::sqrt((d_epipolar * d_epipolar) / std::max(denom, 1e-12)));

        valid_points.push_back({u1[i], v1[i], static_cast<float>(P1z), sampson});
    }

    result.num_valid = static_cast<int>(valid_points.size());

    if (valid_points.empty()) {
        result.z1_tri = cv::Mat(H_, W_, CV_32F, cv::Scalar(config_.fill_value));
        result.rpx_tri = cv::Mat(H_, W_, CV_32F, cv::Scalar(config_.fill_value));
        return result;
    }

    // ===== Z-buffer splatting (optimized 2-pass) =====
    result.z1_tri = cv::Mat(H_, W_, CV_32F, cv::Scalar(config_.fill_value));
    result.rpx_tri = cv::Mat(H_, W_, CV_32F, cv::Scalar(config_.fill_value));

    // Pass 1: Find max invZ per pixel
    std::vector<float> max_invZ(H_ * W_, -std::numeric_limits<float>::infinity());
    for (const auto& p : valid_points) {
        int ui = static_cast<int>(std::round(p.u1_px));
        int vi = static_cast<int>(std::round(p.v1_px));
        int idx = vi * W_ + ui;
        float invZ = 1.0f / p.depth;
        if (invZ > max_invZ[idx]) {
            max_invZ[idx] = invZ;
        }
    }

    // Pass 2: Splat points with max invZ
    for (const auto& p : valid_points) {
        int ui = static_cast<int>(std::round(p.u1_px));
        int vi = static_cast<int>(std::round(p.v1_px));
        int idx = vi * W_ + ui;
        float invZ = 1.0f / p.depth;
        if (invZ >= max_invZ[idx]) {
            result.z1_tri.ptr<float>(vi)[ui] = p.depth;
            result.rpx_tri.ptr<float>(vi)[ui] = p.sampson_err;
        }
    }

    return result;
}

}  // namespace pr_depth
