#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace pr_depth {

/**
 * Configuration for triangulation
 */
struct TriangulationConfig {
    float fill_value = NAN;  // Fill value for invalid pixels
    bool debug = false;
};

/**
 * Result from triangulation
 */
struct TriangulationResult {
    cv::Mat z1_tri;       // Triangulated depth map (H, W) float32
    cv::Mat rpx_tri;      // Reprojection error map (H, W) float32
    int num_valid = 0;    // Number of valid triangulated points
};

/**
 * Triangulator class for computing depth from stereo correspondences
 */
class Triangulator {
public:
    /**
     * Constructor
     * @param H Image height
     * @param W Image width
     * @param fx Focal length X
     * @param fy Focal length Y
     * @param cx Principal point X
     * @param cy Principal point Y
     * @param config Triangulation configuration
     */
    Triangulator(int H, int W, float fx, float fy, float cx, float cy,
                 const TriangulationConfig& config = TriangulationConfig());

    /**
     * Triangulate depth from optical flow (cv::Mat version)
     * @param u0 Source x coordinates (H, W) float32
     * @param v0 Source y coordinates (H, W) float32
     * @param u1 Target x coordinates (H, W) float32
     * @param v1 Target y coordinates (H, W) float32
     * @param R Rotation matrix from frame 0 to frame 1 (3x3)
     * @param t Translation vector from frame 0 to frame 1 (3x1)
     * @return TriangulationResult with depth map and errors
     */
    TriangulationResult triangulate(
        const cv::Mat& u0, const cv::Mat& v0,
        const cv::Mat& u1, const cv::Mat& v1,
        const Eigen::Matrix3d& R,
        const Eigen::Vector3d& t
    );

    /**
     * Triangulate depth from optical flow (vector version - more efficient)
     * Avoids cv::Mat conversion overhead when input is already vectorized.
     */
    TriangulationResult triangulate_vec(
        const std::vector<float>& u0,
        const std::vector<float>& v0,
        const std::vector<float>& u1,
        const std::vector<float>& v1,
        const Eigen::Matrix3d& R,
        const Eigen::Vector3d& t
    );

private:
    // Camera parameters
    int H_, W_;
    float fx_, fy_, cx_, cy_;
    TriangulationConfig config_;

    // Camera matrices
    Eigen::Matrix3d K_;      // Intrinsic matrix
    Eigen::Matrix3d K_inv_;  // Inverse intrinsic matrix

    /**
     * Core triangulation: compute depths from ray-ray intersection
     * @param r0 Ray directions in frame 0 (Nx3)
     * @param r1 Ray directions in frame 1 (Nx3)
     * @param dt Translation vector
     * @param dR Rotation matrix
     * @param u1 Target u coordinates for validity check
     * @param v1 Target v coordinates for validity check
     * @param rho0 Output: depths in frame 0
     * @param rho1 Output: depths in frame 1
     * @param valid Output: mask of valid triangulations
     */
    void triangulate_core(
        const Eigen::MatrixXd& r0,
        const Eigen::MatrixXd& r1,
        const Eigen::Vector3d& dt,
        const Eigen::Matrix3d& dR,
        const std::vector<float>& u1,
        const std::vector<float>& v1,
        std::vector<double>& rho0,
        std::vector<double>& rho1,
        std::vector<bool>& valid
    );

    /**
     * Project 3D points to image coordinates
     * @param P0 3D points in camera 0 (Nx3)
     * @param u Output: x image coordinates
     * @param v Output: y image coordinates
     */
    void project_points(
        const Eigen::MatrixXd& P0,
        std::vector<float>& u,
        std::vector<float>& v
    );

    /**
     * Z-buffer splatting: render depth and error maps
     * @param P0 3D points in frame 0 (Nx3)
     * @param dt Translation vector
     * @param dR Rotation matrix
     * @param u0 Source u coordinates
     * @param v0 Source v coordinates
     * @param u1 Target u coordinates
     * @param v1 Target v coordinates
     * @param valid Mask of valid points
     * @param z_out Output depth map
     * @param e_out Output error map
     */
    void splat_zbuffer(
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
    );
};

}  // namespace pr_depth
