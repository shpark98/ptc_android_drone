#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc/segmentation.hpp>
#include <iostream>
#include "pr_depth/optical_flow.hpp"
#include "pr_depth/motion_field.hpp"
#include "pr_depth/triangulation.hpp"
#include "pr_depth/depth_fusion.hpp"
#include "pr_depth/depth_refinement.hpp"

namespace py = pybind11;

// Utility: Convert numpy array to cv::Mat
cv::Mat numpy_to_mat(py::array_t<uint8_t> arr) {
    py::buffer_info buf = arr.request();

    if (buf.ndim == 2) {
        // Grayscale HxW
        return cv::Mat(buf.shape[0], buf.shape[1], CV_8U, buf.ptr);
    } else if (buf.ndim == 3 && buf.shape[2] == 3) {
        // Color HxWx3
        return cv::Mat(buf.shape[0], buf.shape[1], CV_8UC3, buf.ptr);
    } else {
        throw std::runtime_error("Unsupported array shape for image");
    }
}

// Utility: Convert cv::Mat to numpy array
py::array_t<float> mat_to_numpy(const cv::Mat& mat) {
    if (mat.type() != CV_32FC2) {
        throw std::runtime_error("Only CV_32FC2 flow supported");
    }

    // Return HxWx2 numpy array
    return py::array_t<float>(
        {mat.rows, mat.cols, 2},                    // shape
        {mat.step[0], mat.step[1], sizeof(float)},  // strides
        (float*)mat.data                             // data pointer
    );
}

// Minimal binding: compute_optical_flow(img0, img1, preset, finest_scale) -> flow
py::array_t<float> compute_optical_flow_binding(
    py::array_t<uint8_t> img_prev,
    py::array_t<uint8_t> img_curr,
    int preset = 2,  // cv::DISOpticalFlow::PRESET_MEDIUM
    int finest_scale = 0
) {
    // Convert numpy to cv::Mat (zero-copy view)
    cv::Mat cv_prev = numpy_to_mat(img_prev);
    cv::Mat cv_curr = numpy_to_mat(img_curr);

    // Create flow estimator
    pr_depth::OpticalFlowConfig cfg;
    cfg.preset = preset;
    cfg.finest_scale = finest_scale;

    pr_depth::OpticalFlowDIS flow_estimator(cfg);

    // Compute flow
    cv::Mat flow = flow_estimator.compute(cv_prev, cv_curr);

    // Convert back to numpy (copy, since lifetime of cv::Mat ends here)
    py::array_t<float> flow_np = mat_to_numpy(flow);

    // Force copy to ensure data persists after cv::Mat destruction
    return py::array_t<float>(flow_np.request());
}

// Motion field estimation binding with full config
py::dict estimate_motion_field(
    py::array_t<float, py::array::c_style | py::array::forcecast> flow,
    py::array_t<float, py::array::c_style | py::array::forcecast> inv_depth,
    float fx, float fy, float cx, float cy,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask = py::array_t<uint8_t>(),
    // Boolean flags
    bool use_ransac = true,
    // RANSAC parameters
    int ransac_max_iters = 500,
    int ransac_min_sample = 6,
    float ransac_thresh_ratio = 1.5f,
    float min_flow_px = 0.01f,
    // IRLS parameters
    int lo_irls_iters = 5,
    float huber_delta_rel = 1.0f,
    // Filtering
    float mad_scale = 3.5f,
    // Point sampling
    int max_points = 2000,
    int depth_bins = 3,
    float margin_x_pct = 0.05f,
    float margin_y_pct = 0.05f,
    float adaptive_flow_depth_scale = 0.0f,
    // Direction-based inlier criterion
    float cos_sim_thresh = 0.5f,
    bool use_direction_check = true,
    // Depth scale mode: 0=none, 1=scale only, 2=affine (scale+offset)
    int depth_scale_mode = 2,
    // Scoring method: true=MAGSAC++ soft weighting, false=MAD-based binary threshold (paper method)
    bool use_magsac_scoring = true,
    // MAGSAC++ parameters (only used when use_magsac_scoring=true)
    float magsac_rel_sigma_max = 0.2f,
    float magsac_inlier_weight = 0.5f,
    // Random seed
    unsigned int seed = 42
) {
    // Convert numpy arrays to cv::Mat
    // Note: py::array::c_style | py::array::forcecast ensures contiguous C-style arrays
    py::buffer_info flow_info = flow.request();
    py::buffer_info inv_depth_info = inv_depth.request();

    if (flow_info.ndim != 3 || flow_info.shape[2] != 2) {
        throw std::runtime_error("Flow must be HxWx2");
    }
    if (inv_depth_info.ndim != 2) {
        throw std::runtime_error("Inverse depth must be HxW");
    }

    int H = flow_info.shape[0];
    int W = flow_info.shape[1];

    cv::Mat flow_mat(H, W, CV_32FC2, flow_info.ptr);
    cv::Mat inv_depth_mat(H, W, CV_32F, inv_depth_info.ptr);

    cv::Mat mask_mat;
    if (mask.size() > 0) {
        py::buffer_info mask_info = mask.request();
        mask_mat = cv::Mat(H, W, CV_8U, mask_info.ptr);
    }

    // Create camera intrinsics
    pr_depth::CameraIntrinsics intrinsics(H, W, fx, fy, cx, cy);

    // Create config with all parameters
    pr_depth::MotionFieldConfig config;
    // Boolean flags
    config.use_ransac = use_ransac;
    // RANSAC parameters
    config.ransac_max_iters = ransac_max_iters;
    config.ransac_min_sample = ransac_min_sample;
    config.ransac_thresh_ratio = ransac_thresh_ratio;
    config.min_flow_px = min_flow_px;
    // IRLS parameters
    config.lo_irls_iters = lo_irls_iters;
    config.huber_delta_rel = huber_delta_rel;
    // Filtering
    config.mad_scale = mad_scale;
    // Point sampling
    config.max_points = max_points;
    config.depth_bins = depth_bins;
    config.margin_x_pct = margin_x_pct;
    config.margin_y_pct = margin_y_pct;
    config.adaptive_flow_depth_scale = adaptive_flow_depth_scale;
    // Direction-based inlier criterion
    config.cos_sim_thresh = cos_sim_thresh;
    config.use_direction_check = use_direction_check;
    // Depth scale mode
    config.depth_scale_mode = depth_scale_mode;
    // Scoring method toggle
    config.use_magsac_scoring = use_magsac_scoring;
    // MAGSAC++ parameters (only used when use_magsac_scoring=true)
    config.magsac_rel_sigma_max = magsac_rel_sigma_max;
    config.magsac_inlier_weight = magsac_inlier_weight;
    // Random seed
    config.seed = seed;

    // Estimate motion
    pr_depth::MotionFieldEstimator estimator(config);
    pr_depth::MotionFieldResult result = estimator.estimate(flow_mat, inv_depth_mat, intrinsics, mask_mat);

    // Convert results to Python
    py::dict output;

    // Rotation matrix (3x3)
    py::array_t<double> R_np({3, 3});
    auto R_ptr = R_np.mutable_unchecked<2>();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R_ptr(i, j) = result.R(i, j);
        }
    }
    output["R"] = R_np;

    // Translation vector (3,) - use pybind11's automatic Eigen conversion
    output["t"] = result.t;

    // Rotation vector (3,) - use pybind11's automatic Eigen conversion
    output["omega"] = result.omega;

    // Metadata
    output["num_inliers"] = result.num_inliers;
    output["num_points_used"] = result.num_points_used;
    output["mean_residual"] = result.mean_residual;

    // Depth scale/offset estimation (if depth_scale_mode > 0)
    output["depth_scale"] = result.depth_scale;
    output["depth_offset"] = result.depth_offset;

    // Inlier matches (like Python's pose_res['matches'])
    py::dict matches;
    size_t n_inliers = result.u0.size();

    // Create numpy arrays using the data from the vectors directly
    // This creates a copy that Python owns
    matches["u0"] = py::array_t<float>(
        {n_inliers},                           // shape
        {sizeof(float)},                       // strides
        result.u0.data()                       // data pointer (will be copied)
    );
    matches["v0"] = py::array_t<float>(
        {n_inliers},
        {sizeof(float)},
        result.v0.data()
    );
    matches["u1"] = py::array_t<float>(
        {n_inliers},
        {sizeof(float)},
        result.u1.data()
    );
    matches["v1"] = py::array_t<float>(
        {n_inliers},
        {sizeof(float)},
        result.v1.data()
    );
    output["matches"] = matches;

    // Inlier mask (H, W) - CV_8U, 1 for inlier, 0 for outlier
    int mask_H = result.inlier_mask.rows;
    int mask_W = result.inlier_mask.cols;
    py::array_t<uint8_t> inlier_mask_np({mask_H, mask_W});
    auto mask_ptr = inlier_mask_np.mutable_unchecked<2>();
    for (int y = 0; y < mask_H; ++y) {
        for (int x = 0; x < mask_W; ++x) {
            mask_ptr(y, x) = result.inlier_mask.at<uint8_t>(y, x);
        }
    }
    output["inlier_mask"] = inlier_mask_np;

    return output;
}

// Compute depth edge map (Sobel gradient magnitude)
cv::Mat compute_depth_edge(const cv::Mat& depth) {
    cv::Mat grad_x, grad_y;
    cv::Sobel(depth, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(depth, grad_y, CV_32F, 0, 1, 3);

    cv::Mat grad_mag;
    cv::magnitude(grad_x, grad_y, grad_mag);

    // Normalize to [0, 1]
    double minVal, maxVal;
    cv::minMaxLoc(grad_mag, &minVal, &maxVal);
    if (maxVal > minVal) {
        grad_mag = (grad_mag - minVal) / (maxVal - minVal);
    }

    return grad_mag;
}

// Build multi-channel guide image for edge-aware segmentation
cv::Mat build_segmentation_guide(const cv::Mat& img, const cv::Mat& inv_depth,
                                  const cv::Mat& sky_mask,
                                  float w_rgb = 1.0f, float w_depth = 0.5f, float w_edge = 0.3f) {
    const int H = img.rows;
    const int W = img.cols;

    // Convert to Lab color space
    cv::Mat lab;
    cv::cvtColor(img, lab, cv::COLOR_BGR2Lab);

    // Compute depth edge
    cv::Mat depth_edge = compute_depth_edge(inv_depth);

    // Build 5-channel guide: [L, a, b, depth, edge]
    // But GraphSegmentation only accepts 3 channels, so we use [a, b, depth_edge_weighted]
    cv::Mat guide(H, W, CV_8UC3);

    for (int r = 0; r < H; ++r) {
        const cv::Vec3b* lab_row = lab.ptr<cv::Vec3b>(r);
        const float* depth_row = inv_depth.ptr<float>(r);
        const float* edge_row = depth_edge.ptr<float>(r);
        cv::Vec3b* guide_row = guide.ptr<cv::Vec3b>(r);

        for (int c = 0; c < W; ++c) {
            // Lab a, b channels (already 0-255)
            float a = static_cast<float>(lab_row[c][1]);
            float b = static_cast<float>(lab_row[c][2]);

            // Depth normalized to 0-255
            float d = depth_row[c];
            if (!std::isfinite(d)) d = 0.0f;
            float d_norm = std::min(255.0f, std::max(0.0f, d * 255.0f));

            // Edge weighted
            float e = edge_row[c];
            if (!std::isfinite(e)) e = 0.0f;
            float e_weighted = std::min(255.0f, e * w_edge * 255.0f);

            // Combine: use Lab a,b and depth+edge mix
            guide_row[c][0] = static_cast<uint8_t>(a * w_rgb);
            guide_row[c][1] = static_cast<uint8_t>(b * w_rgb);
            guide_row[c][2] = static_cast<uint8_t>(std::min(255.0f, d_norm * w_depth + e_weighted));
        }
    }

    // Handle sky mask: set sky pixels to median of non-sky
    if (!sky_mask.empty()) {
        std::vector<cv::Vec3f> non_sky_vals;
        for (int r = 0; r < H; ++r) {
            const uint8_t* mask_row = sky_mask.ptr<uint8_t>(r);
            const cv::Vec3b* guide_row = guide.ptr<cv::Vec3b>(r);
            for (int c = 0; c < W; ++c) {
                if (mask_row[c] == 0) {
                    non_sky_vals.push_back(cv::Vec3f(guide_row[c][0], guide_row[c][1], guide_row[c][2]));
                }
            }
        }

        if (!non_sky_vals.empty()) {
            // Compute median for each channel
            cv::Vec3b median_val;
            for (int ch = 0; ch < 3; ++ch) {
                std::vector<float> vals;
                for (const auto& v : non_sky_vals) vals.push_back(v[ch]);
                std::nth_element(vals.begin(), vals.begin() + vals.size()/2, vals.end());
                median_val[ch] = static_cast<uint8_t>(vals[vals.size()/2]);
            }

            // Set sky pixels to median
            for (int r = 0; r < H; ++r) {
                const uint8_t* mask_row = sky_mask.ptr<uint8_t>(r);
                cv::Vec3b* guide_row = guide.ptr<cv::Vec3b>(r);
                for (int c = 0; c < W; ++c) {
                    if (mask_row[c] != 0) {
                        guide_row[c] = median_val;
                    }
                }
            }
        }
    }

    return guide;
}

// Felzenszwalb segmentation binding with optional depth-aware guide
py::dict compute_segmentation(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img,
    float sigma = 0.5f,
    float k = 300.0f,
    int min_size = 100,
    float downsample = 1.0f,
    py::array_t<float, py::array::c_style | py::array::forcecast> inv_depth = py::array_t<float>(),
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> sky_mask = py::array_t<uint8_t>(),
    float w_depth = 0.5f,
    float w_edge = 0.3f
) {
    py::buffer_info img_info = img.request();

    if (img_info.ndim != 3 || img_info.shape[2] != 3) {
        throw std::runtime_error("Image must be HxWx3 (BGR)");
    }

    int H = img_info.shape[0];
    int W = img_info.shape[1];

    cv::Mat img_mat(H, W, CV_8UC3, img_info.ptr);

    // Check if inv_depth is provided
    py::buffer_info depth_info = inv_depth.request();
    bool use_depth_guide = (depth_info.ndim == 2 && depth_info.shape[0] == H && depth_info.shape[1] == W);

    cv::Mat inv_depth_mat;
    cv::Mat sky_mask_mat;

    if (use_depth_guide) {
        inv_depth_mat = cv::Mat(H, W, CV_32F, depth_info.ptr);

        py::buffer_info mask_info = sky_mask.request();
        if (mask_info.ndim == 2 && mask_info.shape[0] == H && mask_info.shape[1] == W) {
            sky_mask_mat = cv::Mat(H, W, CV_8U, mask_info.ptr);
        }
    }

    // Downsample if needed
    cv::Mat img_for_seg;
    cv::Mat depth_for_seg;
    cv::Mat mask_for_seg;
    int seg_H = H, seg_W = W;

    if (downsample < 1.0f && downsample > 0.0f) {
        seg_W = static_cast<int>(W * downsample);
        seg_H = static_cast<int>(H * downsample);
        cv::resize(img_mat, img_for_seg, cv::Size(seg_W, seg_H), 0, 0, cv::INTER_AREA);
        if (use_depth_guide) {
            cv::resize(inv_depth_mat, depth_for_seg, cv::Size(seg_W, seg_H), 0, 0, cv::INTER_AREA);
            if (!sky_mask_mat.empty()) {
                cv::resize(sky_mask_mat, mask_for_seg, cv::Size(seg_W, seg_H), 0, 0, cv::INTER_NEAREST);
            }
        }
    } else {
        img_for_seg = img_mat;
        depth_for_seg = inv_depth_mat;
        mask_for_seg = sky_mask_mat;
    }

    // Build guide image (depth-aware if inv_depth provided, otherwise just RGB)
    cv::Mat guide;
    if (use_depth_guide) {
        guide = build_segmentation_guide(img_for_seg, depth_for_seg, mask_for_seg, 1.0f, w_depth, w_edge);
    } else {
        guide = img_for_seg;
    }

    // Apply Gaussian blur
    cv::Mat guide_blurred;
    if (sigma > 0) {
        int ksize = static_cast<int>(sigma * 4) | 1;
        ksize = std::max(ksize, 3);
        cv::GaussianBlur(guide, guide_blurred, cv::Size(ksize, ksize), sigma);
    } else {
        guide_blurred = guide;
    }

    // Create GraphSegmentation and process
    auto seg = cv::ximgproc::segmentation::createGraphSegmentation(sigma, k, min_size);
    cv::Mat labels_small;
    seg->processImage(guide_blurred, labels_small);

    // Upsample labels if downsampled
    cv::Mat labels;
    if (downsample < 1.0f && downsample > 0.0f) {
        cv::resize(labels_small, labels, cv::Size(W, H), 0, 0, cv::INTER_NEAREST);
    } else {
        labels = labels_small;
    }

    // Count number of segments
    double minVal, maxVal;
    cv::minMaxLoc(labels, &minVal, &maxVal);
    int num_segments = static_cast<int>(maxVal) + 1;

    // Convert to numpy
    py::array_t<int32_t> labels_np({H, W});
    auto labels_ptr = labels_np.mutable_unchecked<2>();
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            labels_ptr(y, x) = labels.at<int32_t>(y, x);
        }
    }

    py::dict output;
    output["labels"] = labels_np;
    output["num_segments"] = num_segments;
    output["use_depth_guide"] = use_depth_guide;

    return output;
}

PYBIND11_MODULE(pr_depth_cpp, m) {
    m.doc() = "PR-Depth C++ core library - minimal bindings for parity testing";

    // Expose DIS presets as constants (integer values from OpenCV)
    m.attr("PRESET_ULTRAFAST") = 0;  // cv::DISOpticalFlow::PRESET_ULTRAFAST
    m.attr("PRESET_FAST") = 1;       // cv::DISOpticalFlow::PRESET_FAST
    m.attr("PRESET_MEDIUM") = 2;     // cv::DISOpticalFlow::PRESET_MEDIUM

    m.def("compute_optical_flow", &compute_optical_flow_binding,
          "Compute optical flow using DIS",
          py::arg("img_prev"),
          py::arg("img_curr"),
          py::arg("preset") = 2,  // cv::DISOpticalFlow::PRESET_MEDIUM
          py::arg("finest_scale") = 0);

    m.def("compute_segmentation", &compute_segmentation,
          "Compute Felzenszwalb graph-based segmentation with optional depth-aware guide",
          py::arg("img"),
          py::arg("sigma") = 0.5f,
          py::arg("k") = 300.0f,
          py::arg("min_size") = 100,
          py::arg("downsample") = 1.0f,
          py::arg("inv_depth") = py::array_t<float>(),
          py::arg("sky_mask") = py::array_t<uint8_t>(),
          py::arg("w_depth") = 0.5f,
          py::arg("w_edge") = 0.3f);

    m.def("estimate_motion_field", &estimate_motion_field,
          "Estimate camera rotation and translation from optical flow",
          py::arg("flow"),
          py::arg("inv_depth"),
          py::arg("fx"),
          py::arg("fy"),
          py::arg("cx"),
          py::arg("cy"),
          py::arg("mask") = py::array_t<uint8_t>(),
          // Boolean flags
          py::arg("use_ransac") = true,
          // RANSAC parameters
          py::arg("ransac_max_iters") = 500,
          py::arg("ransac_min_sample") = 6,
          py::arg("ransac_thresh_ratio") = 1.5f,
          py::arg("min_flow_px") = 0.01f,
          // IRLS parameters
          py::arg("lo_irls_iters") = 5,
          py::arg("huber_delta_rel") = 1.0f,
          // Filtering
          py::arg("mad_scale") = 3.5f,
          // Point sampling
          py::arg("max_points") = 2000,
          py::arg("depth_bins") = 3,
          py::arg("margin_x_pct") = 0.05f,
          py::arg("margin_y_pct") = 0.05f,
          py::arg("adaptive_flow_depth_scale") = 0.0f,
          // Direction-based inlier criterion
          py::arg("cos_sim_thresh") = 0.5f,
          py::arg("use_direction_check") = true,
          // Depth scale mode: 0=none, 1=scale only, 2=affine (scale+offset)
          py::arg("depth_scale_mode") = 2,
          // Scoring method: true=MAGSAC++ soft weighting, false=MAD-based binary threshold (paper)
          py::arg("use_magsac_scoring") = true,
          // MAGSAC++ parameters (only used when use_magsac_scoring=true)
          py::arg("magsac_rel_sigma_max") = 0.2f,
          py::arg("magsac_inlier_weight") = 0.5f,
          // Random seed
          py::arg("seed") = 42);

    // Triangulation binding
    m.def("triangulate_depth",
          [](py::array_t<float> u0_np, py::array_t<float> v0_np,
             py::array_t<float> u1_np, py::array_t<float> v1_np,
             py::array_t<double> R_np, py::array_t<double> t_np,
             float fx, float fy, float cx, float cy,
             int H = 375, int W = 1242) -> py::dict {
              try {
                  // Convert R and t from numpy to Eigen
                  auto R_info = R_np.request();
                  auto t_info = t_np.request();

                  if (R_info.ndim != 2 || R_info.shape[0] != 3 || R_info.shape[1] != 3) {
                      throw std::runtime_error("R must be 3x3 array");
                  }
                  if (t_info.ndim != 1 || t_info.shape[0] != 3) {
                      throw std::runtime_error("t must be 3-element array");
                  }

                  double* R_data = (double*)R_info.ptr;
                  double* t_data = (double*)t_info.ptr;

                  // Handle non-contiguous numpy arrays by respecting strides
                  Eigen::Matrix3d R;
                  size_t row_stride = R_info.strides[0] / sizeof(double);
                  size_t col_stride = R_info.strides[1] / sizeof(double);
                  for (int i = 0; i < 3; ++i) {
                      for (int j = 0; j < 3; ++j) {
                          R(i, j) = R_data[i * row_stride + j * col_stride];
                      }
                  }

                  Eigen::Vector3d t;
                  for (int i = 0; i < 3; ++i) {
                      t(i) = t_data[i];
                  }

                  // Get input dimensions and handle both 1D (sparse) and 2D (dense)
                  auto u0_info = u0_np.request();
                  auto v0_info = v0_np.request();
                  auto u1_info = u1_np.request();
                  auto v1_info = v1_np.request();

                  cv::Mat u0, v0, u1, v1;

                  if (u0_info.ndim == 1) {
                      // 1D sparse matches (N,) - use provided H, W for output
                      int N = u0_info.shape[0];
                      if (v0_info.shape[0] != N || u1_info.shape[0] != N || v1_info.shape[0] != N) {
                          throw std::runtime_error("All 1D input arrays must have same length");
                      }

                      // Copy data to ensure contiguity
                      u0 = cv::Mat(N, 1, CV_32F);
                      v0 = cv::Mat(N, 1, CV_32F);
                      u1 = cv::Mat(N, 1, CV_32F);
                      v1 = cv::Mat(N, 1, CV_32F);

                      float* u0_src = (float*)u0_info.ptr;
                      float* v0_src = (float*)v0_info.ptr;
                      float* u1_src = (float*)u1_info.ptr;
                      float* v1_src = (float*)v1_info.ptr;

                      for (int i = 0; i < N; ++i) {
                          u0.at<float>(i, 0) = u0_src[i];
                          v0.at<float>(i, 0) = v0_src[i];
                          u1.at<float>(i, 0) = u1_src[i];
                          v1.at<float>(i, 0) = v1_src[i];
                      }
                  } else if (u0_info.ndim == 2) {
                      // 2D dense grid - override H, W from array shape
                      H = u0_info.shape[0];
                      W = u0_info.shape[1];

                      if (v0_info.shape[0] != H || v0_info.shape[1] != W ||
                          u1_info.shape[0] != H || u1_info.shape[1] != W ||
                          v1_info.shape[0] != H || v1_info.shape[1] != W) {
                          throw std::runtime_error("All 2D input arrays must have same shape");
                      }

                      u0 = cv::Mat(H, W, CV_32F);
                      v0 = cv::Mat(H, W, CV_32F);
                      u1 = cv::Mat(H, W, CV_32F);
                      v1 = cv::Mat(H, W, CV_32F);

                      float* u0_src = u0_np.mutable_data();
                      float* v0_src = v0_np.mutable_data();
                      float* u1_src = u1_np.mutable_data();
                      float* v1_src = v1_np.mutable_data();

                      for (int r = 0; r < H; ++r) {
                          for (int c = 0; c < W; ++c) {
                              u0.at<float>(r, c) = u0_src[r * W + c];
                              v0.at<float>(r, c) = v0_src[r * W + c];
                              u1.at<float>(r, c) = u1_src[r * W + c];
                              v1.at<float>(r, c) = v1_src[r * W + c];
                          }
                      }
                  } else {
                      throw std::runtime_error("Input arrays must be 1D (sparse) or 2D (dense)");
                  }

                  // Create triangulator
                  pr_depth::TriangulationConfig config;
                  pr_depth::Triangulator triangulator(H, W, fx, fy, cx, cy, config);

                  // Perform triangulation
                  auto result = triangulator.triangulate(u0, v0, u1, v1, R, t);

                  // Convert output to numpy arrays
                  py::array_t<float> z1_tri_np({H, W});
                  py::array_t<float> rpx_tri_np({H, W});

                  float* z_ptr = z1_tri_np.mutable_data();
                  float* e_ptr = rpx_tri_np.mutable_data();

                  for (int r = 0; r < H; ++r) {
                      for (int c = 0; c < W; ++c) {
                          z_ptr[r * W + c] = result.z1_tri.at<float>(r, c);
                          e_ptr[r * W + c] = result.rpx_tri.at<float>(r, c);
                      }
                  }

                  // Build output dictionary
                  py::dict output;
                  output["z1_tri"] = z1_tri_np;
                  output["rpx_tri"] = rpx_tri_np;
                  output["num_valid"] = result.num_valid;

                  return output;
              } catch (const std::exception& e) {
                  throw std::runtime_error(std::string("Triangulation failed: ") + e.what());
              }
          },
          "Triangulate depth from optical flow correspondences",
          py::arg("u0"), py::arg("v0"), py::arg("u1"), py::arg("v1"),
          py::arg("R"), py::arg("t"),
          py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
          py::arg("H") = 375, py::arg("W") = 1242);

    // Depth fusion: rpx_to_variance_angle_aware
    m.def("rpx_to_variance_angle_aware",
          [](py::array_t<float> rpx_np,
             float fx, float fy,
             float baseline, float b_ref_auto,
             float tau0_deg, float sigma2_at_tau,
             float var_floor, float var_cap) -> py::array_t<float> {
              // Get input dimensions
              auto rpx_info = rpx_np.request();
              if (rpx_info.ndim != 2) {
                  throw std::runtime_error("rpx must be 2D array (HxW)");
              }

              int H = rpx_info.shape[0];
              int W = rpx_info.shape[1];

              // Create cv::Mat view of input
              cv::Mat rpx(H, W, CV_32F, rpx_info.ptr);

              // Configure and call
              pr_depth::DepthFusionConfig config;
              config.tau0_deg = tau0_deg;
              config.sigma2_at_tau = sigma2_at_tau;
              config.var_floor = var_floor;
              config.var_cap = var_cap;

              cv::Mat variance = pr_depth::rpx_to_variance_angle_aware(
                  rpx, fx, fy, baseline, b_ref_auto, config
              );

              // Convert output to numpy
              py::array_t<float> result({H, W});
              float* out_ptr = result.mutable_data();
              for (int r = 0; r < H; ++r) {
                  const float* row = variance.ptr<float>(r);
                  for (int c = 0; c < W; ++c) {
                      out_ptr[r * W + c] = row[c];
                  }
              }

              return result;
          },
          "Convert Sampson error to observation variance (angle-aware)",
          py::arg("rpx"),
          py::arg("fx"), py::arg("fy"),
          py::arg("baseline"), py::arg("b_ref_auto"),
          py::arg("tau0_deg") = -1.0f,
          py::arg("sigma2_at_tau") = 1.0f,
          py::arg("var_floor") = 1e-3f,
          py::arg("var_cap") = 1e+3f);

    // Depth fusion: robust_scale_match
    m.def("robust_scale_match",
          [](py::array_t<float> z_tri_np, py::array_t<float> z_ref_np,
             py::array_t<uint8_t> mask_np,
             float max_depth, int min_overlap, float tol_median) -> py::dict {
              // Get dimensions
              auto tri_info = z_tri_np.request();
              if (tri_info.ndim != 2) {
                  throw std::runtime_error("z_tri must be 2D array (HxW)");
              }
              int H = tri_info.shape[0];
              int W = tri_info.shape[1];

              // Create cv::Mat views
              cv::Mat z_tri(H, W, CV_32F, z_tri_np.mutable_data());
              cv::Mat z_ref(H, W, CV_32F, z_ref_np.mutable_data());

              cv::Mat mask;
              if (mask_np.size() > 0) {
                  auto mask_info = mask_np.request();
                  mask = cv::Mat(H, W, CV_8U, mask_np.mutable_data());
              }

              // Configure
              pr_depth::RobustScaleConfig config;
              config.max_depth = max_depth;
              config.min_overlap = min_overlap;
              config.tol_median = tol_median;

              // Call
              auto result = pr_depth::robust_scale_match(z_tri, z_ref, mask, config);

              // Build output dict (matching Python's return format)
              py::dict info;
              info["scale"] = result.scale;
              info["scale_ok"] = result.scale_ok;
              info["overlap"] = result.overlap;
              info["median_relerr"] = result.median_relerr;

              return info;
          },
          "Robustly estimate scale between triangulated and reference depth",
          py::arg("z_tri"), py::arg("z_ref"),
          py::arg("mask") = py::array_t<uint8_t>(),
          py::arg("max_depth") = 80.0f,
          py::arg("min_overlap") = 2000,
          py::arg("tol_median") = 0.3f);

    // Depth fusion: warp_map_fast
    m.def("warp_map_fast",
          [](py::array_t<float> u0_np, py::array_t<float> v0_np,
             py::array_t<float> u1_np, py::array_t<float> v1_np,
             py::array_t<float> M0_np,
             float fill) -> py::array_t<float> {
              // Get M0 dimensions
              auto m0_info = M0_np.request();
              if (m0_info.ndim != 2) {
                  throw std::runtime_error("M0 must be 2D array (HxW)");
              }
              int H = m0_info.shape[0];
              int W = m0_info.shape[1];

              // Create cv::Mat view of M0
              cv::Mat M0(H, W, CV_32F, M0_np.mutable_data());

              // Convert coordinate arrays to vectors
              auto u0_info = u0_np.request();
              int N = u0_info.shape[0];

              std::vector<float> u0(N), v0(N), u1(N), v1(N);
              float* u0_ptr = u0_np.mutable_data();
              float* v0_ptr = v0_np.mutable_data();
              float* u1_ptr = u1_np.mutable_data();
              float* v1_ptr = v1_np.mutable_data();

              for (int i = 0; i < N; ++i) {
                  u0[i] = u0_ptr[i];
                  v0[i] = v0_ptr[i];
                  u1[i] = u1_ptr[i];
                  v1[i] = v1_ptr[i];
              }

              // Call
              cv::Mat result = pr_depth::warp_map_fast(u0, v0, u1, v1, M0, fill);

              // Convert output to numpy
              py::array_t<float> out({H, W});
              float* out_ptr = out.mutable_data();
              for (int r = 0; r < H; ++r) {
                  const float* row = result.ptr<float>(r);
                  for (int c = 0; c < W; ++c) {
                      out_ptr[r * W + c] = row[c];
                  }
              }

              return out;
          },
          "Fast map warping using nearest-neighbor splatting",
          py::arg("u0"), py::arg("v0"),
          py::arg("u1"), py::arg("v1"),
          py::arg("M0"),
          py::arg("fill") = std::numeric_limits<float>::quiet_NaN());

    // Depth fusion: aggregate_label_median
    m.def("aggregate_label_median",
          [](py::array_t<float> map2d_np, py::array_t<int32_t> labels_np,
             int min_pts) -> py::array_t<float> {
              // Get dimensions
              auto map_info = map2d_np.request();
              if (map_info.ndim != 2) {
                  throw std::runtime_error("map2d must be 2D array (HxW)");
              }
              int H = map_info.shape[0];
              int W = map_info.shape[1];

              // Create cv::Mat views
              cv::Mat map2d(H, W, CV_32F, map2d_np.mutable_data());
              cv::Mat labels(H, W, CV_32S, labels_np.mutable_data());

              // Call
              auto result = pr_depth::aggregate_label_median(map2d, labels, min_pts);

              // Convert to numpy
              py::array_t<float> out(result.size());
              float* out_ptr = out.mutable_data();
              for (size_t i = 0; i < result.size(); ++i) {
                  out_ptr[i] = result[i];
              }
              return out;
          },
          "Compute per-label median of a 2D map",
          py::arg("map2d"), py::arg("labels"), py::arg("min_pts") = 20);

    // BaselineAutoState class
    py::class_<pr_depth::BaselineAutoState>(m, "BaselineAutoState")
        .def(py::init<float, int>(),
             py::arg("ema_beta") = 0.5f, py::arg("hist_len") = 100)
        .def("update", &pr_depth::BaselineAutoState::update)
        .def("b_ref", &pr_depth::BaselineAutoState::b_ref)
        .def("guard_threshold", &pr_depth::BaselineAutoState::guard_threshold)
        .def("should_disable", &pr_depth::BaselineAutoState::should_disable,
             py::arg("baseline"), py::arg("extra_min") = 0.0f);

    // Depth fusion: solve_metric_from_rel
    m.def("solve_metric_from_rel",
          [](py::array_t<float> rel_depth_np, py::array_t<float> z_obs_np,
             py::array_t<uint8_t> mask_np, py::array_t<int32_t> labels_np,
             py::array_t<float> v_px_np,
             int min_pts_per_label, float min_pts_ratio,
             float single_med_thr, float single_p90_thr,
             float global_trim_k, float var_floor, float var_cap) -> py::dict {
              // Get dimensions
              auto rd_info = rel_depth_np.request();
              if (rd_info.ndim != 2) {
                  throw std::runtime_error("rel_depth must be 2D array (HxW)");
              }
              int H = rd_info.shape[0];
              int W = rd_info.shape[1];

              // Create cv::Mat views
              cv::Mat rel_depth(H, W, CV_32F, rel_depth_np.mutable_data());
              cv::Mat z_obs(H, W, CV_32F, z_obs_np.mutable_data());

              cv::Mat mask;
              if (mask_np.size() > 0) {
                  mask = cv::Mat(H, W, CV_8U, mask_np.mutable_data());
              }

              cv::Mat labels;
              if (labels_np.size() > 0) {
                  labels = cv::Mat(H, W, CV_32S, labels_np.mutable_data());
              }

              cv::Mat v_px;
              if (v_px_np.size() > 0) {
                  v_px = cv::Mat(H, W, CV_32F, v_px_np.mutable_data());
              }

              // Configure
              pr_depth::MetricScaleConfig config;
              config.min_pts_per_label = min_pts_per_label;
              config.min_pts_ratio = min_pts_ratio;
              config.single_med_thr = single_med_thr;
              config.single_p90_thr = single_p90_thr;
              config.global_trim_k = global_trim_k;
              config.var_floor = var_floor;
              config.var_cap = var_cap;

              // Call
              auto result = pr_depth::solve_metric_from_rel(
                  rel_depth, z_obs, mask, labels, v_px, config);

              // Helper to convert cv::Mat to numpy
              auto mat_to_numpy_f32 = [H, W](const cv::Mat& mat) -> py::array_t<float> {
                  py::array_t<float> arr({H, W});
                  float* ptr = arr.mutable_data();
                  for (int r = 0; r < H; ++r) {
                      const float* row = mat.ptr<float>(r);
                      for (int c = 0; c < W; ++c) {
                          ptr[r * W + c] = row[c];
                      }
                  }
                  return arr;
              };

              // Build output dict
              py::dict output;
              output["z_out"] = mat_to_numpy_f32(result.z_out);
              output["S_map"] = mat_to_numpy_f32(result.S_map);
              output["V_out"] = mat_to_numpy_f32(result.V_out);
              output["global_scale"] = result.global_scale;
              output["global_med_rel"] = result.global_med_rel;
              output["global_p90_rel"] = result.global_p90_rel;

              return output;
          },
          "Solve metric scale from relative depth (outdoor)",
          py::arg("rel_depth"), py::arg("z_obs"),
          py::arg("mask") = py::array_t<uint8_t>(),
          py::arg("labels") = py::array_t<int32_t>(),
          py::arg("v_px") = py::array_t<float>(),
          py::arg("min_pts_per_label") = 10,
          py::arg("min_pts_ratio") = 0.001f,  // 0.1% - sparse triangulation
          py::arg("single_med_thr") = 0.12f,
          py::arg("single_p90_thr") = 0.25f,
          py::arg("global_trim_k") = 4.5f,
          py::arg("var_floor") = 1e-3f,
          py::arg("var_cap") = 1e+3f);

    // DepthRefinementConfig class
    py::class_<pr_depth::DepthRefinementConfig>(m, "DepthRefinementConfig")
        .def(py::init<>())
        .def_readwrite("fx", &pr_depth::DepthRefinementConfig::fx)
        .def_readwrite("fy", &pr_depth::DepthRefinementConfig::fy)
        .def_readwrite("cx", &pr_depth::DepthRefinementConfig::cx)
        .def_readwrite("cy", &pr_depth::DepthRefinementConfig::cy)
        .def_readwrite("H", &pr_depth::DepthRefinementConfig::H)
        .def_readwrite("W", &pr_depth::DepthRefinementConfig::W)
        .def_readwrite("ransac_max_iters", &pr_depth::DepthRefinementConfig::ransac_max_iters)
        .def_readwrite("ransac_min_sample", &pr_depth::DepthRefinementConfig::ransac_min_sample)
        .def_readwrite("ransac_thresh_ratio", &pr_depth::DepthRefinementConfig::ransac_thresh_ratio)
        .def_readwrite("min_flow_px", &pr_depth::DepthRefinementConfig::min_flow_px)
        .def_readwrite("max_points", &pr_depth::DepthRefinementConfig::max_points)
        .def_readwrite("margin_x_pct", &pr_depth::DepthRefinementConfig::margin_x_pct)
        .def_readwrite("margin_y_pct", &pr_depth::DepthRefinementConfig::margin_y_pct)
        .def_readwrite("depth_bins", &pr_depth::DepthRefinementConfig::depth_bins)
        .def_readwrite("adaptive_flow_depth_scale", &pr_depth::DepthRefinementConfig::adaptive_flow_depth_scale)
        .def_readwrite("max_depth", &pr_depth::DepthRefinementConfig::max_depth)
        .def_readwrite("fill_value", &pr_depth::DepthRefinementConfig::fill_value)
        .def_readwrite("use_baseline_guard", &pr_depth::DepthRefinementConfig::use_baseline_guard)
        .def_readwrite("min_baseline", &pr_depth::DepthRefinementConfig::min_baseline)
        .def_readwrite("baseline_ema_beta", &pr_depth::DepthRefinementConfig::baseline_ema_beta)
        .def_readwrite("baseline_hist_len", &pr_depth::DepthRefinementConfig::baseline_hist_len)
        .def_readwrite("min_scale_overlap", &pr_depth::DepthRefinementConfig::min_scale_overlap)
        .def_readwrite("scale_tol_median", &pr_depth::DepthRefinementConfig::scale_tol_median)
        .def_readwrite("use_segmentation", &pr_depth::DepthRefinementConfig::use_segmentation)
        .def_readwrite("seg_sigma", &pr_depth::DepthRefinementConfig::seg_sigma)
        .def_readwrite("seg_k", &pr_depth::DepthRefinementConfig::seg_k)
        .def_readwrite("seg_min_size", &pr_depth::DepthRefinementConfig::seg_min_size)
        .def_readwrite("seg_down", &pr_depth::DepthRefinementConfig::seg_down)
        .def_readwrite("use_rgb_guide", &pr_depth::DepthRefinementConfig::use_rgb_guide)
        .def_readwrite("wrgb", &pr_depth::DepthRefinementConfig::wrgb)
        .def_readwrite("wx", &pr_depth::DepthRefinementConfig::wx)
        .def_readwrite("wgrad", &pr_depth::DepthRefinementConfig::wgrad)
        .def_readwrite("grad_power", &pr_depth::DepthRefinementConfig::grad_power)
        .def_readwrite("use_metric_scale", &pr_depth::DepthRefinementConfig::use_metric_scale)
        .def_readwrite("metric_scale_mode", &pr_depth::DepthRefinementConfig::metric_scale_mode)
        .def_readwrite("use_magsac_scoring", &pr_depth::DepthRefinementConfig::use_magsac_scoring)
        .def_readwrite("debug", &pr_depth::DepthRefinementConfig::debug)
        .def_readwrite("timing", &pr_depth::DepthRefinementConfig::timing)
        .def_readwrite("enable_iterative_refinement", &pr_depth::DepthRefinementConfig::enable_iterative_refinement)
        .def_readwrite("iterative_refinement_iters", &pr_depth::DepthRefinementConfig::iterative_refinement_iters)
        .def_readwrite("use_gt_pose_fallback", &pr_depth::DepthRefinementConfig::use_gt_pose_fallback)
        .def_readwrite("gt_pose_rotation_threshold_deg", &pr_depth::DepthRefinementConfig::gt_pose_rotation_threshold_deg)
        .def_readwrite("skip_temporal_fusion", &pr_depth::DepthRefinementConfig::skip_temporal_fusion)
        .def_readwrite("use_gt_R", &pr_depth::DepthRefinementConfig::use_gt_R)
        .def_readwrite("skip_fb_consistency", &pr_depth::DepthRefinementConfig::skip_fb_consistency)
        // Fusion parameters
        .def_readwrite("fusion_tau0_deg", &pr_depth::DepthRefinementConfig::fusion_tau0_deg)
        .def_readwrite("fusion_sigma2_at_tau", &pr_depth::DepthRefinementConfig::fusion_sigma2_at_tau)
        .def_readwrite("fusion_var_floor", &pr_depth::DepthRefinementConfig::fusion_var_floor)
        .def_readwrite("fusion_var_cap", &pr_depth::DepthRefinementConfig::fusion_var_cap)
        .def_readwrite("fusion_chi2_soft", &pr_depth::DepthRefinementConfig::fusion_chi2_soft)
        .def_readwrite("fusion_chi2_hard", &pr_depth::DepthRefinementConfig::fusion_chi2_hard)
        .def_readwrite("fusion_kcap_floor", &pr_depth::DepthRefinementConfig::fusion_kcap_floor)
        .def_readwrite("fusion_lambda_forget", &pr_depth::DepthRefinementConfig::fusion_lambda_forget);

    // DepthRefinement class
    py::class_<pr_depth::DepthRefinement>(m, "DepthRefinement")
        .def(py::init<const pr_depth::DepthRefinementConfig&>())
        .def("reset", &pr_depth::DepthRefinement::reset)
        .def("frame_count", &pr_depth::DepthRefinement::frame_count)
        .def("refine",
             [](pr_depth::DepthRefinement& self,
                py::array_t<uint8_t> img_np,
                py::array_t<float> inv_depth_np,
                float baseline,
                py::array_t<int32_t> seg_labels_np,
                py::array_t<double> gt_R_np,
                py::array_t<double> gt_t_np) -> py::dict {
                 // Convert image
                 auto img_info = img_np.request();
                 cv::Mat img;
                 if (img_info.ndim == 2) {
                     img = cv::Mat(img_info.shape[0], img_info.shape[1], CV_8U, img_info.ptr);
                 } else if (img_info.ndim == 3 && img_info.shape[2] == 3) {
                     img = cv::Mat(img_info.shape[0], img_info.shape[1], CV_8UC3, img_info.ptr);
                 } else {
                     throw std::runtime_error("Image must be HxW (grayscale) or HxWx3 (BGR)");
                 }

                 // Convert inv_depth
                 auto inv_depth_info = inv_depth_np.request();
                 if (inv_depth_info.ndim != 2) {
                     throw std::runtime_error("inv_depth must be 2D array (HxW)");
                 }
                 cv::Mat inv_depth(inv_depth_info.shape[0], inv_depth_info.shape[1],
                                   CV_32F, inv_depth_info.ptr);

                 // Convert seg_labels (optional)
                 cv::Mat seg_labels;
                 if (seg_labels_np.size() > 0) {
                     auto seg_info = seg_labels_np.request();
                     seg_labels = cv::Mat(seg_info.shape[0], seg_info.shape[1],
                                         CV_32S, seg_info.ptr);
                 }

                 // Convert GT pose (optional)
                 std::optional<Eigen::Matrix3d> gt_R = std::nullopt;
                 std::optional<Eigen::Vector3d> gt_t = std::nullopt;

                 if (gt_R_np.size() > 0) {
                     auto R_info = gt_R_np.request();
                     if (R_info.ndim == 2 && R_info.shape[0] == 3 && R_info.shape[1] == 3) {
                         double* R_data = (double*)R_info.ptr;
                         Eigen::Matrix3d R;
                         size_t row_stride = R_info.strides[0] / sizeof(double);
                         size_t col_stride = R_info.strides[1] / sizeof(double);
                         for (int i = 0; i < 3; ++i) {
                             for (int j = 0; j < 3; ++j) {
                                 R(i, j) = R_data[i * row_stride + j * col_stride];
                             }
                         }
                         gt_R = R;
                     }
                 }

                 if (gt_t_np.size() > 0) {
                     auto t_info = gt_t_np.request();
                     if (t_info.ndim == 1 && t_info.shape[0] == 3) {
                         double* t_data = (double*)t_info.ptr;
                         Eigen::Vector3d t(t_data[0], t_data[1], t_data[2]);
                         gt_t = t;
                     }
                 }

                 // Call refine with optional GT pose
                 auto result = self.refine(img, inv_depth, baseline, gt_R, gt_t, seg_labels);

                 // Convert output to numpy
                 int H = result.z_refined.rows;
                 int W = result.z_refined.cols;

                 auto mat_to_numpy_f32 = [H, W](const cv::Mat& mat) -> py::array_t<float> {
                     if (mat.empty() || mat.rows != H || mat.cols != W) {
                         py::array_t<float> arr({H, W});
                         std::fill(arr.mutable_data(), arr.mutable_data() + H * W, 0.0f);
                         return arr;
                     }
                     py::array_t<float> arr({H, W});
                     float* ptr = arr.mutable_data();
                     for (int r = 0; r < H; ++r) {
                         const float* row = mat.ptr<float>(r);
                         for (int c = 0; c < W; ++c) {
                             ptr[r * W + c] = row[c];
                         }
                     }
                     return arr;
                 };

                 // Convert seg_labels (int32)
                 auto mat_to_numpy_i32 = [H, W](const cv::Mat& mat) -> py::array_t<int32_t> {
                     py::array_t<int32_t> arr({H, W});
                     int32_t* ptr = arr.mutable_data();
                     for (int r = 0; r < H; ++r) {
                         const int32_t* row = mat.ptr<int32_t>(r);
                         for (int c = 0; c < W; ++c) {
                             ptr[r * W + c] = row[c];
                         }
                     }
                     return arr;
                 };

                 py::dict output;
                 output["z_refined"] = mat_to_numpy_f32(result.z_refined);
                 output["z_tri"] = mat_to_numpy_f32(result.z_tri);
                 output["confidence"] = mat_to_numpy_f32(result.confidence);
                 output["seg_labels"] = mat_to_numpy_i32(result.seg_labels);

                 // Convert R (3x3 rotation matrix) to numpy
                 py::array_t<double> R_np({3, 3});
                 double* R_ptr = R_np.mutable_data();
                 for (int i = 0; i < 3; ++i) {
                     for (int j = 0; j < 3; ++j) {
                         R_ptr[i * 3 + j] = result.R(i, j);
                     }
                 }
                 output["R"] = R_np;

                 // Convert t (3x1 translation direction) to numpy
                 // NOTE: Use explicit {shape},{strides} to avoid pybind11 stride=0 bug
                 py::array_t<double> t_np({3}, {sizeof(double)});
                 double* t_ptr = t_np.mutable_data();
                 t_ptr[0] = result.t(0);
                 t_ptr[1] = result.t(1);
                 t_ptr[2] = result.t(2);
                 output["t"] = t_np;

                 output["baseline"] = result.baseline;
                 output["num_matches"] = result.num_matches;
                 output["num_valid_tri"] = result.num_valid_tri;
                 output["num_segments"] = result.num_segments;
                 output["tri_disabled"] = result.tri_disabled;
                 output["baseline_correction"] = result.baseline_correction;

                 // Debug info - scalar values
                 output["used_backward"] = result.used_backward;
                 output["metric_scale_forward"] = result.metric_scale_forward;
                 output["metric_scale_backward"] = result.metric_scale_backward;

                 // GT pose fallback info
                 output["used_gt_pose"] = result.used_gt_pose;
                 output["rotation_angle_deg"] = result.rotation_angle_deg;

                 // Debug depth maps - use separate lambda with dynamic size
                 auto safe_mat_to_numpy = [](const cv::Mat& mat) -> py::array_t<float> {
                     if (mat.empty()) {
                         return py::array_t<float>();  // Empty array
                     }
                     int h = mat.rows;
                     int w = mat.cols;
                     py::array_t<float> arr({h, w});
                     float* ptr = arr.mutable_data();
                     for (int r = 0; r < h; ++r) {
                         const float* row = mat.ptr<float>(r);
                         for (int c = 0; c < w; ++c) {
                             ptr[r * w + c] = row[c];
                         }
                     }
                     return arr;
                 };

                 // Only add non-empty depth maps
                 if (!result.z_tri_forward.empty()) {
                     output["z_tri_forward"] = safe_mat_to_numpy(result.z_tri_forward);
                 }
                 if (!result.z_tri_backward.empty()) {
                     output["z_tri_backward"] = safe_mat_to_numpy(result.z_tri_backward);
                 }
                 if (!result.z_warp_flow.empty()) {
                     output["z_warp_flow"] = safe_mat_to_numpy(result.z_warp_flow);
                 }
                 if (!result.z_warp_pose.empty()) {
                     output["z_warp_pose"] = safe_mat_to_numpy(result.z_warp_pose);
                 }
                 if (!result.prev_depth_used.empty()) {
                     output["prev_depth_used"] = safe_mat_to_numpy(result.prev_depth_used);
                 }
                 if (!result.V_prior.empty()) {
                     output["V_prior"] = safe_mat_to_numpy(result.V_prior);
                 }
                 if (!result.V_post.empty()) {
                     output["V_post"] = safe_mat_to_numpy(result.V_post);
                 }
                 if (!result.z_warp_gt.empty()) {
                     output["z_warp_gt"] = safe_mat_to_numpy(result.z_warp_gt);
                 }

                 // Per-iteration depth maps (for analyzing iterative refinement)
                 if (!result.iteration_info.empty()) {
                     py::list iter_list;
                     for (size_t i = 0; i < result.iteration_info.size(); ++i) {
                         const auto& iter_info = result.iteration_info[i];
                         py::dict iter_dict;
                         iter_dict["iter"] = iter_info.iter;
                         iter_dict["is_backward"] = iter_info.is_backward;
                         iter_dict["num_inliers"] = iter_info.num_inliers;
                         iter_dict["num_valid_tri"] = iter_info.num_valid_tri;
                         iter_dict["metric_scale"] = iter_info.metric_scale;

                         // Store depth maps for each iteration
                         if (!iter_info.z_tri.empty()) {
                             iter_dict["z_tri"] = safe_mat_to_numpy(iter_info.z_tri);
                         }
                         if (!iter_info.z_fused_sparse.empty()) {
                             iter_dict["z_fused_sparse"] = safe_mat_to_numpy(iter_info.z_fused_sparse);
                         }
                         if (!iter_info.z_refined.empty()) {
                             iter_dict["z_refined"] = safe_mat_to_numpy(iter_info.z_refined);
                         }

                         iter_list.append(iter_dict);
                     }
                     output["iteration_info"] = iter_list;
                 }

                 return output;
             },
             "Refine depth from monocular inverse depth",
             py::arg("img"),
             py::arg("inv_depth"),
             py::arg("baseline"),
             py::arg("seg_labels") = py::array_t<int32_t>(),
             py::arg("gt_R") = py::array_t<double>(),
             py::arg("gt_t") = py::array_t<double>());
}
