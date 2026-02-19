#include "pr_depth/optical_flow.hpp"
#include <stdexcept>

namespace pr_depth {

OpticalFlowDIS::OpticalFlowDIS(const OpticalFlowConfig& cfg)
    : cfg_(cfg)
{
    // Paper: Section 3.1, line 163 - "DISFlow [21]"
    // Use OpenCV enum directly (already set in config)
    dis_ = cv::DISOpticalFlow::create(cfg.preset);
    dis_->setFinestScale(cfg.finest_scale);
    dis_->setUseSpatialPropagation(cfg.use_spatial_propagation);
}

cv::Mat OpticalFlowDIS::compute(const cv::Mat& img_prev, const cv::Mat& img_curr) {
    if (img_prev.empty() || img_curr.empty()) {
        throw std::invalid_argument("Input images cannot be empty");
    }

    if (img_prev.size() != img_curr.size()) {
        throw std::invalid_argument("Image size mismatch");
    }

    // Convert to grayscale
    cv::Mat gray_prev = to_gray(img_prev);
    cv::Mat gray_curr = to_gray(img_curr);

    // Compute flow
    cv::Mat flow;
    dis_->calc(gray_prev, gray_curr, flow);

    // FIX: DIS already returns CV_32FC2, just verify
    if (flow.type() != CV_32FC2) {
        throw std::runtime_error(
            "DISOpticalFlow returned unexpected type: " +
            std::to_string(flow.type()) + " (expected CV_32FC2=" +
            std::to_string(CV_32FC2) + ")"
        );
    }

    return flow;
}

cv::Mat OpticalFlowDIS::to_gray(const cv::Mat& img) {
    if (img.channels() == 1) {
        // Already grayscale
        if (img.type() == CV_8U) {
            return img;
        } else {
            cv::Mat out;
            img.convertTo(out, CV_8U);
            return out;
        }
    } else {
        cv::Mat gray;
        // Handle BGR or RGB (OpenCV uses BGR by default)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }
}

} // namespace pr_depth
