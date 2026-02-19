#pragma once
#include "types.hpp"
#include <opencv2/video.hpp>  // FIX: More robust include for DISOpticalFlow
#include <opencv2/imgproc.hpp>

namespace pr_depth {

// Paper: Section 3.1, lines 163-165
// "For optical flow estimation, we used DISFlow [21]"
class OpticalFlowDIS {
public:
    explicit OpticalFlowDIS(const OpticalFlowConfig& cfg = OpticalFlowConfig());

    // Compute dense optical flow
    // Returns: HxWx2 float32 (u,v) flow field
    cv::Mat compute(const cv::Mat& img_prev, const cv::Mat& img_curr);

private:
    cv::Ptr<cv::DISOpticalFlow> dis_;
    OpticalFlowConfig cfg_;

    static cv::Mat to_gray(const cv::Mat& img);
};

} // namespace pr_depth
