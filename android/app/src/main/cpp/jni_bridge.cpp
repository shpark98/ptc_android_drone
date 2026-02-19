/**
 * JNI Bridge for PR-Depth Android
 * Connects Java/Kotlin to C++ DepthRefinement pipeline.
 *
 * Accepts YUV camera planes directly (same format as QNN bridge) and
 * builds a BGR image using the same rotation+resize mapping as the QNN
 * preprocessor, ensuring perfect spatial alignment between the camera
 * image and QNN depth output.
 */

#include <jni.h>
#include <android/log.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Dense>
#include <memory>
#include <string>
#include <chrono>
#include <algorithm>
#include <cmath>

#include "pr_depth/depth_refinement.hpp"

#define LOG_TAG "PR-Depth-JNI"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

// ============================================================================
// Helper functions
// ============================================================================

/**
 * Build BGR image from YUV planes using the SAME rotation+resize mapping
 * as the QNN preprocessor. This ensures spatial alignment with QNN depth.
 *
 * Produces a targetW x targetH BGR image at the model's coordinate space.
 */
static cv::Mat buildBGRFromYUV(
    const uint8_t* yData, const uint8_t* uData, const uint8_t* vData,
    int imgW, int imgH, int yRowStride, int uvRowStride, int uvPixelStride,
    int targetW, int targetH, int rotDeg)
{
    cv::Mat bgr(targetH, targetW, CV_8UC3);

    for (int ty = 0; ty < targetH; ty++) {
        auto* row = bgr.ptr<cv::Vec3b>(ty);
        for (int tx = 0; tx < targetW; tx++) {
            float srcXf, srcYf;

            if (rotDeg == 90) {
                srcXf = static_cast<float>(ty) * (imgW - 1) / (targetH - 1);
                srcYf = (imgH - 1) - static_cast<float>(tx) * (imgH - 1) / (targetW - 1);
            } else if (rotDeg == 270) {
                srcXf = (imgW - 1) - static_cast<float>(ty) * (imgW - 1) / (targetH - 1);
                srcYf = static_cast<float>(tx) * (imgH - 1) / (targetW - 1);
            } else if (rotDeg == 180) {
                srcXf = (imgW - 1) - static_cast<float>(tx) * (imgW - 1) / (targetW - 1);
                srcYf = (imgH - 1) - static_cast<float>(ty) * (imgH - 1) / (targetH - 1);
            } else {
                srcXf = static_cast<float>(tx) * (imgW - 1) / (targetW - 1);
                srcYf = static_cast<float>(ty) * (imgH - 1) / (targetH - 1);
            }

            int sX = std::max(0, std::min(static_cast<int>(srcXf + 0.5f), imgW - 1));
            int sY = std::max(0, std::min(static_cast<int>(srcYf + 0.5f), imgH - 1));

            int yIdx = sY * yRowStride + sX;
            int uvIdx = (sY / 2) * uvRowStride + (sX / 2) * uvPixelStride;

            int Y = yData[yIdx];
            int U = uData[uvIdx] - 128;
            int V = vData[uvIdx] - 128;

            int r = std::max(0, std::min(255, Y + static_cast<int>(1.370705f * V)));
            int g = std::max(0, std::min(255, Y - static_cast<int>(0.337633f * U) - static_cast<int>(0.698001f * V)));
            int b = std::max(0, std::min(255, Y + static_cast<int>(1.732446f * U)));

            row[tx] = cv::Vec3b(static_cast<uint8_t>(b), static_cast<uint8_t>(g), static_cast<uint8_t>(r));
        }
    }

    return bgr;
}

static cv::Mat javaFloatArrayToMat(JNIEnv* env, jfloatArray arr, int rows, int cols) {
    if (arr == nullptr) return cv::Mat();
    jfloat* data = env->GetFloatArrayElements(arr, nullptr);
    cv::Mat mat(rows, cols, CV_32F);
    memcpy(mat.data, data, rows * cols * sizeof(float));
    env->ReleaseFloatArrayElements(arr, data, JNI_ABORT);
    return mat;
}

static Eigen::Matrix3d javaFloatArrayToEigenMatrix3d(JNIEnv* env, jfloatArray arr) {
    if (arr == nullptr) return Eigen::Matrix3d::Identity();
    jfloat* data = env->GetFloatArrayElements(arr, nullptr);
    Eigen::Matrix3d mat;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            mat(i, j) = data[i * 3 + j];
    env->ReleaseFloatArrayElements(arr, data, JNI_ABORT);
    return mat;
}

static Eigen::Vector3d javaFloatArrayToEigenVector3d(JNIEnv* env, jfloatArray arr) {
    if (arr == nullptr) return Eigen::Vector3d::Zero();
    jfloat* data = env->GetFloatArrayElements(arr, nullptr);
    Eigen::Vector3d vec(data[0], data[1], data[2]);
    env->ReleaseFloatArrayElements(arr, data, JNI_ABORT);
    return vec;
}

static jfloatArray matToJavaFloatArray(JNIEnv* env, const cv::Mat& mat) {
    int size = mat.rows * mat.cols;
    jfloatArray arr = env->NewFloatArray(size);
    if (mat.type() == CV_32F) {
        env->SetFloatArrayRegion(arr, 0, size, (jfloat*)mat.data);
    } else {
        cv::Mat floatMat;
        mat.convertTo(floatMat, CV_32F);
        env->SetFloatArrayRegion(arr, 0, size, (jfloat*)floatMat.data);
    }
    return arr;
}

static jfloatArray eigenMatrix3dToJavaFloatArray(JNIEnv* env, const Eigen::Matrix3d& mat) {
    jfloatArray arr = env->NewFloatArray(9);
    jfloat data[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            data[i * 3 + j] = static_cast<float>(mat(i, j));
    env->SetFloatArrayRegion(arr, 0, 9, data);
    return arr;
}

static jfloatArray eigenVector3dToJavaFloatArray(JNIEnv* env, const Eigen::Vector3d& vec) {
    jfloatArray arr = env->NewFloatArray(3);
    jfloat data[3] = {
        static_cast<float>(vec(0)),
        static_cast<float>(vec(1)),
        static_cast<float>(vec(2))
    };
    env->SetFloatArrayRegion(arr, 0, 3, data);
    return arr;
}

// ============================================================================
// Middlebury flow visualization (flow_viz colormap)
// ============================================================================

static int s_colorwheel[55][3];
static int s_ncols = 0;

static void makeColorWheel() {
    const int RY=15, YG=6, GC=4, CB=11, BM=13, MR=6;
    s_ncols = RY+YG+GC+CB+BM+MR;
    int k=0;
    for(int i=0;i<RY;i++){s_colorwheel[k][0]=255;s_colorwheel[k][1]=255*i/RY;s_colorwheel[k][2]=0;k++;}
    for(int i=0;i<YG;i++){s_colorwheel[k][0]=255-255*i/YG;s_colorwheel[k][1]=255;s_colorwheel[k][2]=0;k++;}
    for(int i=0;i<GC;i++){s_colorwheel[k][0]=0;s_colorwheel[k][1]=255;s_colorwheel[k][2]=255*i/GC;k++;}
    for(int i=0;i<CB;i++){s_colorwheel[k][0]=0;s_colorwheel[k][1]=255-255*i/CB;s_colorwheel[k][2]=255;k++;}
    for(int i=0;i<BM;i++){s_colorwheel[k][0]=255*i/BM;s_colorwheel[k][1]=0;s_colorwheel[k][2]=255;k++;}
    for(int i=0;i<MR;i++){s_colorwheel[k][0]=255;s_colorwheel[k][1]=0;s_colorwheel[k][2]=255-255*i/MR;k++;}
}

static void computeFlowColor(float fx, float fy, uint8_t& r, uint8_t& g, uint8_t& b) {
    if (s_ncols == 0) makeColorWheel();
    float rad = sqrtf(fx*fx + fy*fy);
    float a = atan2f(-fy, -fx) / (float)M_PI;  // [-1, 1]
    float fk = (a + 1.0f) / 2.0f * (s_ncols - 1);
    int k0 = (int)fk;
    int k1 = (k0 + 1) % s_ncols;
    float f = fk - k0;
    float cols[3];
    for (int c = 0; c < 3; c++) {
        float col0 = s_colorwheel[k0][c] / 255.0f;
        float col1 = s_colorwheel[k1][c] / 255.0f;
        float col = (1-f)*col0 + f*col1;
        if (rad <= 1.0f) col = 1.0f - rad * (1.0f - col);
        else             col *= 0.75f;
        cols[c] = col;
    }
    r = (uint8_t)(255.0f * cols[0]);
    g = (uint8_t)(255.0f * cols[1]);
    b = (uint8_t)(255.0f * cols[2]);
}

/**
 * Convert CV_32FC2 flow to packed ARGB int array for Android Bitmap.
 */
static jintArray flowToARGB(JNIEnv* env, const cv::Mat& flow) {
    if (flow.empty() || flow.type() != CV_32FC2) return nullptr;

    int H = flow.rows, W = flow.cols;

    // Find max magnitude for normalization
    float maxRad = 0;
    for (int y = 0; y < H; y++) {
        const float* row = flow.ptr<float>(y);
        for (int x = 0; x < W; x++) {
            float u = row[x*2], v = row[x*2+1];
            float rad = sqrtf(u*u + v*v);
            if (rad > maxRad) maxRad = rad;
        }
    }
    if (maxRad < 1e-5f) maxRad = 1e-5f;

    jintArray arr = env->NewIntArray(H * W);
    jint* pixels = env->GetIntArrayElements(arr, nullptr);

    for (int y = 0; y < H; y++) {
        const float* row = flow.ptr<float>(y);
        for (int x = 0; x < W; x++) {
            float u = row[x*2] / maxRad;
            float v = row[x*2+1] / maxRad;
            uint8_t r, g, b;
            computeFlowColor(u, v, r, g, b);
            pixels[y * W + x] = (0xFF << 24) | (r << 16) | (g << 8) | b;  // ARGB
        }
    }

    env->ReleaseIntArrayElements(arr, pixels, 0);
    return arr;
}

// ============================================================================
// JNI Functions
// ============================================================================

extern "C" {

/**
 * Create DepthRefinement pipeline.
 * Intrinsics should be in the MODEL space (518x518 rotated).
 */
JNIEXPORT jlong JNICALL
Java_com_prdepth_android_DepthRefinementManager_nativeCreatePipeline(
    JNIEnv* env, jobject obj,
    jfloat fx, jfloat fy, jfloat cx, jfloat cy,
    jint width, jint height,
    jboolean useGTPose
) {
    LOGI("Creating PR-Depth pipeline: %dx%d, fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f",
         width, height, fx, fy, cx, cy);

    try {
        pr_depth::DepthRefinementConfig config;
        config.fx = fx;
        config.fy = fy;
        config.cx = cx;
        config.cy = cy;
        config.H = height;
        config.W = width;

        config.use_gt_pose_fallback = useGTPose;
        config.gt_pose_rotation_threshold_deg = 3.0f;

        // Use default config — all params controllable from UI via nativeUpdateFullConfig
        config.timing = true;

        auto* pipeline = new pr_depth::DepthRefinement(config);
        LOGI("PR-Depth pipeline created at %p", pipeline);
        return reinterpret_cast<jlong>(pipeline);

    } catch (const std::exception& e) {
        LOGE("Failed to create pipeline: %s", e.what());
        return 0;
    }
}

/**
 * Process a single frame with YUV camera planes.
 *
 * Builds BGR image from YUV using same rotation mapping as QNN preprocessor.
 * Takes QNN inv_depth (at display resolution), resizes to model space (518x518).
 */
JNIEXPORT jobject JNICALL
Java_com_prdepth_android_DepthRefinementManager_nativeProcessFrameYUV(
    JNIEnv* env, jobject obj,
    jlong pipelinePtr,
    jbyteArray yArray, jbyteArray uArray, jbyteArray vArray,
    jint imgW, jint imgH,
    jint yRowStride, jint uvRowStride, jint uvPixelStride,
    jint rotDeg,
    jfloatArray invDepth, jint depthW, jint depthH,
    jfloat baseline,
    jfloatArray gtR, jfloatArray gtT
) {
    using Clock = std::chrono::high_resolution_clock;

    auto* pipeline = reinterpret_cast<pr_depth::DepthRefinement*>(pipelinePtr);
    if (!pipeline) {
        LOGE("Invalid pipeline pointer");
        return nullptr;
    }

    try {
        auto config = pipeline->getConfig();
        int targetW = config.W;
        int targetH = config.H;

        auto t0 = Clock::now();

        // Convert YUV to BGR at model resolution (518x518) with rotation
        auto* yData = static_cast<uint8_t*>(env->GetPrimitiveArrayCritical(yArray, nullptr));
        auto* uData = static_cast<uint8_t*>(env->GetPrimitiveArrayCritical(uArray, nullptr));
        auto* vData = static_cast<uint8_t*>(env->GetPrimitiveArrayCritical(vArray, nullptr));

        cv::Mat img = buildBGRFromYUV(yData, uData, vData,
            imgW, imgH, yRowStride, uvRowStride, uvPixelStride,
            targetW, targetH, rotDeg);

        env->ReleasePrimitiveArrayCritical(yArray, yData, JNI_ABORT);
        env->ReleasePrimitiveArrayCritical(uArray, uData, JNI_ABORT);
        env->ReleasePrimitiveArrayCritical(vArray, vData, JNI_ABORT);

        auto t1 = Clock::now();

        // Get inv_depth and resize from display resolution to model resolution
        cv::Mat inv_depth_display = javaFloatArrayToMat(env, invDepth, depthH, depthW);
        cv::Mat inv_depth;
        if (depthW != targetW || depthH != targetH) {
            cv::resize(inv_depth_display, inv_depth, cv::Size(targetW, targetH),
                       0, 0, cv::INTER_LINEAR);
        } else {
            inv_depth = inv_depth_display;
        }

        auto t2 = Clock::now();

        // Optional GT pose — rotate from original camera frame to rotated image frame
        std::optional<Eigen::Matrix3d> gt_R_opt;
        std::optional<Eigen::Vector3d> gt_t_opt;
        if (gtR != nullptr && gtT != nullptr) {
            Eigen::Matrix3d R_orig = javaFloatArrayToEigenMatrix3d(env, gtR);
            Eigen::Vector3d t_orig = javaFloatArrayToEigenVector3d(env, gtT);

            // Build rotation matrix for image rotation (camera 3D coords)
            // 90° CW:  X_rot = -Y_orig, Y_rot = X_orig, Z_rot = Z_orig
            // 270° CW: X_rot = Y_orig, Y_rot = -X_orig, Z_rot = Z_orig
            // 180°:    X_rot = -X_orig, Y_rot = -Y_orig, Z_rot = Z_orig
            if (rotDeg == 90) {
                Eigen::Matrix3d C;
                C << 0, -1, 0,
                     1,  0, 0,
                     0,  0, 1;
                gt_R_opt = C * R_orig * C.transpose();
                gt_t_opt = C * t_orig;
            } else if (rotDeg == 270) {
                Eigen::Matrix3d C;
                C <<  0, 1, 0,
                     -1, 0, 0,
                      0, 0, 1;
                gt_R_opt = C * R_orig * C.transpose();
                gt_t_opt = C * t_orig;
            } else if (rotDeg == 180) {
                Eigen::Matrix3d C;
                C << -1,  0, 0,
                      0, -1, 0,
                      0,  0, 1;
                gt_R_opt = C * R_orig * C.transpose();
                gt_t_opt = C * t_orig;
            } else {
                gt_R_opt = R_orig;
                gt_t_opt = t_orig;
            }
        }

        // Run pipeline
        auto result = pipeline->refine(img, inv_depth, baseline, gt_R_opt, gt_t_opt);

        auto t3 = Clock::now();

        auto bgrMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        auto resizeMs = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        auto refineMs = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
        LOGD("bgr=%lldms resize=%lldms refine=%lldms total=%lldms matches=%d tri=%d",
             bgrMs, resizeMs, refineMs, bgrMs + resizeMs + refineMs,
             result.num_matches, result.num_valid_tri);

        // Build DepthResult Java object
        jclass resultClass = env->FindClass("com/prdepth/android/DepthResult");
        if (!resultClass) { LOGE("DepthResult class not found"); return nullptr; }

        jmethodID constructor = env->GetMethodID(resultClass, "<init>",
            "([F[F[F[F[FFIIZF[III)V");
        if (!constructor) { LOGE("DepthResult constructor not found"); return nullptr; }

        jfloatArray refinedDepth = matToJavaFloatArray(env, result.z_refined);
        jfloatArray triDepth = matToJavaFloatArray(env, result.z_tri);
        jfloatArray confidence = matToJavaFloatArray(env, result.confidence);
        jfloatArray R_arr = eigenMatrix3dToJavaFloatArray(env, result.R);
        jfloatArray t_arr = eigenVector3dToJavaFloatArray(env, result.t);

        // Flow visualization (Middlebury color wheel)
        jintArray flowPixels = flowToARGB(env, result.flow);
        int flowW = result.flow.empty() ? 0 : result.flow.cols;
        int flowH = result.flow.empty() ? 0 : result.flow.rows;

        return env->NewObject(resultClass, constructor,
            refinedDepth, triDepth, confidence, R_arr, t_arr,
            result.baseline, result.num_matches, result.num_valid_tri,
            result.used_gt_pose, result.rotation_angle_deg,
            flowPixels, flowW, flowH);

    } catch (const std::exception& e) {
        LOGE("processFrame failed: %s", e.what());
        return nullptr;
    }
}

/**
 * Update full pipeline configuration at runtime.
 * Accepts key tunable parameters as individual values.
 */
JNIEXPORT void JNICALL
Java_com_prdepth_android_DepthRefinementManager_nativeUpdateFullConfig(
    JNIEnv* env, jobject obj,
    jlong pipelinePtr,
    jint ransacIters,
    jfloat minFlowPx,
    jfloat maxDepth,
    jfloat minBaseline,
    jfloat fusionLambdaForget,
    jfloat fusionChi2Soft,
    jfloat fusionVarFloor,
    jboolean useSegmentation,
    jboolean enableIterativeRefinement,
    jboolean skipFbConsistency,
    jboolean useGTPose,
    jboolean timing,
    jfloat skyMaskInvThresh
) {
    auto* pipeline = reinterpret_cast<pr_depth::DepthRefinement*>(pipelinePtr);
    if (!pipeline) return;

    try {
        auto config = pipeline->getConfig();

        config.ransac_max_iters = ransacIters;
        config.min_flow_px = minFlowPx;
        config.max_depth = maxDepth;
        config.min_baseline = minBaseline;
        config.use_baseline_guard = (minBaseline > 0.0f);
        config.fusion_lambda_forget = fusionLambdaForget;
        config.fusion_chi2_soft = fusionChi2Soft;
        config.fusion_var_floor = fusionVarFloor;
        config.use_segmentation = useSegmentation;
        config.enable_iterative_refinement = enableIterativeRefinement;
        config.skip_fb_consistency = !skipFbConsistency;  // UI: "FB Consistency" ON = skip=false
        config.use_gt_pose_fallback = useGTPose;
        config.timing = timing;
        config.sky_mask_inv_thresh = skyMaskInvThresh;

        pipeline->updateConfig(config);

        LOGI("Full config updated: ransac=%d maxD=%.0f lambda=%.2f chi2=%.1f skyThresh=%.1e",
             ransacIters, maxDepth, fusionLambdaForget, fusionChi2Soft, skyMaskInvThresh);

    } catch (const std::exception& e) {
        LOGE("Failed to update config: %s", e.what());
    }
}

/**
 * Legacy config update (GT pose only)
 */
JNIEXPORT void JNICALL
Java_com_prdepth_android_DepthRefinementManager_nativeUpdateConfig(
    JNIEnv* env, jobject obj,
    jlong pipelinePtr,
    jboolean useGTPose,
    jboolean useGTR
) {
    auto* pipeline = reinterpret_cast<pr_depth::DepthRefinement*>(pipelinePtr);
    if (!pipeline) return;

    try {
        auto config = pipeline->getConfig();
        config.use_gt_pose_fallback = useGTPose;
        config.use_gt_R = useGTR;
        pipeline->updateConfig(config);
    } catch (const std::exception& e) {
        LOGE("Failed to update config: %s", e.what());
    }
}

JNIEXPORT void JNICALL
Java_com_prdepth_android_DepthRefinementManager_nativeReset(
    JNIEnv* env, jobject obj,
    jlong pipelinePtr
) {
    auto* pipeline = reinterpret_cast<pr_depth::DepthRefinement*>(pipelinePtr);
    if (!pipeline) return;
    try { pipeline->reset(); LOGI("Pipeline reset"); }
    catch (const std::exception& e) { LOGE("Reset failed: %s", e.what()); }
}

JNIEXPORT void JNICALL
Java_com_prdepth_android_DepthRefinementManager_nativeDestroyPipeline(
    JNIEnv* env, jobject obj,
    jlong pipelinePtr
) {
    auto* pipeline = reinterpret_cast<pr_depth::DepthRefinement*>(pipelinePtr);
    if (pipeline) {
        delete pipeline;
        LOGI("Pipeline destroyed");
    }
}

/**
 * Phase 1 of pipelined execution: Compute BGR from YUV + optical flow.
 * Call this while QNN inference is running on NPU.
 * The computed flow will be consumed by the next nativeProcessFrameYUV() call.
 */
JNIEXPORT void JNICALL
Java_com_prdepth_android_DepthRefinementManager_nativePrepareFlow(
    JNIEnv* env, jobject obj,
    jlong pipelinePtr,
    jbyteArray yArray, jbyteArray uArray, jbyteArray vArray,
    jint imgW, jint imgH,
    jint yRowStride, jint uvRowStride, jint uvPixelStride,
    jint rotDeg
) {
    auto* pipeline = reinterpret_cast<pr_depth::DepthRefinement*>(pipelinePtr);
    if (!pipeline) {
        LOGE("prepareFlow: Invalid pipeline pointer");
        return;
    }

    try {
        auto config = pipeline->getConfig();
        int targetW = config.W;
        int targetH = config.H;

        // Convert YUV to BGR at model resolution with rotation
        auto* yData = static_cast<uint8_t*>(env->GetPrimitiveArrayCritical(yArray, nullptr));
        auto* uData = static_cast<uint8_t*>(env->GetPrimitiveArrayCritical(uArray, nullptr));
        auto* vData = static_cast<uint8_t*>(env->GetPrimitiveArrayCritical(vArray, nullptr));

        cv::Mat img = buildBGRFromYUV(yData, uData, vData,
            imgW, imgH, yRowStride, uvRowStride, uvPixelStride,
            targetW, targetH, rotDeg);

        env->ReleasePrimitiveArrayCritical(yArray, yData, JNI_ABORT);
        env->ReleasePrimitiveArrayCritical(uArray, uData, JNI_ABORT);
        env->ReleasePrimitiveArrayCritical(vArray, vData, JNI_ABORT);

        // Compute optical flow (stores internally for next refine() call)
        pipeline->prepare_flow(img);
    } catch (const std::exception& e) {
        LOGE("prepareFlow failed: %s", e.what());
    }
}

}  // extern "C"
