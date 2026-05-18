/**
 * JNI Bridge for PTC-Depth Android
 *
 * Connects Java/Kotlin to the PTC-Depth C++ pipeline (ptc_depth::PTCDepth).
 *
 * Accepts YUV camera planes directly (same format as QNN bridge) and builds
 * a grayscale image using the same rotation+resize mapping as the QNN
 * preprocessor — guaranteeing spatial alignment between camera image and
 * QNN depth output.
 *
 * Real-time optimization preserved from the previous (pr_depth) bridge:
 *   - prepareFlow(): runs DIS optical flow on the CPU while the NPU is busy
 *     running depth inference. The result is fed to the next refine() call so
 *     the C++ pipeline never has to recompute flow synchronously.
 */

#include <jni.h>
#include <android/log.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <Eigen/Dense>
#include <memory>
#include <string>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <omp.h>

#include "ptc_depth/ptc_depth.hpp"
#include "ptc_depth/depth_warp.hpp"  // rotation_matrix_to_omega

#define LOG_TAG "PTC-Depth-JNI"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

// ============================================================================
// Pipeline state (one struct per Kotlin DepthRefinementManager instance)
// ============================================================================

struct PipelineHandle {
    ptc_depth::PTCDepth pipeline;

    // External flow pre-computation state (NPU/CPU pipelining).
    // Mirrors PTC-Depth's internal prev_img_ — we keep our own so we can run
    // flow on the CPU before refine() is called.
    cv::Mat prev_gray;
    cv::Mat precomputed_flow;
    bool flow_ready = false;
    cv::Ptr<cv::DISOpticalFlow> dis_flow;

    // Saved external pose flag for diagnostics.
    bool last_used_external_pose = false;
    float last_rotation_angle_deg = 0.0f;

    explicit PipelineHandle(const ptc_depth::PTCDepthConfig& cfg)
        : pipeline(cfg) {
        // PRESET_FAST ~3x faster than PRESET_MEDIUM at 480x640. RANSAC inside
        // PTC-Depth filters most of the noise so the accuracy loss on dense
        // matches is tolerable.
        dis_flow = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_FAST);
        dis_flow->setFinestScale(1);  // skip the finest pyramid level
        dis_flow->setUseSpatialPropagation(true);
    }
};

// ============================================================================
// YUV → grayscale (with rotation + resize) — identical mapping to QNN bridge
// ============================================================================

static cv::Mat buildGrayFromYUV(
    const uint8_t* yData,
    int imgW, int imgH, int yRowStride,
    int targetW, int targetH, int rotDeg)
{
    cv::Mat gray(targetH, targetW, CV_8UC1);

    for (int ty = 0; ty < targetH; ty++) {
        auto* row = gray.ptr<uint8_t>(ty);
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

            row[tx] = yData[sY * yRowStride + sX];
        }
    }

    return gray;
}

// ============================================================================
// Java <-> C++ conversion helpers
// ============================================================================

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
    if (mat.empty()) {
        jfloatArray arr = env->NewFloatArray(0);
        return arr;
    }
    int size = mat.rows * mat.cols;
    jfloatArray arr = env->NewFloatArray(size);
    if (mat.type() == CV_32F && mat.isContinuous()) {
        env->SetFloatArrayRegion(arr, 0, size, (jfloat*)mat.data);
    } else {
        cv::Mat floatMat;
        mat.convertTo(floatMat, CV_32F);
        if (!floatMat.isContinuous()) floatMat = floatMat.clone();
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
// JNI exports
// ============================================================================

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_ptcdepth_android_DepthRefinementManager_nativeCreatePipeline(
    JNIEnv* env, jobject /*obj*/,
    jfloat fx, jfloat fy, jfloat cx, jfloat cy,
    jint width, jint height,
    jboolean /*useGTPose*/  // legacy — external pose is now passed per-frame
) {
    LOGI("Creating PTC-Depth pipeline: %dx%d, fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f",
         width, height, fx, fy, cx, cy);

    LOGI("OpenMP default max_threads = %d", omp_get_max_threads());

    try {
        ptc_depth::PTCDepthConfig config;
        config.fx = fx;
        config.fy = fy;
        config.cx = cx;
        config.cy = cy;
        config.W = width;
        config.H = height;
        config.sync();

        auto* handle = new PipelineHandle(config);
        LOGI("PTC-Depth pipeline created at %p", handle);
        return reinterpret_cast<jlong>(handle);
    } catch (const std::exception& e) {
        LOGE("Failed to create pipeline: %s", e.what());
        return 0;
    }
}

// Helper: copy a CV_32F cv::Mat row-by-row into a pre-allocated jfloatArray.
// Returns true on success. Skips allocation entirely — the destination array
// is owned by Kotlin (ping-pong buffer).
static bool writeMatToFloatArray(JNIEnv* env, const cv::Mat& mat, jfloatArray dst,
                                  int expectedRows, int expectedCols) {
    const int N = expectedRows * expectedCols;
    if (env->GetArrayLength(dst) < N) {
        LOGE("writeMatToFloatArray: destination too small (%d < %d)",
             env->GetArrayLength(dst), N);
        return false;
    }
    jfloat* ptr = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(dst, nullptr));
    if (!ptr) return false;
    if (mat.empty()) {
        // Pipeline produced no result for this frame — zero the buffer so the
        // consumer sees a defined state.
        std::memset(ptr, 0, sizeof(float) * N);
    } else if (mat.type() == CV_32F && mat.isContinuous() &&
               mat.rows == expectedRows && mat.cols == expectedCols) {
        std::memcpy(ptr, mat.data, sizeof(float) * N);
    } else {
        // Resize / type-convert defensively if pipeline returned unexpected shape.
        cv::Mat tmp;
        cv::Mat src = mat;
        if (src.type() != CV_32F) src.convertTo(src, CV_32F);
        if (src.rows != expectedRows || src.cols != expectedCols) {
            cv::resize(src, tmp, cv::Size(expectedCols, expectedRows));
            src = tmp;
        }
        if (!src.isContinuous()) src = src.clone();
        std::memcpy(ptr, src.data, sizeof(float) * N);
    }
    env->ReleasePrimitiveArrayCritical(dst, ptr, 0);  // 0 = commit changes
    return true;
}

JNIEXPORT jboolean JNICALL
Java_com_ptcdepth_android_DepthRefinementManager_nativeProcessFrameYUV(
    JNIEnv* env, jobject /*obj*/,
    jlong pipelinePtr,
    jbyteArray yArray, jbyteArray /*uArray*/, jbyteArray /*vArray*/,
    jint imgW, jint imgH,
    jint yRowStride, jint /*uvRowStride*/, jint /*uvPixelStride*/,
    jint rotDeg,
    jfloatArray invDepth, jint depthW, jint depthH,
    jfloat baseline,
    jfloatArray gtR, jfloatArray gtT,
    jfloatArray refinedOut, jfloatArray triOut,
    jfloatArray rOut, jfloatArray tOut, jfloatArray scalarOut)
{
    using Clock = std::chrono::high_resolution_clock;

    auto* handle = reinterpret_cast<PipelineHandle*>(pipelinePtr);
    if (!handle) {
        LOGE("Invalid pipeline pointer");
        return JNI_FALSE;
    }

    try {
        auto config = handle->pipeline.config();
        int targetW = config.W;
        int targetH = config.H;

        auto t0 = Clock::now();

        // Build grayscale (skip if prepareFlow() already produced a gray + flow)
        cv::Mat gray;
        if (!handle->flow_ready) {
            auto* yData = static_cast<uint8_t*>(env->GetPrimitiveArrayCritical(yArray, nullptr));
            gray = buildGrayFromYUV(yData,
                imgW, imgH, yRowStride,
                targetW, targetH, rotDeg);
            env->ReleasePrimitiveArrayCritical(yArray, yData, JNI_ABORT);
        } else {
            gray = handle->prev_gray;  // already converted in prepareFlow()
        }

        auto t1 = Clock::now();

        cv::Mat inv_depth_display = javaFloatArrayToMat(env, invDepth, depthH, depthW);
        cv::Mat inv_depth;
        if (depthW != targetW || depthH != targetH) {
            cv::resize(inv_depth_display, inv_depth, cv::Size(targetW, targetH),
                       0, 0, cv::INTER_LINEAR);
        } else {
            inv_depth = inv_depth_display;
        }

        auto t2 = Clock::now();

        // Optional external pose (rotate from camera frame into rotated-image frame).
        //
        // Kotlin passes `gtT` as a UNIT direction vector (ARCoreManager normalizes
        // it). PTC-Depth's setup_pose() short-circuits motion estimation when both
        // external_R AND external_t are supplied, using them as-is for the
        // p_curr = R·p_prev + t pose — so the translation MUST be in metric units
        // (||t|| = baseline). Scale by baseline here to match the internal
        // motion-estimation path which calls normalize_t(t_dir, baseline).
        std::optional<Eigen::Matrix3d> ext_R;
        std::optional<Eigen::Vector3d> ext_t;
        bool used_external = false;

        if (gtR != nullptr && gtT != nullptr) {
            Eigen::Matrix3d R_orig = javaFloatArrayToEigenMatrix3d(env, gtR);
            Eigen::Vector3d t_orig = javaFloatArrayToEigenVector3d(env, gtT);

            Eigen::Matrix3d C = Eigen::Matrix3d::Identity();
            if (rotDeg == 90)        { C <<  0, -1, 0,   1, 0, 0,   0, 0, 1; }
            else if (rotDeg == 270)  { C <<  0,  1, 0,  -1, 0, 0,   0, 0, 1; }
            else if (rotDeg == 180)  { C << -1,  0, 0,   0,-1, 0,   0, 0, 1; }

            ext_R = C * R_orig * C.transpose();
            // Re-normalize first (defense against non-unit input) then scale to baseline.
            Eigen::Vector3d t_rotated = C * t_orig;
            double tn = t_rotated.norm();
            if (tn > 1e-8) {
                ext_t = (t_rotated / tn) * static_cast<double>(baseline);
            } else {
                ext_t = Eigen::Vector3d(0.0, 0.0, -static_cast<double>(baseline));
            }
            used_external = true;
        }
        handle->last_used_external_pose = used_external;

        // Use precomputed flow if available (NPU/CPU pipelined path)
        cv::Mat flow = handle->flow_ready ? handle->precomputed_flow : cv::Mat();

        ptc_depth::ScaleFusionResult result = handle->pipeline.refine(
            gray, inv_depth, baseline, ext_R, ext_t, cv::Mat(), flow);

        // Update JNI-side prev frame for the next prepareFlow() call.
        handle->prev_gray = gray;
        handle->precomputed_flow.release();
        handle->flow_ready = false;

        auto t3 = Clock::now();

        auto grayMs   = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        auto resizeMs = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        auto refineMs = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
        LOGD("gray=%lldms resize=%lldms refine=%lldms total=%lldms",
             grayMs, resizeMs, refineMs, grayMs + resizeMs + refineMs);

        // Extract R/t from 4x4 pose. PTC-Depth pose convention: p_curr = R p_prev + t
        Eigen::Matrix3d R = result.pose.block<3,3>(0,0);
        Eigen::Vector3d t = result.pose.block<3,1>(0,3);

        // Rotation magnitude in degrees (used for status text).
        Eigen::Vector3d omega = ptc_depth::rotation_matrix_to_omega(R);
        float rotation_angle_deg = static_cast<float>(omega.norm() * 180.0 / M_PI);
        if (!std::isfinite(rotation_angle_deg)) rotation_angle_deg = 0.0f;
        handle->last_rotation_angle_deg = rotation_angle_deg;

        // Count valid triangulated points (non-NaN, positive z_obs).
        int num_valid_tri = 0;
        if (!result.z_obs.empty()) {
            const float* p = result.z_obs.ptr<float>(0);
            int N = result.z_obs.rows * result.z_obs.cols;
            for (int i = 0; i < N; ++i)
                if (std::isfinite(p[i]) && p[i] > 0.0f) num_valid_tri++;
        }
        int num_matches = num_valid_tri;  // PTC-Depth dense triangulation: matches ≈ valid

        // Zero-allocation result write: dump cv::Mat data straight into the
        // caller-supplied ping-pong FloatArray buffers + scalar buffer. No
        // env->NewFloatArray() / env->NewObject() in this hot path.
        if (!writeMatToFloatArray(env, result.z_refined, refinedOut, targetH, targetW)) return JNI_FALSE;
        if (!writeMatToFloatArray(env, result.z_obs,     triOut,     targetH, targetW)) return JNI_FALSE;

        {
            jfloat* rPtr = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(rOut, nullptr));
            if (rPtr) {
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        rPtr[i * 3 + j] = static_cast<float>(R(i, j));
                env->ReleasePrimitiveArrayCritical(rOut, rPtr, 0);
            }
        }
        {
            jfloat* tPtr = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(tOut, nullptr));
            if (tPtr) {
                tPtr[0] = static_cast<float>(t(0));
                tPtr[1] = static_cast<float>(t(1));
                tPtr[2] = static_cast<float>(t(2));
                env->ReleasePrimitiveArrayCritical(tOut, tPtr, 0);
            }
        }
        {
            jfloat* sPtr = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(scalarOut, nullptr));
            if (sPtr) {
                sPtr[0] = baseline;
                sPtr[1] = static_cast<float>(num_matches);
                sPtr[2] = static_cast<float>(num_valid_tri);
                sPtr[3] = used_external ? 1.0f : 0.0f;
                sPtr[4] = rotation_angle_deg;
                env->ReleasePrimitiveArrayCritical(scalarOut, sPtr, 0);
            }
        }

        return JNI_TRUE;

    } catch (const std::exception& e) {
        LOGE("processFrame failed: %s", e.what());
        return JNI_FALSE;
    }
}

JNIEXPORT void JNICALL
Java_com_ptcdepth_android_DepthRefinementManager_nativeUpdateFullConfig(
    JNIEnv* /*env*/, jobject /*obj*/,
    jlong pipelinePtr,
    jint ransacIters,
    jfloat minFlowPx,
    jfloat maxDepth,
    jfloat minBaseline,
    jfloat lambdaForget,
    jfloat kappaMin,
    jfloat tau0Deg,
    jboolean outdoor,
    jint iterative,
    jboolean verbose)
{
    auto* handle = reinterpret_cast<PipelineHandle*>(pipelinePtr);
    if (!handle) return;

    try {
        // PTCDepth doesn't expose mutable config_ directly. We rebuild the pipeline
        // only if structural parameters change — for the tunable knobs we update
        // the config on the existing instance via reset().
        auto cfg = handle->pipeline.config();
        cfg.ransac_max_iters = ransacIters;
        cfg.min_flow_px      = minFlowPx;
        cfg.max_depth        = maxDepth;
        cfg.min_baseline     = minBaseline;
        cfg.lambda_forget    = lambdaForget;
        cfg.kappa_min        = kappaMin;
        cfg.tau0_deg         = tau0Deg;
        cfg.outdoor          = outdoor;
        cfg.iterative        = iterative;
        cfg.verbose          = verbose;
        cfg.sync();

        // Recreate the pipeline with the new config (preserves the JNI-side
        // prev_gray / flow state so NPU/CPU pipelining keeps working).
        handle->pipeline.~PTCDepth();
        new (&handle->pipeline) ptc_depth::PTCDepth(cfg);

        LOGI("Config updated: ransac=%d maxD=%.1f lambda=%.2f kappa=%.2f outdoor=%d iter=%d",
             ransacIters, maxDepth, lambdaForget, kappaMin, (int)outdoor, iterative);

    } catch (const std::exception& e) {
        LOGE("Failed to update config: %s", e.what());
    }
}

JNIEXPORT void JNICALL
Java_com_ptcdepth_android_DepthRefinementManager_nativeReset(
    JNIEnv* /*env*/, jobject /*obj*/,
    jlong pipelinePtr)
{
    auto* handle = reinterpret_cast<PipelineHandle*>(pipelinePtr);
    if (!handle) return;
    try {
        handle->pipeline.reset();
        handle->prev_gray.release();
        handle->precomputed_flow.release();
        handle->flow_ready = false;
        LOGI("Pipeline reset");
    } catch (const std::exception& e) {
        LOGE("Reset failed: %s", e.what());
    }
}

JNIEXPORT void JNICALL
Java_com_ptcdepth_android_DepthRefinementManager_nativeDestroyPipeline(
    JNIEnv* /*env*/, jobject /*obj*/,
    jlong pipelinePtr)
{
    auto* handle = reinterpret_cast<PipelineHandle*>(pipelinePtr);
    if (handle) {
        delete handle;
        LOGI("Pipeline destroyed");
    }
}

/**
 * Phase 1 of pipelined execution: build grayscale + compute optical flow on CPU
 * while QNN runs depth inference on the NPU. The flow gets fed into the next
 * processFrameYUV() call so refine() can skip its own flow computation.
 */
JNIEXPORT void JNICALL
Java_com_ptcdepth_android_DepthRefinementManager_nativePrepareFlow(
    JNIEnv* env, jobject /*obj*/,
    jlong pipelinePtr,
    jbyteArray yArray, jbyteArray /*uArray*/, jbyteArray /*vArray*/,
    jint imgW, jint imgH,
    jint yRowStride, jint /*uvRowStride*/, jint /*uvPixelStride*/,
    jint rotDeg)
{
    auto* handle = reinterpret_cast<PipelineHandle*>(pipelinePtr);
    if (!handle) {
        LOGE("prepareFlow: Invalid pipeline pointer");
        return;
    }

    try {
        using Clock = std::chrono::high_resolution_clock;
        auto t0 = Clock::now();

        auto cfg = handle->pipeline.config();
        int targetW = cfg.W;
        int targetH = cfg.H;

        auto* yData = static_cast<uint8_t*>(env->GetPrimitiveArrayCritical(yArray, nullptr));
        cv::Mat gray_curr = buildGrayFromYUV(yData,
            imgW, imgH, yRowStride,
            targetW, targetH, rotDeg);
        env->ReleasePrimitiveArrayCritical(yArray, yData, JNI_ABORT);

        auto t1 = Clock::now();

        if (handle->prev_gray.empty()) {
            // First frame: no flow yet, just remember gray.
            handle->prev_gray = gray_curr;
            handle->flow_ready = false;
            return;
        }

        cv::Mat flow;
        handle->dis_flow->calc(handle->prev_gray, gray_curr, flow);

        // Stash flow + new gray for the upcoming refine() call.
        handle->precomputed_flow = flow;
        handle->prev_gray = gray_curr;
        handle->flow_ready = true;

        auto t2 = Clock::now();
        auto grayMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        auto flowMs = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        LOGD("prepareFlow: gray=%lldms flow=%lldms total=%lldms", grayMs, flowMs, grayMs + flowMs);

    } catch (const std::exception& e) {
        LOGE("prepareFlow failed: %s", e.what());
    }
}

}  // extern "C"
