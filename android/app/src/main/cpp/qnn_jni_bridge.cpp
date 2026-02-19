#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <chrono>
#include "qnn_depth_estimator.h"

#define LOG_TAG "QnnJniBridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_prdepth_android_DepthEstimatorQNN_nativeInit(
        JNIEnv* env, jobject /*thiz*/,
        jstring modelPath, jstring nativeLibDir) {

    const char* modelPathStr = env->GetStringUTFChars(modelPath, nullptr);
    const char* nativeLibDirStr = env->GetStringUTFChars(nativeLibDir, nullptr);

    LOGI("nativeInit: model=%s, libDir=%s", modelPathStr, nativeLibDirStr);

    auto* estimator = new QnnDepthEstimator();
    bool ok = estimator->initialize(nativeLibDirStr, modelPathStr);

    env->ReleaseStringUTFChars(modelPath, modelPathStr);
    env->ReleaseStringUTFChars(nativeLibDir, nativeLibDirStr);

    if (!ok) {
        LOGE("QNN initialization failed");
        delete estimator;
        return 0;
    }

    LOGI("QNN initialized, input=%u, output=%u",
         estimator->getInputElementCount(), estimator->getOutputElementCount());
    return reinterpret_cast<jlong>(estimator);
}

JNIEXPORT jfloatArray JNICALL
Java_com_prdepth_android_DepthEstimatorQNN_nativeInfer(
        JNIEnv* env, jobject /*thiz*/,
        jlong handle, jfloatArray inputArray) {

    auto* estimator = reinterpret_cast<QnnDepthEstimator*>(handle);
    if (!estimator || !estimator->isInitialized()) {
        LOGE("nativeInfer: invalid handle or not initialized");
        return nullptr;
    }

    jsize inputLen = env->GetArrayLength(inputArray);
    if (static_cast<uint32_t>(inputLen) != estimator->getInputElementCount()) {
        LOGE("Input size mismatch: got %d, expected %u",
             inputLen, estimator->getInputElementCount());
        return nullptr;
    }

    jfloat* inputData = env->GetFloatArrayElements(inputArray, nullptr);

    uint32_t outputCount = estimator->getOutputElementCount();
    std::vector<float> outputData(outputCount);

    bool ok = estimator->infer(inputData, outputData.data());
    env->ReleaseFloatArrayElements(inputArray, inputData, JNI_ABORT);

    if (!ok) {
        LOGE("nativeInfer: inference failed");
        return nullptr;
    }

    jfloatArray result = env->NewFloatArray(static_cast<jsize>(outputCount));
    env->SetFloatArrayRegion(result, 0, static_cast<jsize>(outputCount), outputData.data());
    return result;
}

JNIEXPORT jint JNICALL
Java_com_prdepth_android_DepthEstimatorQNN_nativeGetInputSize(
        JNIEnv* /*env*/, jobject /*thiz*/, jlong handle) {
    auto* estimator = reinterpret_cast<QnnDepthEstimator*>(handle);
    if (!estimator) return 0;
    return static_cast<jint>(estimator->getInputElementCount());
}

JNIEXPORT jint JNICALL
Java_com_prdepth_android_DepthEstimatorQNN_nativeGetOutputSize(
        JNIEnv* /*env*/, jobject /*thiz*/, jlong handle) {
    auto* estimator = reinterpret_cast<QnnDepthEstimator*>(handle);
    if (!estimator) return 0;
    return static_cast<jint>(estimator->getOutputElementCount());
}

JNIEXPORT void JNICALL
Java_com_prdepth_android_DepthEstimatorQNN_nativeDestroy(
        JNIEnv* /*env*/, jobject /*thiz*/, jlong handle) {
    auto* estimator = reinterpret_cast<QnnDepthEstimator*>(handle);
    if (estimator) {
        estimator->destroy();
        delete estimator;
        LOGI("nativeDestroy: QNN estimator released");
    }
}

JNIEXPORT void JNICALL
Java_com_prdepth_android_DepthEstimatorQNN_nativePausePerf(
        JNIEnv* /*env*/, jobject /*thiz*/, jlong handle) {
    auto* estimator = reinterpret_cast<QnnDepthEstimator*>(handle);
    if (estimator && estimator->isInitialized()) {
        estimator->pausePerf();
    }
}

JNIEXPORT void JNICALL
Java_com_prdepth_android_DepthEstimatorQNN_nativeResumePerf(
        JNIEnv* /*env*/, jobject /*thiz*/, jlong handle) {
    auto* estimator = reinterpret_cast<QnnDepthEstimator*>(handle);
    if (estimator && estimator->isInitialized()) {
        estimator->resumePerf();
    }
}

JNIEXPORT jfloatArray JNICALL
Java_com_prdepth_android_DepthEstimatorQNN_nativePreprocessYUV(
        JNIEnv* env, jobject /*thiz*/,
        jobject yBuffer, jobject uBuffer, jobject vBuffer,
        jint imgWidth, jint imgHeight,
        jint yRowStride, jint uvRowStride, jint uvPixelStride,
        jint targetWidth, jint targetHeight,
        jint rotationDegrees) {

    auto* yData = static_cast<const uint8_t*>(env->GetDirectBufferAddress(yBuffer));
    auto* uData = static_cast<const uint8_t*>(env->GetDirectBufferAddress(uBuffer));
    auto* vData = static_cast<const uint8_t*>(env->GetDirectBufferAddress(vBuffer));

    if (!yData || !uData || !vData) {
        LOGE("nativePreprocessYUV: null buffer address");
        return nullptr;
    }

    int tW = targetWidth;
    int tH = targetHeight;
    int srcW = imgWidth;
    int srcH = imgHeight;
    int pixelCount = tW * tH;

    // Output: NCHW [1, 3, tH, tW] float in [0, 1]
    std::vector<float> output(3 * pixelCount);

    for (int ty = 0; ty < tH; ty++) {
        for (int tx = 0; tx < tW; tx++) {
            float srcXf, srcYf;

            if (rotationDegrees == 90) {
                // 90° CW: rotated image is srcH wide x srcW tall
                // target(tx,ty) maps to source(col, row):
                //   col = ty * (srcW-1) / (tH-1)
                //   row = (srcH-1) - tx * (srcH-1) / (tW-1)
                srcXf = static_cast<float>(ty) * (srcW - 1) / (tH - 1);
                srcYf = (srcH - 1) - static_cast<float>(tx) * (srcH - 1) / (tW - 1);
            } else if (rotationDegrees == 270) {
                srcXf = (srcW - 1) - static_cast<float>(ty) * (srcW - 1) / (tH - 1);
                srcYf = static_cast<float>(tx) * (srcH - 1) / (tW - 1);
            } else if (rotationDegrees == 180) {
                srcXf = (srcW - 1) - static_cast<float>(tx) * (srcW - 1) / (tW - 1);
                srcYf = (srcH - 1) - static_cast<float>(ty) * (srcH - 1) / (tH - 1);
            } else {
                // 0° — no rotation
                srcXf = static_cast<float>(tx) * (srcW - 1) / (tW - 1);
                srcYf = static_cast<float>(ty) * (srcH - 1) / (tH - 1);
            }

            int srcX = static_cast<int>(srcXf + 0.5f);
            int srcY = static_cast<int>(srcYf + 0.5f);
            srcX = std::max(0, std::min(srcX, srcW - 1));
            srcY = std::max(0, std::min(srcY, srcH - 1));

            // Read YUV
            int yIdx = srcY * yRowStride + srcX;
            int uvIdx = (srcY / 2) * uvRowStride + (srcX / 2) * uvPixelStride;

            int Y = yData[yIdx];
            int U = uData[uvIdx] - 128;
            int V = vData[uvIdx] - 128;

            // YUV BT.601 to RGB
            int r = Y + static_cast<int>(1.370705f * V);
            int g = Y - static_cast<int>(0.337633f * U) - static_cast<int>(0.698001f * V);
            int b = Y + static_cast<int>(1.732446f * U);

            r = std::max(0, std::min(255, r));
            g = std::max(0, std::min(255, g));
            b = std::max(0, std::min(255, b));

            // NCHW: R plane, G plane, B plane, normalized to [0, 1]
            int idx = ty * tW + tx;
            output[idx] = r / 255.0f;
            output[pixelCount + idx] = g / 255.0f;
            output[2 * pixelCount + idx] = b / 255.0f;
        }
    }

    jfloatArray result = env->NewFloatArray(3 * pixelCount);
    env->SetFloatArrayRegion(result, 0, 3 * pixelCount, output.data());
    return result;
}

JNIEXPORT jfloatArray JNICALL
Java_com_prdepth_android_DepthEstimatorQNN_nativeResizeDepth(
        JNIEnv* env, jobject /*thiz*/,
        jfloatArray depthArray,
        jint srcW, jint srcH,
        jint dstW, jint dstH) {

    jsize len = env->GetArrayLength(depthArray);
    if (len != srcW * srcH) {
        LOGE("nativeResizeDepth: size mismatch: %d != %d*%d", len, srcW, srcH);
        return nullptr;
    }

    jfloat* srcData = env->GetFloatArrayElements(depthArray, nullptr);
    int outSize = dstW * dstH;
    std::vector<float> output(outSize);

    // Bilinear interpolation
    for (int dy = 0; dy < dstH; dy++) {
        float srcYf = static_cast<float>(dy) * (srcH - 1) / (dstH - 1);
        int sy0 = static_cast<int>(srcYf);
        int sy1 = std::min(sy0 + 1, srcH - 1);
        float fy = srcYf - sy0;

        for (int dx = 0; dx < dstW; dx++) {
            float srcXf = static_cast<float>(dx) * (srcW - 1) / (dstW - 1);
            int sx0 = static_cast<int>(srcXf);
            int sx1 = std::min(sx0 + 1, srcW - 1);
            float fx = srcXf - sx0;

            float v00 = srcData[sy0 * srcW + sx0];
            float v10 = srcData[sy0 * srcW + sx1];
            float v01 = srcData[sy1 * srcW + sx0];
            float v11 = srcData[sy1 * srcW + sx1];

            output[dy * dstW + dx] = v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) +
                                     v01 * (1 - fx) * fy + v11 * fx * fy;
        }
    }

    env->ReleaseFloatArrayElements(depthArray, srcData, JNI_ABORT);

    jfloatArray result = env->NewFloatArray(outSize);
    env->SetFloatArrayRegion(result, 0, outSize, output.data());
    return result;
}

// ========== Pipelined approach: preprocess on GL thread, infer on depth thread ==========
// Native double-buffer pool to avoid per-frame heap allocation
static float* g_prepBuffers[2] = {nullptr, nullptr};
static int g_prepBufIdx = 0;

/**
 * Preprocess YUV on GL thread → returns native pointer (zero-copy to depth thread).
 * Uses double-buffered native pool — no allocation after first 2 frames.
 */
JNIEXPORT jlong JNICALL
Java_com_prdepth_android_DepthEstimatorQNN_nativePreprocessYUVNative(
        JNIEnv* env, jobject /*thiz*/,
        jobject yBuffer, jobject uBuffer, jobject vBuffer,
        jint imgWidth, jint imgHeight,
        jint yRowStride, jint uvRowStride, jint uvPixelStride,
        jint targetWidth, jint targetHeight,
        jint rotationDegrees) {

    auto* yData = static_cast<const uint8_t*>(env->GetDirectBufferAddress(yBuffer));
    auto* uData = static_cast<const uint8_t*>(env->GetDirectBufferAddress(uBuffer));
    auto* vData = static_cast<const uint8_t*>(env->GetDirectBufferAddress(vBuffer));

    if (!yData || !uData || !vData) {
        LOGE("nativePreprocessYUVNative: null buffer");
        return 0;
    }

    int tW = targetWidth;
    int tH = targetHeight;
    int srcW = imgWidth;
    int srcH = imgHeight;
    int pixelCount = tW * tH;

    // Get buffer from pool (double-buffered, no allocation after first 2 calls)
    int idx = g_prepBufIdx;
    if (!g_prepBuffers[idx]) g_prepBuffers[idx] = new float[3 * pixelCount];
    g_prepBufIdx = 1 - idx;
    float* output = g_prepBuffers[idx];

    for (int ty = 0; ty < tH; ty++) {
        for (int tx = 0; tx < tW; tx++) {
            float srcXf, srcYf;

            if (rotationDegrees == 90) {
                srcXf = static_cast<float>(ty) * (srcW - 1) / (tH - 1);
                srcYf = (srcH - 1) - static_cast<float>(tx) * (srcH - 1) / (tW - 1);
            } else if (rotationDegrees == 270) {
                srcXf = (srcW - 1) - static_cast<float>(ty) * (srcW - 1) / (tH - 1);
                srcYf = static_cast<float>(tx) * (srcH - 1) / (tW - 1);
            } else if (rotationDegrees == 180) {
                srcXf = (srcW - 1) - static_cast<float>(tx) * (srcW - 1) / (tW - 1);
                srcYf = (srcH - 1) - static_cast<float>(ty) * (srcH - 1) / (tH - 1);
            } else {
                srcXf = static_cast<float>(tx) * (srcW - 1) / (tW - 1);
                srcYf = static_cast<float>(ty) * (srcH - 1) / (tH - 1);
            }

            int sX = static_cast<int>(srcXf + 0.5f);
            int sY = static_cast<int>(srcYf + 0.5f);
            sX = std::max(0, std::min(sX, srcW - 1));
            sY = std::max(0, std::min(sY, srcH - 1));

            int yIdx = sY * yRowStride + sX;
            int uvIdx = (sY / 2) * uvRowStride + (sX / 2) * uvPixelStride;

            int Y = yData[yIdx];
            int U = uData[uvIdx] - 128;
            int V = vData[uvIdx] - 128;

            int r = Y + static_cast<int>(1.370705f * V);
            int g = Y - static_cast<int>(0.337633f * U) - static_cast<int>(0.698001f * V);
            int b = Y + static_cast<int>(1.732446f * U);

            r = std::max(0, std::min(255, r));
            g = std::max(0, std::min(255, g));
            b = std::max(0, std::min(255, b));

            int i = ty * tW + tx;
            output[i] = r / 255.0f;
            output[pixelCount + i] = g / 255.0f;
            output[2 * pixelCount + i] = b / 255.0f;
        }
    }

    return reinterpret_cast<jlong>(output);
}

/**
 * Inference + postprocess + resize from native pointer.
 * Runs on depth thread. Does NOT free the native pointer (pool-managed).
 */
JNIEXPORT jboolean JNICALL
Java_com_prdepth_android_DepthEstimatorQNN_nativeInferFromNative(
        JNIEnv* env, jobject /*thiz*/,
        jlong handle, jlong nativeInputPtr,
        jfloatArray outputArray, jint dstW, jint dstH) {

    using Clock = std::chrono::high_resolution_clock;

    auto* estimator = reinterpret_cast<QnnDepthEstimator*>(handle);
    auto* inputData = reinterpret_cast<const float*>(nativeInputPtr);

    if (!estimator || !estimator->isInitialized() || !inputData) {
        LOGE("nativeInferFromNative: invalid params");
        return JNI_FALSE;
    }

    auto t0 = Clock::now();

    // Run inference
    uint32_t outputCount = estimator->getOutputElementCount();
    static thread_local std::vector<float> rawOutput;
    rawOutput.resize(outputCount);

    bool ok = estimator->infer(inputData, rawOutput.data());
    if (!ok) {
        LOGE("nativeInferFromNative: inference failed");
        return JNI_FALSE;
    }

    auto t1 = Clock::now();

    // Normalize + resize → output
    float minVal = rawOutput[0], maxVal = rawOutput[0];
    for (uint32_t i = 1; i < outputCount; i++) {
        if (rawOutput[i] < minVal) minVal = rawOutput[i];
        if (rawOutput[i] > maxVal) maxVal = rawOutput[i];
    }
    float range = maxVal - minVal;

    int outSrcSide = static_cast<int>(std::sqrt(static_cast<float>(outputCount)));

    jfloat* outData = static_cast<jfloat*>(
        env->GetPrimitiveArrayCritical(outputArray, nullptr));
    if (!outData) {
        LOGE("nativeInferFromNative: GetPrimitiveArrayCritical failed");
        return JNI_FALSE;
    }

    int outSize = dstW * dstH;
    if (range < 1e-6f) {
        for (int i = 0; i < outSize; i++) outData[i] = 0.5f;
    } else {
        float invRange = 1.0f / range;
        for (int dy = 0; dy < dstH; dy++) {
            float srcYf = static_cast<float>(dy) * (outSrcSide - 1) / (dstH - 1);
            int sy0 = static_cast<int>(srcYf);
            int sy1 = std::min(sy0 + 1, outSrcSide - 1);
            float fy = srcYf - sy0;

            for (int dx = 0; dx < dstW; dx++) {
                float srcXf = static_cast<float>(dx) * (outSrcSide - 1) / (dstW - 1);
                int sx0 = static_cast<int>(srcXf);
                int sx1 = std::min(sx0 + 1, outSrcSide - 1);
                float fx = srcXf - sx0;

                float v00 = rawOutput[sy0 * outSrcSide + sx0];
                float v10 = rawOutput[sy0 * outSrcSide + sx1];
                float v01 = rawOutput[sy1 * outSrcSide + sx0];
                float v11 = rawOutput[sy1 * outSrcSide + sx1];

                float val = v00*(1-fx)*(1-fy) + v10*fx*(1-fy) +
                           v01*(1-fx)*fy + v11*fx*fy;

                float normalized = (val - minVal) * invRange;
                outData[dy * dstW + dx] = normalized;  // True inverse depth: high=close, low=far
            }
        }
    }

    env->ReleasePrimitiveArrayCritical(outputArray, outData, 0);

    auto t2 = Clock::now();
    auto inferMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    auto postMs = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    LOGD("infer=%lldms post=%lldms total=%lldms", inferMs, postMs, inferMs + postMs);
    return JNI_TRUE;
}

// ========== Non-pipelined combined approach (kept for fallback) ==========

/**
 * Combined pipeline: YUV preprocess → inference → normalize → resize.
 * Uses static thread_local buffers to avoid per-frame heap allocation (~4MB saved).
 * Uses GetPrimitiveArrayCritical for zero-copy JNI access.
 */
JNIEXPORT jboolean JNICALL
Java_com_prdepth_android_DepthEstimatorQNN_nativeProcessFrame(
        JNIEnv* env, jobject /*thiz*/,
        jlong handle,
        jbyteArray yArray, jbyteArray uArray, jbyteArray vArray,
        jint imgWidth, jint imgHeight,
        jint yRowStride, jint uvRowStride, jint uvPixelStride,
        jint modelWidth, jint modelHeight,
        jint rotationDegrees,
        jfloatArray outputArray, jint dstW, jint dstH) {

    using Clock = std::chrono::high_resolution_clock;

    auto* estimator = reinterpret_cast<QnnDepthEstimator*>(handle);
    if (!estimator || !estimator->isInitialized()) {
        LOGE("nativeProcessFrame: invalid handle");
        return JNI_FALSE;
    }

    int srcW = imgWidth;
    int srcH = imgHeight;
    int tW = modelWidth;
    int tH = modelHeight;
    int pixelCount = tW * tH;

    auto t0 = Clock::now();

    // ---- Step 1: YUV → RGB + rotate + resize → NCHW float ----
    // Use GetPrimitiveArrayCritical for zero-copy access to byte arrays
    auto* yRaw = static_cast<const uint8_t*>(
        env->GetPrimitiveArrayCritical(yArray, nullptr));
    auto* uRaw = static_cast<const uint8_t*>(
        env->GetPrimitiveArrayCritical(uArray, nullptr));
    auto* vRaw = static_cast<const uint8_t*>(
        env->GetPrimitiveArrayCritical(vArray, nullptr));

    if (!yRaw || !uRaw || !vRaw) {
        if (yRaw) env->ReleasePrimitiveArrayCritical(yArray, (void*)yRaw, JNI_ABORT);
        if (uRaw) env->ReleasePrimitiveArrayCritical(uArray, (void*)uRaw, JNI_ABORT);
        if (vRaw) env->ReleasePrimitiveArrayCritical(vArray, (void*)vRaw, JNI_ABORT);
        LOGE("nativeProcessFrame: null array data");
        return JNI_FALSE;
    }

    // Reuse static buffers — no heap allocation after first frame
    static thread_local std::vector<float> inputData;
    inputData.resize(3 * pixelCount);

    for (int ty = 0; ty < tH; ty++) {
        for (int tx = 0; tx < tW; tx++) {
            float srcXf, srcYf;

            if (rotationDegrees == 90) {
                srcXf = static_cast<float>(ty) * (srcW - 1) / (tH - 1);
                srcYf = (srcH - 1) - static_cast<float>(tx) * (srcH - 1) / (tW - 1);
            } else if (rotationDegrees == 270) {
                srcXf = (srcW - 1) - static_cast<float>(ty) * (srcW - 1) / (tH - 1);
                srcYf = static_cast<float>(tx) * (srcH - 1) / (tW - 1);
            } else if (rotationDegrees == 180) {
                srcXf = (srcW - 1) - static_cast<float>(tx) * (srcW - 1) / (tW - 1);
                srcYf = (srcH - 1) - static_cast<float>(ty) * (srcH - 1) / (tH - 1);
            } else {
                srcXf = static_cast<float>(tx) * (srcW - 1) / (tW - 1);
                srcYf = static_cast<float>(ty) * (srcH - 1) / (tH - 1);
            }

            int sX = static_cast<int>(srcXf + 0.5f);
            int sY = static_cast<int>(srcYf + 0.5f);
            sX = std::max(0, std::min(sX, srcW - 1));
            sY = std::max(0, std::min(sY, srcH - 1));

            int yIdx = sY * yRowStride + sX;
            int uvIdx = (sY / 2) * uvRowStride + (sX / 2) * uvPixelStride;

            int Y = yRaw[yIdx];
            int U = uRaw[uvIdx] - 128;
            int V = vRaw[uvIdx] - 128;

            int r = Y + static_cast<int>(1.370705f * V);
            int g = Y - static_cast<int>(0.337633f * U) - static_cast<int>(0.698001f * V);
            int b = Y + static_cast<int>(1.732446f * U);

            r = std::max(0, std::min(255, r));
            g = std::max(0, std::min(255, g));
            b = std::max(0, std::min(255, b));

            int idx = ty * tW + tx;
            inputData[idx] = r / 255.0f;
            inputData[pixelCount + idx] = g / 255.0f;
            inputData[2 * pixelCount + idx] = b / 255.0f;
        }
    }

    // Release byte arrays before long-running inference
    env->ReleasePrimitiveArrayCritical(yArray, (void*)yRaw, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(uArray, (void*)uRaw, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(vArray, (void*)vRaw, JNI_ABORT);

    auto t1 = Clock::now();

    // ---- Step 2: Run inference ----
    uint32_t outputCount = estimator->getOutputElementCount();
    static thread_local std::vector<float> rawOutput;
    rawOutput.resize(outputCount);

    bool ok = estimator->infer(inputData.data(), rawOutput.data());

    if (!ok) {
        LOGE("nativeProcessFrame: inference failed");
        return JNI_FALSE;
    }

    auto t2 = Clock::now();

    // ---- Step 3: Normalize + resize → output array ----
    float minVal = rawOutput[0], maxVal = rawOutput[0];
    for (uint32_t i = 1; i < outputCount; i++) {
        if (rawOutput[i] < minVal) minVal = rawOutput[i];
        if (rawOutput[i] > maxVal) maxVal = rawOutput[i];
    }
    float range = maxVal - minVal;

    int outSrcSide = static_cast<int>(std::sqrt(static_cast<float>(outputCount)));
    int outSize = dstW * dstH;

    jfloat* outData = static_cast<jfloat*>(
        env->GetPrimitiveArrayCritical(outputArray, nullptr));
    if (!outData) {
        LOGE("nativeProcessFrame: GetPrimitiveArrayCritical failed");
        return JNI_FALSE;
    }

    if (range < 1e-6f) {
        for (int i = 0; i < outSize; i++) outData[i] = 0.5f;
    } else {
        float invRange = 1.0f / range;
        for (int dy = 0; dy < dstH; dy++) {
            float srcYf = static_cast<float>(dy) * (outSrcSide - 1) / (dstH - 1);
            int sy0 = static_cast<int>(srcYf);
            int sy1 = std::min(sy0 + 1, outSrcSide - 1);
            float fy = srcYf - sy0;

            for (int dx = 0; dx < dstW; dx++) {
                float srcXf = static_cast<float>(dx) * (outSrcSide - 1) / (dstW - 1);
                int sx0 = static_cast<int>(srcXf);
                int sx1 = std::min(sx0 + 1, outSrcSide - 1);
                float fx = srcXf - sx0;

                float v00 = rawOutput[sy0 * outSrcSide + sx0];
                float v10 = rawOutput[sy0 * outSrcSide + sx1];
                float v01 = rawOutput[sy1 * outSrcSide + sx0];
                float v11 = rawOutput[sy1 * outSrcSide + sx1];

                float val = v00*(1-fx)*(1-fy) + v10*fx*(1-fy) +
                           v01*(1-fx)*fy + v11*fx*fy;

                float normalized = (val - minVal) * invRange;
                outData[dy * dstW + dx] = normalized;  // True inverse depth: high=close, low=far
            }
        }
    }

    env->ReleasePrimitiveArrayCritical(outputArray, outData, 0);

    auto t3 = Clock::now();
    auto prepMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    auto inferMs = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    auto postMs = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
    LOGD("prep=%lldms infer=%lldms post=%lldms total=%lldms",
         prepMs, inferMs, postMs, prepMs + inferMs + postMs);
    return JNI_TRUE;
}

} // extern "C"
