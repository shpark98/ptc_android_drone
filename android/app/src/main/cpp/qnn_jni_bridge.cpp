/**
 * Native helpers for DepthEstimatorQNN.
 *
 * After migrating from direct QNN SDK to ONNX Runtime + QNN Execution Provider
 * (matches AI Hub's published benchmark setup), inference itself is handled
 * by ORT in Java. This file only retains two performance-sensitive helpers:
 *   - YUV (NV12/I420) → RGB → resize+rotate → NCHW float[1,3,518,518] in [0,1]
 *   - Bilinear resize of the float depth map to the display resolution.
 */

#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <cstdint>
#include <vector>

#define LOG_TAG "QnnJniBridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" {

// Zero-allocation YUV preprocessing: writes NCHW float[1,3,tH,tW] in [0,1]
// directly into a caller-supplied jfloatArray. No env->NewFloatArray() and no
// intermediate std::vector — saves ~3 MB Java heap + ~3 MB native heap per call.
JNIEXPORT jboolean JNICALL
Java_com_ptcdepth_android_DepthEstimatorQNN_nativePreprocessYUVBytesInto(
        JNIEnv* env, jobject /*thiz*/,
        jbyteArray yArray, jbyteArray uArray, jbyteArray vArray,
        jint imgWidth, jint imgHeight,
        jint yRowStride, jint uvRowStride, jint uvPixelStride,
        jint targetWidth, jint targetHeight,
        jint rotationDegrees,
        jfloatArray outBuffer) {

    const int tW = targetWidth, tH = targetHeight;
    const int srcW = imgWidth, srcH = imgHeight;
    const int pixelCount = tW * tH;
    const int expected = 3 * pixelCount;

    if (env->GetArrayLength(outBuffer) < expected) {
        LOGE("nativePreprocessYUVBytesInto: dst too small (%d < %d)",
             env->GetArrayLength(outBuffer), expected);
        return JNI_FALSE;
    }

    auto* yData = static_cast<const uint8_t*>(env->GetPrimitiveArrayCritical(yArray, nullptr));
    auto* uData = static_cast<const uint8_t*>(env->GetPrimitiveArrayCritical(uArray, nullptr));
    auto* vData = static_cast<const uint8_t*>(env->GetPrimitiveArrayCritical(vArray, nullptr));
    auto* output = static_cast<float*>(env->GetPrimitiveArrayCritical(outBuffer, nullptr));
    if (!yData || !uData || !vData || !output) {
        if (yData)  env->ReleasePrimitiveArrayCritical(yArray,    (void*)yData,  JNI_ABORT);
        if (uData)  env->ReleasePrimitiveArrayCritical(uArray,    (void*)uData,  JNI_ABORT);
        if (vData)  env->ReleasePrimitiveArrayCritical(vArray,    (void*)vData,  JNI_ABORT);
        if (output) env->ReleasePrimitiveArrayCritical(outBuffer, (void*)output, JNI_ABORT);
        LOGE("nativePreprocessYUVBytesInto: null buffer");
        return JNI_FALSE;
    }

    for (int ty = 0; ty < tH; ++ty) {
        for (int tx = 0; tx < tW; ++tx) {
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

            int srcX = std::max(0, std::min(static_cast<int>(srcXf + 0.5f), srcW - 1));
            int srcY = std::max(0, std::min(static_cast<int>(srcYf + 0.5f), srcH - 1));

            int yIdx  = srcY * yRowStride + srcX;
            int uvIdx = (srcY / 2) * uvRowStride + (srcX / 2) * uvPixelStride;

            int Y = yData[yIdx];
            int U = uData[uvIdx] - 128;
            int V = vData[uvIdx] - 128;

            int r = Y + static_cast<int>(1.370705f * V);
            int g = Y - static_cast<int>(0.337633f * U) - static_cast<int>(0.698001f * V);
            int b = Y + static_cast<int>(1.732446f * U);
            r = std::max(0, std::min(255, r));
            g = std::max(0, std::min(255, g));
            b = std::max(0, std::min(255, b));

            int idx = ty * tW + tx;
            output[idx]                  = r / 255.0f;
            output[pixelCount + idx]     = g / 255.0f;
            output[2 * pixelCount + idx] = b / 255.0f;
        }
    }

    env->ReleasePrimitiveArrayCritical(yArray,    (void*)yData,  JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(uArray,    (void*)uData,  JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(vArray,    (void*)vData,  JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(outBuffer, (void*)output, 0);  // commit
    return JNI_TRUE;
}

JNIEXPORT jfloatArray JNICALL
Java_com_ptcdepth_android_DepthEstimatorQNN_nativeResizeDepth(
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
    const int outSize = dstW * dstH;
    std::vector<float> output(outSize);

    for (int dy = 0; dy < dstH; ++dy) {
        float srcYf = static_cast<float>(dy) * (srcH - 1) / (dstH - 1);
        int sy0 = static_cast<int>(srcYf);
        int sy1 = std::min(sy0 + 1, srcH - 1);
        float fy = srcYf - sy0;

        for (int dx = 0; dx < dstW; ++dx) {
            float srcXf = static_cast<float>(dx) * (srcW - 1) / (dstW - 1);
            int sx0 = static_cast<int>(srcXf);
            int sx1 = std::min(sx0 + 1, srcW - 1);
            float fx = srcXf - sx0;

            float v00 = srcData[sy0 * srcW + sx0];
            float v10 = srcData[sy0 * srcW + sx1];
            float v01 = srcData[sy1 * srcW + sx0];
            float v11 = srcData[sy1 * srcW + sx1];

            output[dy * dstW + dx] = v00 * (1 - fx) * (1 - fy)
                                   + v10 * fx       * (1 - fy)
                                   + v01 * (1 - fx) * fy
                                   + v11 * fx       * fy;
        }
    }

    env->ReleaseFloatArrayElements(depthArray, srcData, JNI_ABORT);
    jfloatArray result = env->NewFloatArray(outSize);
    env->SetFloatArrayRegion(result, 0, outSize, output.data());
    return result;
}

}  // extern "C"
