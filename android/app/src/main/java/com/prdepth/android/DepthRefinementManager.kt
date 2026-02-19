package com.prdepth.android

import android.util.Log

/**
 * Manager for PR-Depth C++ pipeline.
 *
 * Accepts YUV camera planes directly (same format as QNN bridge).
 * Builds BGR image internally using the same rotation+resize mapping
 * as QNN preprocessor for perfect spatial alignment.
 *
 * Intrinsics should be provided in MODEL space (518x518 rotated).
 */
class DepthRefinementManager(
    private val fx: Float,
    private val fy: Float,
    private val cx: Float,
    private val cy: Float,
    private val width: Int,
    private val height: Int,
    private var useGTPose: Boolean = false
) {
    private var pipelinePtr: Long = 0

    init {
        System.loadLibrary("pr_depth_jni")

        pipelinePtr = nativeCreatePipeline(
            fx, fy, cx, cy, width, height, useGTPose
        )
        if (pipelinePtr == 0L) {
            throw RuntimeException("Failed to create native PR-Depth pipeline")
        }
        Log.i(TAG, "PR-Depth pipeline initialized: ${width}x${height}, fx=$fx, fy=$fy")
    }

    /**
     * Process a frame synchronously (call from background thread).
     *
     * @param yData Y plane byte array
     * @param uData U plane byte array
     * @param vData V plane byte array
     * @param imgW Camera image width (e.g., 640)
     * @param imgH Camera image height (e.g., 480)
     * @param yRowStride Y plane row stride
     * @param uvRowStride UV plane row stride
     * @param uvPixelStride UV pixel stride
     * @param rotDeg Rotation degrees (same as QNN: 0, 90, 180, 270)
     * @param invDepth QNN inverse depth at display resolution
     * @param depthW Display depth width
     * @param depthH Display depth height
     * @param baseline Baseline between frames (meters)
     * @param gtR Optional GT rotation (9 floats, row-major)
     * @param gtT Optional GT translation (3 floats)
     */
    fun processFrameSync(
        yData: ByteArray, uData: ByteArray, vData: ByteArray,
        imgW: Int, imgH: Int,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int,
        rotDeg: Int,
        invDepth: FloatArray, depthW: Int, depthH: Int,
        baseline: Float,
        gtR: FloatArray? = null,
        gtT: FloatArray? = null
    ): DepthResult? {
        if (pipelinePtr == 0L) return null
        return try {
            nativeProcessFrameYUV(
                pipelinePtr,
                yData, uData, vData,
                imgW, imgH, yRowStride, uvRowStride, uvPixelStride,
                rotDeg,
                invDepth, depthW, depthH,
                baseline,
                gtR, gtT
            )
        } catch (e: Exception) {
            Log.e(TAG, "processFrame failed", e)
            null
        }
    }

    /**
     * Update full pipeline configuration.
     */
    fun updateConfig(
        ransacIters: Int = 50,
        minFlowPx: Float = 0.01f,
        maxDepth: Float = 80f,
        minBaseline: Float = 0.05f,
        fusionLambdaForget: Float = 0.35f,
        fusionChi2Soft: Float = 6.635f,
        fusionVarFloor: Float = 0.003f,
        useSegmentation: Boolean = true,
        enableIterativeRefinement: Boolean = false,
        fbConsistency: Boolean = false,
        useGTPose: Boolean = false,
        timing: Boolean = true,
        skyMaskInvThresh: Float = 1e-7f
    ) {
        if (pipelinePtr == 0L) return
        nativeUpdateFullConfig(
            pipelinePtr,
            ransacIters, minFlowPx, maxDepth, minBaseline,
            fusionLambdaForget, fusionChi2Soft, fusionVarFloor,
            useSegmentation, enableIterativeRefinement, fbConsistency,
            useGTPose, timing, skyMaskInvThresh
        )
    }

    fun setUseGTPose(enabled: Boolean) {
        useGTPose = enabled
        if (pipelinePtr != 0L) nativeUpdateConfig(pipelinePtr, enabled, false)
    }

    /**
     * Phase 1 of pipelined execution: BGR conversion + optical flow.
     * Call this while QNN inference is running on NPU.
     * The computed flow will be consumed by the next processFrameSync() call.
     */
    fun prepareFlow(
        yData: ByteArray, uData: ByteArray, vData: ByteArray,
        imgW: Int, imgH: Int,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int,
        rotDeg: Int
    ) {
        if (pipelinePtr == 0L) return
        try {
            nativePrepareFlow(
                pipelinePtr,
                yData, uData, vData,
                imgW, imgH, yRowStride, uvRowStride, uvPixelStride,
                rotDeg
            )
        } catch (e: Exception) {
            Log.e(TAG, "prepareFlow failed", e)
        }
    }

    fun reset() {
        if (pipelinePtr != 0L) nativeReset(pipelinePtr)
    }

    fun destroy() {
        if (pipelinePtr != 0L) {
            nativeDestroyPipeline(pipelinePtr)
            pipelinePtr = 0
        }
        Log.i(TAG, "PR-Depth pipeline destroyed")
    }

    // Native methods
    private external fun nativeCreatePipeline(
        fx: Float, fy: Float, cx: Float, cy: Float,
        width: Int, height: Int, useGTPose: Boolean
    ): Long

    private external fun nativeProcessFrameYUV(
        pipelinePtr: Long,
        yData: ByteArray, uData: ByteArray, vData: ByteArray,
        imgW: Int, imgH: Int,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int,
        rotDeg: Int,
        invDepth: FloatArray, depthW: Int, depthH: Int,
        baseline: Float,
        gtR: FloatArray?, gtT: FloatArray?
    ): DepthResult?

    private external fun nativeUpdateFullConfig(
        pipelinePtr: Long,
        ransacIters: Int, minFlowPx: Float, maxDepth: Float, minBaseline: Float,
        fusionLambdaForget: Float, fusionChi2Soft: Float, fusionVarFloor: Float,
        useSegmentation: Boolean, enableIterativeRefinement: Boolean,
        fbConsistency: Boolean, useGTPose: Boolean, timing: Boolean,
        skyMaskInvThresh: Float
    )

    private external fun nativePrepareFlow(
        pipelinePtr: Long,
        yData: ByteArray, uData: ByteArray, vData: ByteArray,
        imgW: Int, imgH: Int,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int,
        rotDeg: Int
    )

    private external fun nativeUpdateConfig(pipelinePtr: Long, useGTPose: Boolean, useGTR: Boolean)
    private external fun nativeReset(pipelinePtr: Long)
    private external fun nativeDestroyPipeline(pipelinePtr: Long)

    companion object {
        private const val TAG = "DepthRefineManager"

        /**
         * Compute rotated camera intrinsics.
         * For 90° CW rotation: 640x480 landscape → 480x640 portrait.
         * No resizing — uses the camera's native resolution (just rotated).
         * PR-Depth should operate at this resolution, NOT at 518x518 model size.
         */
        fun computeRotatedIntrinsics(
            camFx: Float, camFy: Float, camCx: Float, camCy: Float,
            camW: Int, camH: Int,
            rotDeg: Int = 90
        ): CameraIntrinsics {
            return when (rotDeg) {
                90 -> {
                    // 90° CW: new_x = old_y_flipped, new_y = old_x
                    // New image: width=camH, height=camW (480x640)
                    val rfx = camFy                     // horizontal → was vertical
                    val rfy = camFx                     // vertical → was horizontal
                    val rcx = camH - 1f - camCy         // horizontal center (flipped Y)
                    val rcy = camCx                     // vertical center (was X)
                    Log.d(TAG, "Rotated intrinsics: fx=$rfx fy=$rfy cx=$rcx cy=$rcy ${camH}x$camW")
                    CameraIntrinsics(
                        fx = rfx, fy = rfy, cx = rcx, cy = rcy,
                        width = camH, height = camW     // 480x640
                    )
                }
                else -> CameraIntrinsics(
                    fx = camFx, fy = camFy, cx = camCx, cy = camCy,
                    width = camW, height = camH
                )
            }
        }
    }
}
