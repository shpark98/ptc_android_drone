package com.ptcdepth.android

import android.util.Log

/**
 * Manager for the PTC-Depth C++ pipeline (ptc_depth::PTCDepth).
 *
 * Accepts YUV camera planes directly (same format as the QNN bridge). Builds
 * a grayscale image internally using the same rotation+resize mapping as the
 * QNN preprocessor — guaranteeing spatial alignment between camera image and
 * QNN depth output.
 *
 * Hot-loop allocations: zero. Per-frame depth/triangulation results are written
 * into ping-pong `FloatArray` buffers (two of each, alternated each frame) so
 * MainActivity can still keep a reference to the previous frame's depth without
 * having it overwritten. No `new FloatArray` happens in `processFrameSync`.
 */
class DepthRefinementManager(
    private val fx: Float,
    private val fy: Float,
    private val cx: Float,
    private val cy: Float,
    private val width: Int,
    private val height: Int,
    private var useExternalPose: Boolean = false,
) {
    private var pipelinePtr: Long = 0

    // Ping-pong buffers — written by JNI, returned by reference inside DepthResult.
    private val refinedBuffers = Array(2) { FloatArray(width * height) }
    private val triBuffers     = Array(2) { FloatArray(width * height) }
    // Pre-allocated small buffers for R (3x3 row-major) and t (3) so DepthResult
    // construction does not allocate either.
    private val rBuf = FloatArray(9)
    private val tBuf = FloatArray(3)
    // Scratch for scalar/integer outputs from JNI (baseline, numMatches,
    // numValidTri, usedGTPose, rotationAngleDeg). 5 floats packs all of them.
    private val scalarOut = FloatArray(5)
    private var bufIdx = 0

    init {
        System.loadLibrary("ptc_depth_jni")

        pipelinePtr = nativeCreatePipeline(
            fx, fy, cx, cy, width, height, useExternalPose
        )
        if (pipelinePtr == 0L) {
            throw RuntimeException("Failed to create native PTC-Depth pipeline")
        }
        Log.i(TAG, "PTC-Depth pipeline initialized: ${width}x${height}, fx=$fx, fy=$fy " +
            "(ping-pong buffers: 2× ${width * height * 4 / 1024} KB refined + same tri)")
    }

    fun processFrameSync(
        yData: ByteArray, uData: ByteArray, vData: ByteArray,
        imgW: Int, imgH: Int,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int,
        rotDeg: Int,
        invDepth: FloatArray, depthW: Int, depthH: Int,
        baseline: Float,
        gtR: FloatArray? = null,
        gtT: FloatArray? = null,
    ): DepthResult? {
        if (pipelinePtr == 0L) return null
        val refinedBuf = refinedBuffers[bufIdx]
        val triBuf     = triBuffers[bufIdx]
        bufIdx = (bufIdx + 1) and 1

        val ok = try {
            nativeProcessFrameYUV(
                pipelinePtr,
                yData, uData, vData,
                imgW, imgH, yRowStride, uvRowStride, uvPixelStride,
                rotDeg,
                invDepth, depthW, depthH,
                baseline,
                gtR, gtT,
                refinedBuf, triBuf,
                rBuf, tBuf, scalarOut,
            )
        } catch (e: Exception) {
            Log.e(TAG, "processFrame failed", e)
            false
        }
        if (!ok) return null

        return DepthResult(
            refinedDepth = refinedBuf,
            triDepth = triBuf,
            R = rBuf,
            t = tBuf,
            baseline = scalarOut[0],
            numMatches = scalarOut[1].toInt(),
            numValidTri = scalarOut[2].toInt(),
            usedGTPose = scalarOut[3] != 0f,
            rotationAngleDeg = scalarOut[4],
        )
    }

    fun updateConfig(
        ransacIters: Int = 50,
        minFlowPx: Float = 0.1f,
        maxDepth: Float = 80f,
        minBaseline: Float = 0.05f,
        lambdaForget: Float = 0.1f,
        kappaMin: Float = 0.25f,
        tau0Deg: Float = 1.0f,
        outdoor: Boolean = true,
        iterative: Int = 0,
        verbose: Boolean = false,
    ) {
        if (pipelinePtr == 0L) return
        nativeUpdateFullConfig(
            pipelinePtr,
            ransacIters, minFlowPx, maxDepth, minBaseline,
            lambdaForget, kappaMin, tau0Deg,
            outdoor, iterative, verbose
        )
    }

    fun setUseExternalPose(enabled: Boolean) {
        useExternalPose = enabled
    }

    fun prepareFlow(
        yData: ByteArray, uData: ByteArray, vData: ByteArray,
        imgW: Int, imgH: Int,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int,
        rotDeg: Int,
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
        Log.i(TAG, "PTC-Depth pipeline destroyed")
    }

    // Native methods --------------------------------------------------------

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
        gtR: FloatArray?, gtT: FloatArray?,
        // Caller-supplied destination buffers (no per-frame allocation):
        refinedOut: FloatArray,      // size = width * height
        triOut: FloatArray,          // size = width * height
        rOut: FloatArray,            // size = 9
        tOut: FloatArray,            // size = 3
        scalarOut: FloatArray,       // [baseline, numMatches, numValidTri, usedGTPose, rotAngleDeg]
    ): Boolean

    private external fun nativeUpdateFullConfig(
        pipelinePtr: Long,
        ransacIters: Int, minFlowPx: Float, maxDepth: Float, minBaseline: Float,
        lambdaForget: Float, kappaMin: Float, tau0Deg: Float,
        outdoor: Boolean, iterative: Int, verbose: Boolean,
    )

    private external fun nativePrepareFlow(
        pipelinePtr: Long,
        yData: ByteArray, uData: ByteArray, vData: ByteArray,
        imgW: Int, imgH: Int,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int,
        rotDeg: Int
    )

    private external fun nativeReset(pipelinePtr: Long)
    private external fun nativeDestroyPipeline(pipelinePtr: Long)

    companion object {
        private const val TAG = "PTCDepthMgr"

        fun computeRotatedIntrinsics(
            camFx: Float, camFy: Float, camCx: Float, camCy: Float,
            camW: Int, camH: Int,
            rotDeg: Int = 90,
        ): CameraIntrinsics {
            return when (rotDeg) {
                90 -> {
                    val rfx = camFy
                    val rfy = camFx
                    val rcx = camH - 1f - camCy
                    val rcy = camCx
                    Log.d(TAG, "Rotated intrinsics: fx=$rfx fy=$rfy cx=$rcx cy=$rcy ${camH}x$camW")
                    CameraIntrinsics(
                        fx = rfx, fy = rfy, cx = rcx, cy = rcy,
                        width = camH, height = camW
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
