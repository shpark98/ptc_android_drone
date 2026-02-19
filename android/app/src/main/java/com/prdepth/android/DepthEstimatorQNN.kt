package com.prdepth.android

import android.content.Context
import android.graphics.Bitmap
import android.media.Image
import android.util.Log
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import kotlin.math.max
import kotlin.math.min

/**
 * QNN HTP depth estimator using Depth Anything V2 DLC model.
 * Uses Qualcomm QNN SDK v2.42 with Hexagon HTP backend.
 */
class DepthEstimatorQNN(
    private val context: Context,
    private val modelName: String = "depth_anything_v2.dlc"
) {
    private val tag = "DepthEstimatorQNN"
    private var nativeHandle: Long = 0

    val inputWidth = 518
    val inputHeight = 518

    // Pre-allocated buffers to reduce GC pressure
    private var invDepthBuffer: FloatArray? = null

    // Note: ImageNet normalization is already baked into the DLC model
    // DLC expects input in [0, 1] range, RGB, NCHW [1,3,H,W]

    init {
        try {
            System.loadLibrary("qnn_depth_jni")
            Log.i(tag, "Loaded qnn_depth_jni native library")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(tag, "Failed to load qnn_depth_jni: ${e.message}")
            throw e
        }

        try {
            Log.i(tag, "Initializing QNN HTP depth estimator: $modelName")

            val modelFile = copyModelToCache()
            val nativeLibDir = context.applicationInfo.nativeLibraryDir

            Log.i(tag, "  Model path: ${modelFile.absolutePath}")
            Log.i(tag, "  Native lib dir: $nativeLibDir")

            nativeHandle = nativeInit(modelFile.absolutePath, nativeLibDir)
            if (nativeHandle == 0L) {
                throw RuntimeException("QNN native initialization returned null handle")
            }

            val inputSize = nativeGetInputSize(nativeHandle)
            val outputSize = nativeGetOutputSize(nativeHandle)
            Log.i(tag, "QNN HTP depth estimator initialized")
            Log.i(tag, "  Input elements: $inputSize")
            Log.i(tag, "  Output elements: $outputSize")
            Log.i(tag, "  Backend: Hexagon HTP (QNN v2.42)")
        } catch (e: Exception) {
            Log.e(tag, "Failed to initialize QNN depth estimator", e)
            throw e
        }
    }

    private fun copyModelToCache(): File {
        val cacheFile = File(context.cacheDir, modelName)
        if (!cacheFile.exists()) {
            Log.i(tag, "Copying DLC model to cache...")
            try {
                context.assets.open(modelName).use { input ->
                    cacheFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
                Log.i(tag, "Model copied: ${cacheFile.length() / 1024 / 1024}MB")
            } catch (e: IOException) {
                throw RuntimeException("Failed to copy DLC model from assets", e)
            }
        }
        return cacheFile
    }

    /**
     * Fast path: preprocess YUV camera image directly in native code.
     * Does YUV→RGB + rotate 90° CW + resize to 518x518 + NCHW layout in one pass.
     * Returns NCHW float array ready for inference, or null on failure.
     */
    fun preprocessYUV(image: Image, rotationDegrees: Int = 90): FloatArray? {
        try {
            val yPlane = image.planes[0]
            val uPlane = image.planes[1]
            val vPlane = image.planes[2]

            return nativePreprocessYUV(
                yPlane.buffer,
                uPlane.buffer,
                vPlane.buffer,
                image.width,
                image.height,
                yPlane.rowStride,
                uPlane.rowStride,
                uPlane.pixelStride,
                inputWidth,
                inputHeight,
                rotationDegrees
            )
        } catch (e: Exception) {
            Log.e(tag, "YUV preprocessing failed", e)
            return null
        }
    }

    /**
     * Run inference on pre-processed NCHW float data.
     * Returns normalized inverse depth [0,1].
     */
    fun inferDepth(inputData: FloatArray): FloatArray {
        if (nativeHandle == 0L) throw IllegalStateException("QNN not initialized")

        val startTime = System.currentTimeMillis()
        val rawOutput = nativeInfer(nativeHandle, inputData)
            ?: throw RuntimeException("QNN inference returned null")
        val inferenceTime = System.currentTimeMillis() - startTime
        Log.d(tag, "QNN HTP inference: ${inferenceTime}ms")

        return postprocessDepth(rawOutput)
    }

    /**
     * Legacy path: estimate depth from Bitmap (for ONNX fallback).
     */
    fun estimateDepth(bitmap: Bitmap): FloatArray {
        if (nativeHandle == 0L) throw IllegalStateException("QNN not initialized")

        try {
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
            val inputData = preprocessImageNCHW(resizedBitmap)

            val startTime = System.currentTimeMillis()
            val rawOutput = nativeInfer(nativeHandle, inputData)
                ?: throw RuntimeException("QNN inference returned null")
            val inferenceTime = System.currentTimeMillis() - startTime

            Log.d(tag, "QNN HTP inference: ${inferenceTime}ms")
            return postprocessDepth(rawOutput)
        } catch (e: Exception) {
            Log.e(tag, "QNN inference failed", e)
            return FloatArray(inputHeight * inputWidth) { 0.5f }
        }
    }

    private fun preprocessImageNCHW(bitmap: Bitmap): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixelCount = width * height
        val pixels = IntArray(pixelCount)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // NCHW layout: [1, 3, H, W] — all R values, then all G, then all B
        val floatArray = FloatArray(pixelCount * 3)
        for (i in 0 until pixelCount) {
            val pixel = pixels[i]
            floatArray[i] = ((pixel shr 16) and 0xFF) / 255f                // R plane
            floatArray[pixelCount + i] = ((pixel shr 8) and 0xFF) / 255f    // G plane
            floatArray[2 * pixelCount + i] = (pixel and 0xFF) / 255f        // B plane
        }
        return floatArray
    }

    private fun postprocessDepth(depth: FloatArray): FloatArray {
        var minDepth = Float.MAX_VALUE
        var maxDepth = Float.MIN_VALUE
        for (d in depth) {
            minDepth = min(minDepth, d)
            maxDepth = max(maxDepth, d)
        }

        Log.d(tag, "Raw depth range: min=$minDepth, max=$maxDepth")

        val range = maxDepth - minDepth

        // Reuse pre-allocated buffer
        var invDepth = invDepthBuffer
        if (invDepth == null || invDepth.size != depth.size) {
            invDepth = FloatArray(depth.size)
            invDepthBuffer = invDepth
        }

        if (range < 1e-6f) {
            for (i in invDepth.indices) invDepth[i] = 0.5f
            return invDepth
        }

        for (i in depth.indices) {
            val normalized = (depth[i] - minDepth) / range
            invDepth[i] = normalized  // True inverse depth: high=close, low=far
        }
        return invDepth
    }

    /**
     * Resize depth map using native bilinear interpolation.
     */
    fun resizeDepth(depth: FloatArray, srcW: Int, srcH: Int, dstW: Int, dstH: Int): FloatArray {
        return nativeResizeDepth(depth, srcW, srcH, dstW, dstH) ?: depth
    }

    /**
     * Combined pipeline: YUV preprocess + infer + normalize + resize.
     * All processing in a single JNI call — no intermediate Java arrays.
     * Writes normalized inverse depth [0,1] directly to output array.
     */
    fun processFrame(
        yData: ByteArray, uData: ByteArray, vData: ByteArray,
        imgW: Int, imgH: Int,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int,
        rotationDegrees: Int,
        output: FloatArray, dstW: Int, dstH: Int
    ): Boolean {
        if (nativeHandle == 0L) throw IllegalStateException("QNN not initialized")
        val startTime = System.currentTimeMillis()
        val ok = nativeProcessFrame(
            nativeHandle, yData, uData, vData,
            imgW, imgH, yRowStride, uvRowStride, uvPixelStride,
            inputWidth, inputHeight, rotationDegrees,
            output, dstW, dstH
        )
        val elapsed = System.currentTimeMillis() - startTime
        Log.d(tag, "QNN processFrame: ${elapsed}ms")
        return ok
    }

    fun pausePerf() {
        if (nativeHandle != 0L) nativePausePerf(nativeHandle)
    }

    fun resumePerf() {
        if (nativeHandle != 0L) nativeResumePerf(nativeHandle)
    }

    fun close() {
        if (nativeHandle != 0L) {
            nativeDestroy(nativeHandle)
            nativeHandle = 0
            Log.i(tag, "QNN depth estimator released")
        }
    }

    // JNI methods
    private external fun nativeInit(modelPath: String, nativeLibDir: String): Long
    private external fun nativeInfer(handle: Long, input: FloatArray): FloatArray?
    private external fun nativeGetInputSize(handle: Long): Int
    private external fun nativeGetOutputSize(handle: Long): Int
    private external fun nativeDestroy(handle: Long)
    private external fun nativePausePerf(handle: Long)
    private external fun nativeResumePerf(handle: Long)
    private external fun nativeResizeDepth(
        depth: FloatArray,
        srcW: Int, srcH: Int,
        dstW: Int, dstH: Int
    ): FloatArray?
    private external fun nativePreprocessYUV(
        yBuffer: ByteBuffer,
        uBuffer: ByteBuffer,
        vBuffer: ByteBuffer,
        imgWidth: Int,
        imgHeight: Int,
        yRowStride: Int,
        uvRowStride: Int,
        uvPixelStride: Int,
        targetWidth: Int,
        targetHeight: Int,
        rotationDegrees: Int
    ): FloatArray?
    private external fun nativeProcessFrame(
        handle: Long,
        yData: ByteArray, uData: ByteArray, vData: ByteArray,
        imgWidth: Int, imgHeight: Int,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int,
        modelWidth: Int, modelHeight: Int,
        rotationDegrees: Int,
        output: FloatArray, dstW: Int, dstH: Int
    ): Boolean
}
