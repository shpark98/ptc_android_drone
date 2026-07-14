package com.ptcdepth.android

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.media.Image
import android.os.Build
import android.util.Log
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import kotlin.math.min

/**
 * Depth Anything V2 estimator running on ONNX Runtime + QNN Execution Provider.
 *
 * This matches Qualcomm AI Hub's published 22.8 ms benchmark setup for the
 * SAME ONNX file on Galaxy S25 (Snapdragon 8 Elite, Hexagon V79). ORT-QNN
 * compiles the ONNX graph into a QNN context binary on first launch (cached
 * at `cache/depth_anything_qnn_ctx.bin`) and reuses the cache on subsequent
 * launches.
 *
 * Public API kept identical to the previous direct-QNN-SDK implementation so
 * [MainActivity] doesn't need to change.
 */
class DepthEstimatorQNN(
    private val context: Context,
    private val onnxAssetName: String = "depth_anything_v2.onnx",
    private val onnxWeightsAsset: String = "depth_anything_v2.data",
) {
    private val tag = "DepthEstimatorQNN"

    val inputWidth = 518
    val inputHeight = 518

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private lateinit var inputName: String
    private var invDepthBuffer: FloatArray? = null
    // Pre-allocated buffers so the hot loop allocates zero LOS objects per frame.
    private val inputBuffer = FloatArray(3 * 518 * 518)   // YUV→NCHW dest, ~3 MB
    private val depthOutputBuffer = FloatArray(518 * 518)  // ORT output, ~1 MB

    init {
        try {
            // Native helper (YUV→RGB→518x518 NCHW preprocessing + bilinear resize).
            System.loadLibrary("qnn_depth_jni")
            Log.i(tag, "Loaded qnn_depth_jni native library (YUV preprocessing)")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(tag, "Failed to load qnn_depth_jni: ${e.message}")
            throw e
        }

        try {
            Log.i(tag, "Initializing ORT-QNN depth estimator")

            // ONNX references its weights file by relative filename, so the
            // two files must end up next to each other on disk.
            val onnxFile = copyAssetToCache(onnxAssetName)
            copyAssetToCache(onnxWeightsAsset)
            val cacheDir = onnxFile.parentFile!!

            Log.i(tag, "  ONNX path: ${onnxFile.absolutePath} (${onnxFile.length() / 1024} KB)")

            ortEnv = OrtEnvironment.getEnvironment()

            val sessionOptions = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                setIntraOpNumThreads(1)

                // QNN EP options — kept explicit so we can diagnose which one
                // QNN_DEVICE_ERROR_INVALID_CONFIG rejects.
                //
                // soc_model / htp_arch must match the actual Hexagon on the
                // device or QNN compiles the ONNX graph for the wrong arch.
                //   soc_model values → QnnTypes.h    (QNN_SOC_MODEL_*)
                //   htp_arch  values → QnnHtpDevice.h (QNN_HTP_DEVICE_ARCH_*)
                // Skel/stub for each arch are bundled in app/libs/arm64-v8a/
                // (V69/V73/V79/V81 shipped from QAIRT 2.42).
                //
                // Using absolute path to libQnnHtp.so guarantees we hit our
                // shipped QAIRT 2.42 lib (not whatever Android linker picks).
                val nativeLibDir = context.applicationInfo.nativeLibraryDir

                val socName = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                    Build.SOC_MODEL
                } else {
                    ""
                }
                val (socModel, htpArch) = when (socName) {
                    "SM8450" -> "36" to "69"  // Snapdragon 8 Gen 1  (S22)  / Hexagon V69
                    "SM8550" -> "43" to "73"  // Snapdragon 8 Gen 2          / Hexagon V73
                    "SM8650" -> "57" to "75"  // Snapdragon 8 Gen 3          / Hexagon V75
                    "SM8750" -> "69" to "79"  // Snapdragon 8 Elite  (S25)   / Hexagon V79
                    "SM8850" -> "87" to "81"  // Snapdragon 8 Elite Gen 5    / Hexagon V81
                    else     -> "87" to "81"  // default to newest bundled skel (V81)
                }
                Log.i(tag, "  Detected SoC: '$socName' → soc_model=$socModel, htp_arch=$htpArch")

                // Cache is keyed by arch so switching devices never reuses a
                // context binary compiled for a different Hexagon version.
                val ctxCachePath = File(cacheDir, "depth_anything_qnn_ctx_v$htpArch.bin").absolutePath
                val qnnOptions = mapOf(
                    "backend_path" to "$nativeLibDir/libQnnHtp.so",
                    "soc_model" to socModel,
                    "htp_arch" to htpArch,
                    // `sustained_high_performance` keeps NPU at a high but
                    // thermally sustainable clock. `burst` is for single-shot
                    // latency benchmarks (matches AI Hub's 22.8 ms claim) but
                    // causes thermal throttling within ~1 minute under continuous
                    // inference.
                    "htp_performance_mode" to "sustained_high_performance",
                    "htp_graph_finalization_optimization_mode" to "3",  // O3 finalize-time opt
                    "qnn_context_cache_enable" to "1",
                    "qnn_context_cache_path" to ctxCachePath,
                    "vtcm_mb" to "8",
                    "rpc_control_latency" to "100",
                )
                Log.i(tag, "  QNN EP options: $qnnOptions")
                addQnn(qnnOptions)
            }

            val t0 = System.currentTimeMillis()
            ortSession = ortEnv!!.createSession(onnxFile.absolutePath, sessionOptions)
            val t1 = System.currentTimeMillis()
            Log.i(tag, "ORT session created in ${t1 - t0} ms " +
                "(first run includes QNN context binary compilation; cached afterward)")

            inputName = ortSession!!.inputNames.first()
            Log.i(tag, "  Input name: $inputName")
            Log.i(tag, "  Output names: ${ortSession!!.outputNames}")
            Log.i(tag, "  Backend: ORT-QNN / Hexagon HTP")
        } catch (e: Exception) {
            Log.e(tag, "Failed to initialize ORT-QNN depth estimator", e)
            throw e
        }
    }

    private fun copyAssetToCache(assetName: String): File {
        val cacheFile = File(context.cacheDir, assetName)
        if (!cacheFile.exists() || cacheFile.length() == 0L) {
            Log.i(tag, "Copying asset $assetName to cache...")
            try {
                context.assets.open(assetName).use { input ->
                    cacheFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
                Log.i(tag, "  $assetName: ${cacheFile.length() / 1024} KB")
            } catch (e: IOException) {
                throw RuntimeException("Failed to copy $assetName from assets", e)
            }
        }
        return cacheFile
    }

    /**
     * One-shot pipeline matching the previous DepthEstimatorQNN.processFrame()
     * signature: YUV preprocess (native) → ORT-QNN infer → normalize + resize.
     *
     * @param output Pre-allocated array sized [dstW * dstH] to receive normalized
     *               inverse depth in [0, 1] (high=close, low=far).
     */
    fun processFrame(
        yData: ByteArray, uData: ByteArray, vData: ByteArray,
        imgW: Int, imgH: Int,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int,
        rotationDegrees: Int,
        output: FloatArray, dstW: Int, dstH: Int,
    ): Boolean {
        val env = ortEnv ?: return false
        val session = ortSession ?: return false

        val t0 = System.currentTimeMillis()

        val ok = nativePreprocessYUVBytesInto(
            yData, uData, vData,
            imgW, imgH, yRowStride, uvRowStride, uvPixelStride,
            inputWidth, inputHeight, rotationDegrees,
            inputBuffer,
        )
        if (!ok) return false

        val t1 = System.currentTimeMillis()

        val shape = longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong())
        val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputBuffer), shape)
        val depthArr = depthOutputBuffer
        inputTensor.use {
            session.run(mapOf(inputName to it)).use { results ->
                // Read the output tensor's raw float buffer directly into a
                // reusable FloatArray. Avoids the Array<Array<Array<FloatArray>>>
                // boxing path which allocates ~1 MB of nested Java arrays.
                val outTensor = results[0] as OnnxTensor
                outTensor.floatBuffer.get(depthArr)
            }
        }

        val t2 = System.currentTimeMillis()

        val normalizedDepth = postprocessDepth(depthArr)
        val resized = if (dstW != inputWidth || dstH != inputHeight) {
            nativeResizeDepth(normalizedDepth, inputWidth, inputHeight, dstW, dstH) ?: normalizedDepth
        } else {
            normalizedDepth
        }
        System.arraycopy(resized, 0, output, 0, min(resized.size, output.size))

        val t3 = System.currentTimeMillis()
        Log.d(tag, "ORT-QNN: prep=${t1 - t0}ms infer=${t2 - t1}ms post=${t3 - t2}ms total=${t3 - t0}ms")

        return true
    }

    private fun postprocessDepth(depth: FloatArray): FloatArray {
        var minDepth = Float.MAX_VALUE
        var maxDepth = -Float.MAX_VALUE
        for (d in depth) {
            if (d < minDepth) minDepth = d
            if (d > maxDepth) maxDepth = d
        }
        val range = maxDepth - minDepth

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
            invDepth[i] = (depth[i] - minDepth) / range
        }
        return invDepth
    }

    fun pausePerf() {
        // ORT-QNN handles HTP power state via htp_performance_mode; no-op kept
        // for API compatibility.
    }

    fun resumePerf() {
        // No-op (see pausePerf).
    }

    fun close() {
        try {
            ortSession?.close()
            ortSession = null
        } catch (e: Exception) {
            Log.e(tag, "Error closing ORT session", e)
        }
        Log.i(tag, "ORT-QNN depth estimator released")
    }

    // JNI: preprocessing helpers only (inference moved to ORT). ----------

    private external fun nativePreprocessYUVBytesInto(
        yData: ByteArray, uData: ByteArray, vData: ByteArray,
        imgWidth: Int, imgHeight: Int,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int,
        targetWidth: Int, targetHeight: Int,
        rotationDegrees: Int,
        outBuffer: FloatArray,
    ): Boolean

    private external fun nativeResizeDepth(
        depth: FloatArray,
        srcW: Int, srcH: Int,
        dstW: Int, dstH: Int,
    ): FloatArray?
}
