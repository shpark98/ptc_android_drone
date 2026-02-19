package com.prdepth.android

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

/**
 * Depth Anything V2 model inference using ONNX Runtime
 */
class DepthEstimator(
    private val context: Context,
    private val modelName: String = "depth_anything.onnx"
) {
    private val tag = "DepthEstimator"
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    private var inputWidth = 518  // Model is trained for 518x518
    private var inputHeight = 518

    // Model normalization parameters (ImageNet stats)
    private val meanRGB = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val stdRGB = floatArrayOf(0.229f, 0.224f, 0.225f)

    init {
        try {
            Log.i(tag, "Loading Depth Anything V2 model: $modelName")

            // Create ONNX Runtime environment
            ortEnv = OrtEnvironment.getEnvironment()

            // Load model from assets
            val modelBytes = context.assets.open(modelName).use { it.readBytes() }

            // Try GPU acceleration first, fallback to CPU if it fails
            var sessionCreated = false
            var lastError: Exception? = null

            // Try 1: GPU acceleration with NNAPI
            try {
                Log.i(tag, "Attempting to load model with NNAPI (GPU)...")
                val sessionOptions = OrtSession.SessionOptions().apply {
                    addNnapi()
                    setIntraOpNumThreads(4)
                    setInterOpNumThreads(4)
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                }
                ortSession = ortEnv?.createSession(modelBytes, sessionOptions)
                sessionCreated = true
                Log.i(tag, "✓ NNAPI (GPU) acceleration enabled successfully")
            } catch (e: Exception) {
                Log.w(tag, "NNAPI failed (this is normal for some models): ${e.message}")
                lastError = e
            }

            // Try 2: CPU-only if GPU failed
            if (!sessionCreated) {
                try {
                    Log.i(tag, "Falling back to CPU-only inference...")
                    val sessionOptions = OrtSession.SessionOptions().apply {
                        setIntraOpNumThreads(4)  // Use 4 CPU threads
                        setInterOpNumThreads(4)
                        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                    }
                    ortSession = ortEnv?.createSession(modelBytes, sessionOptions)
                    sessionCreated = true
                    Log.i(tag, "✓ CPU-only inference initialized (4 threads)")
                } catch (e: Exception) {
                    Log.e(tag, "CPU-only session creation also failed", e)
                    throw e
                }
            }

            Log.i(tag, "Model loaded successfully")
            Log.i(tag, "Input shape: [1, 3, $inputHeight, $inputWidth]")

        } catch (e: Exception) {
            Log.e(tag, "Failed to load model", e)
            throw e
        }
    }

    /**
     * Estimate depth from RGB image
     * @param bitmap Input image (will be resized to model input size)
     * @return Inverse depth map as FloatArray (H*W), values in [0, 1], 0=far, 1=near
     */
    fun estimateDepth(bitmap: Bitmap): FloatArray {
        val session = ortSession ?: throw IllegalStateException("Model not loaded")
        val env = ortEnv ?: throw IllegalStateException("ONNX environment not initialized")

        try {
            Log.d(tag, "Starting inference for bitmap: ${bitmap.width}x${bitmap.height}")

            // Resize input image to model input size
            val resizedBitmap = Bitmap.createScaledBitmap(
                bitmap, inputWidth, inputHeight, true
            )
            Log.d(tag, "Resized to: ${resizedBitmap.width}x${resizedBitmap.height}")

            // Convert bitmap to float array with normalization
            val inputData = preprocessImage(resizedBitmap)
            Log.d(tag, "Preprocessed input data size: ${inputData.size}")

            // Create input tensor: shape [1, 3, H, W]
            val inputShape = longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong())
            val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), inputShape)
            Log.d(tag, "Created input tensor with shape: ${inputShape.contentToString()}")

            // Run inference
            val startTime = System.currentTimeMillis()
            val outputs = session.run(mapOf("l_x_" to inputTensor))  // Model expects "l_x_" not "image"
            val inferenceTime = System.currentTimeMillis() - startTime
            Log.i(tag, "Inference completed in ${inferenceTime}ms")

            // Get output tensor - try different formats
            val outputValue = outputs[0].value
            Log.d(tag, "Output value class: ${outputValue.javaClass.name}")

            val depthArray: FloatArray = when (outputValue) {
                // Case 1: Already a FloatArray [H*W]
                is FloatArray -> {
                    Log.d(tag, "Output is FloatArray of size: ${outputValue.size}")
                    outputValue
                }
                // Case 2: Array structure [1, H, W] or [1, 1, H, W]
                is Array<*> -> {
                    Log.d(tag, "Output is Array, size: ${outputValue.size}")
                    val batch = outputValue[0]
                    Log.d(tag, "Batch class: ${batch?.javaClass?.name}")

                    when (batch) {
                        is FloatArray -> {
                            Log.d(tag, "Batch is FloatArray of size: ${batch.size}")
                            batch
                        }
                        is Array<*> -> {
                            // Try to flatten 2D array [H, W]
                            Log.d(tag, "Batch is Array, size: ${batch.size}, attempting to flatten")

                            // Check if first element is FloatArray (it's a 2D array)
                            val firstRow = batch[0]
                            when (firstRow) {
                                is FloatArray -> {
                                    // [H, W] format - flatten entire 2D array
                                    Log.d(tag, "Batch is 2D array [${batch.size}, ${firstRow.size}], flattening")
                                    val result = FloatArray(inputHeight * inputWidth)
                                    var idx = 0
                                    for (row in batch) {
                                        val rowArray = row as FloatArray
                                        for (value in rowArray) {
                                            result[idx++] = value
                                        }
                                    }
                                    result
                                }
                                is Array<*> -> {
                                    // [1, H, W] format - go one level deeper
                                    val channel = batch[0] as Array<*>
                                    val result = FloatArray(inputHeight * inputWidth)
                                    var idx = 0
                                    for (row in channel) {
                                        val rowArray = row as FloatArray
                                        for (value in rowArray) {
                                            result[idx++] = value
                                        }
                                    }
                                    result
                                }
                                else -> throw IllegalStateException("Unexpected row type: ${firstRow?.javaClass?.name}")
                            }
                        }
                        else -> throw IllegalStateException("Unexpected batch type: ${batch?.javaClass?.name}")
                    }
                }
                else -> throw IllegalStateException("Unexpected output type: ${outputValue.javaClass.name}")
            }

            Log.d(tag, "Output depth array size: ${depthArray.size} (expected: ${inputHeight * inputWidth})")

            // Post-process: normalize to [0, 1] and invert (Depth Anything outputs depth, we need inverse)
            val invDepth = postprocessDepth(depthArray)
            Log.i(tag, "Postprocessing complete, returning inverse depth")

            // Cleanup
            inputTensor.close()
            outputs.close()

            return invDepth

        } catch (e: Exception) {
            Log.e(tag, "Inference failed", e)
            e.printStackTrace()
            // Return dummy data on failure
            Log.w(tag, "Returning dummy depth data (all 0.5f)")
            return FloatArray(inputHeight * inputWidth) { 0.5f }
        }
    }

    /**
     * Preprocess image: RGB normalization
     */
    private fun preprocessImage(bitmap: Bitmap): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixelCount = width * height

        // Get pixels as ARGB int array
        val pixels = IntArray(pixelCount)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // Check if bitmap is valid (not all black/white)
        var sumR = 0L
        var sumG = 0L
        var sumB = 0L
        for (pixel in pixels) {
            sumR += ((pixel shr 16) and 0xFF)
            sumG += ((pixel shr 8) and 0xFF)
            sumB += (pixel and 0xFF)
        }
        val avgR = sumR / pixelCount
        val avgG = sumG / pixelCount
        val avgB = sumB / pixelCount
        Log.d(tag, "Input image average RGB: ($avgR, $avgG, $avgB)")

        // Convert to float array with shape [3, H, W] and normalize
        val floatArray = FloatArray(3 * pixelCount)

        for (i in 0 until pixelCount) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF) / 255f
            val g = ((pixel shr 8) and 0xFF) / 255f
            val b = (pixel and 0xFF) / 255f

            // Normalize using ImageNet stats and pack as [C, H, W]
            floatArray[i] = (r - meanRGB[0]) / stdRGB[0]  // R channel
            floatArray[pixelCount + i] = (g - meanRGB[1]) / stdRGB[1]  // G channel
            floatArray[2 * pixelCount + i] = (b - meanRGB[2]) / stdRGB[2]  // B channel
        }

        return floatArray
    }

    /**
     * Postprocess depth: normalize to [0, 1] and convert to inverse depth
     */
    private fun postprocessDepth(depth: FloatArray): FloatArray {
        // Find min/max for normalization
        var minDepth = Float.MAX_VALUE
        var maxDepth = Float.MIN_VALUE

        for (d in depth) {
            if (d < minDepth) minDepth = d
            if (d > maxDepth) maxDepth = d
        }

        Log.i(tag, "Raw depth range: min=$minDepth, max=$maxDepth")

        val range = maxDepth - minDepth
        Log.d(tag, "Depth range: $range")

        if (range < 1e-6f) {
            // Constant depth, return default
            Log.w(tag, "Depth range too small ($range), returning constant 0.5f")
            return FloatArray(depth.size) { 0.5f }
        }

        // Normalize to [0, 1] and invert (1 = near, 0 = far)
        val invDepth = FloatArray(depth.size)
        for (i in depth.indices) {
            val normalized = (depth[i] - minDepth) / range  // [0, 1], 0=near, 1=far
            invDepth[i] = 1f - normalized  // Invert: [0, 1], 0=far, 1=near
        }

        // Log sample values for debugging
        val sampleIndices = listOf(0, depth.size/4, depth.size/2, depth.size*3/4, depth.size-1)
        val samples = sampleIndices.map { idx ->
            "[$idx]=${String.format("%.3f", invDepth[idx])}"
        }.joinToString(", ")
        Log.d(tag, "Inverse depth samples: $samples")

        return invDepth
    }

    /**
     * Cleanup resources
     */
    fun close() {
        ortSession?.close()
        ortEnv?.close()
        Log.i(tag, "DepthEstimator closed")
    }
}
