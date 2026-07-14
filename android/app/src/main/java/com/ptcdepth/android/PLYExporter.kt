package com.ptcdepth.android

import android.util.Log
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

/**
 * Saves a depth map + intrinsics + YUV camera image as a colored point cloud
 * in binary little-endian PLY format.
 *
 * Output layout (per vertex, 15 bytes):
 *   float32 x  float32 y  float32 z  uchar r  uchar g  uchar b
 *
 * Coordinate convention: OpenCV camera frame
 *   X right, Y down, Z forward. CloudCompare and most viewers accept this
 *   directly (may need to flip Y for "Y up" if you prefer).
 *
 * Skips pixels with non-finite depth, depth <= minDepth, or depth > maxDepth.
 */
object PLYExporter {
    private const val TAG = "PLYExporter"

    /**
     * @param depth Depth map sized [depthW * depthH] in METERS.
     * @param yData/uData/vData YUV planes from the camera (raw, unrotated).
     * @param imgW/imgH Camera image dimensions (unrotated, typically 640x480).
     * @param yRowStride/uvRowStride/uvPixelStride YUV plane strides.
     * @param rotationDegrees Rotation applied for depth (matches what the
     *     pipeline used — typically 90 for portrait). When non-zero, the YUV
     *     pixel that lands at depth coord (u, v) is sampled with the inverse
     *     rotation so colors line up with depth.
     * @param fx/fy/cx/cy Intrinsics matching the depth map (i.e. AFTER rotation).
     * @param outFile Destination .ply file.
     * @param minDepth/maxDepth Valid range (meters). Pixels outside are skipped.
     */
    fun saveColoredPLY(
        depth: FloatArray, depthW: Int, depthH: Int,
        yData: ByteArray, uData: ByteArray, vData: ByteArray,
        imgW: Int, imgH: Int,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int,
        rotationDegrees: Int,
        fx: Float, fy: Float, cx: Float, cy: Float,
        outFile: File,
        minDepth: Float = 0.1f,
        maxDepth: Float = 80f,
    ): Int {
        require(depth.size == depthW * depthH) {
            "depth size ${depth.size} != ${depthW}x${depthH}=${depthW * depthH}"
        }

        // First pass: count valid points so we can write the PLY header with the
        // exact vertex count (binary PLY needs N up-front).
        var validCount = 0
        for (v in 0 until depthH) {
            for (u in 0 until depthW) {
                val z = depth[v * depthW + u]
                if (z.isFinite() && z > minDepth && z <= maxDepth) validCount++
            }
        }

        Log.i(TAG, "saveColoredPLY: ${depthW}x${depthH} depth -> $validCount valid points")

        outFile.parentFile?.mkdirs()
        BufferedOutputStream(FileOutputStream(outFile)).use { out ->
            writeAsciiHeader(out, validCount)
            writeVerticesBinary(
                out, depth, depthW, depthH,
                yData, uData, vData,
                imgW, imgH, yRowStride, uvRowStride, uvPixelStride,
                rotationDegrees,
                fx, fy, cx, cy,
                minDepth, maxDepth,
            )
        }
        Log.i(TAG, "saveColoredPLY: wrote ${outFile.length()} bytes to ${outFile.absolutePath}")
        return validCount
    }

    private fun writeAsciiHeader(out: OutputStream, vertexCount: Int) {
        val header = """
            ply
            format binary_little_endian 1.0
            comment Captured by PTC-Depth Android
            element vertex $vertexCount
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            end_header
        """.trimIndent().trim() + "\n"
        out.write(header.toByteArray(Charsets.US_ASCII))
    }

    /**
     * For each valid depth pixel (u, v): emit 3D point + RGB color sampled from
     * the YUV image at the corresponding source pixel (accounting for rotation).
     *
     * Rotation mapping (90° CW portrait, camera 640x480 → depth 480x640):
     *   srcX = v
     *   srcY = (imgH - 1) - u
     */
    private fun writeVerticesBinary(
        out: OutputStream,
        depth: FloatArray, depthW: Int, depthH: Int,
        yData: ByteArray, uData: ByteArray, vData: ByteArray,
        imgW: Int, imgH: Int,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int,
        rotationDegrees: Int,
        fx: Float, fy: Float, cx: Float, cy: Float,
        minDepth: Float, maxDepth: Float,
    ) {
        val recordBytes = 15  // 3 float + 3 uchar
        val chunkVertices = 4096
        val buf = ByteBuffer.allocate(chunkVertices * recordBytes).order(ByteOrder.LITTLE_ENDIAN)

        var written = 0
        for (v in 0 until depthH) {
            for (u in 0 until depthW) {
                val z = depth[v * depthW + u]
                if (!z.isFinite() || z <= minDepth || z > maxDepth) continue

                // 3D point in camera frame (OpenCV)
                val X = (u - cx) * z / fx
                val Y = (v - cy) * z / fy
                val Z = z

                // Source pixel in unrotated YUV — invert the rotation depth used
                val srcX: Int
                val srcY: Int
                when (rotationDegrees) {
                    90 -> { srcX = v; srcY = (imgH - 1) - u }
                    270 -> { srcX = (imgW - 1) - v; srcY = u }
                    180 -> { srcX = (imgW - 1) - u; srcY = (imgH - 1) - v }
                    else -> { srcX = u; srcY = v }
                }
                if (srcX < 0 || srcX >= imgW || srcY < 0 || srcY >= imgH) continue

                val yIdx = srcY * yRowStride + srcX
                val uvIdx = (srcY / 2) * uvRowStride + (srcX / 2) * uvPixelStride
                val Yv = (yData[yIdx].toInt() and 0xFF)
                val Uv = (uData[uvIdx].toInt() and 0xFF) - 128
                val Vv = (vData[uvIdx].toInt() and 0xFF) - 128
                val r = clamp255(Yv + (1.370705f * Vv).toInt())
                val g = clamp255(Yv - (0.337633f * Uv).toInt() - (0.698001f * Vv).toInt())
                val b = clamp255(Yv + (1.732446f * Uv).toInt())

                buf.putFloat(X).putFloat(Y).putFloat(Z)
                buf.put(r.toByte()).put(g.toByte()).put(b.toByte())
                written++

                if (written % chunkVertices == 0) {
                    out.write(buf.array(), 0, buf.position())
                    buf.clear()
                }
            }
        }
        if (buf.position() > 0) {
            out.write(buf.array(), 0, buf.position())
        }
    }

    private fun clamp255(x: Int): Int = max(0, min(255, x))
}
