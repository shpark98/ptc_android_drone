package com.ptcdepth.android

import android.util.Log
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Fast raw capture writer: dumps depth + YUV planes + intrinsics as a single
 * binary blob (.ptcf). No projection, no YUV→RGB conversion happens on device.
 *
 * File layout (all little-endian):
 *   [80 B header]
 *     magic     u32  "PTCF" (0x46435450)
 *     version   u32  1
 *     depthW    u32
 *     depthH    u32
 *     imgW      u32   (unrotated camera width)
 *     imgH      u32   (unrotated camera height)
 *     rotDeg    i32
 *     fx, fy, cx, cy   4 × f32   (intrinsics matching depth, post-rotation)
 *     yRowStride       u32
 *     uvRowStride      u32
 *     uvPixelStride    u32
 *     yLen, uLen, vLen 3 × u32
 *     timestampMs      u64
 *     padding to 80B
 *   [depth: depthW*depthH × f32   meters]
 *   [Y plane: yLen B]
 *   [U plane: uLen B]
 *   [V plane: vLen B]
 *
 * Conversion to PLY happens offline — see tools/viz/ptcf_to_ply.py.
 */
object RawCaptureWriter {
    private const val TAG = "RawCaptureWriter"
    private const val MAGIC = 0x46435450  // 'PTCF' little-endian
    private const val VERSION = 1
    private const val HEADER_BYTES = 80

    fun save(
        depth: FloatArray, depthW: Int, depthH: Int,
        yData: ByteArray, uData: ByteArray, vData: ByteArray,
        imgW: Int, imgH: Int,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int,
        rotationDegrees: Int,
        fx: Float, fy: Float, cx: Float, cy: Float,
        outFile: File,
        timestampMs: Long = System.currentTimeMillis(),
    ): Long {
        require(depth.size == depthW * depthH) {
            "depth size ${depth.size} != ${depthW}x${depthH}=${depthW * depthH}"
        }
        outFile.parentFile?.mkdirs()

        val header = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        header.putInt(MAGIC)
        header.putInt(VERSION)
        header.putInt(depthW)
        header.putInt(depthH)
        header.putInt(imgW)
        header.putInt(imgH)
        header.putInt(rotationDegrees)
        header.putFloat(fx); header.putFloat(fy)
        header.putFloat(cx); header.putFloat(cy)
        header.putInt(yRowStride)
        header.putInt(uvRowStride)
        header.putInt(uvPixelStride)
        header.putInt(yData.size); header.putInt(uData.size); header.putInt(vData.size)
        header.putLong(timestampMs)
        // remaining bytes are zero-padded

        val depthBytes = ByteBuffer.allocate(depth.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        depthBytes.asFloatBuffer().put(depth)

        BufferedOutputStream(FileOutputStream(outFile), 64 * 1024).use { out ->
            out.write(header.array(), 0, HEADER_BYTES)
            out.write(depthBytes.array())
            out.write(yData)
            out.write(uData)
            out.write(vData)
        }
        val totalBytes = outFile.length()
        Log.i(TAG, "saved $totalBytes B → ${outFile.name}")
        return totalBytes
    }
}
