package com.ptcdepth.android

import android.util.Log
import org.json.JSONObject
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Logs per-frame depth data and camera pose to disk.
 *
 * Output structure:
 *   recordings/<timestamp>/
 *     frame_000000.bin   — raw float32 depth (H*W*4 bytes, little-endian)
 *     frames.jsonl       — one JSON line per frame (timestamp, intrinsics, pose, etc.)
 *     metadata.json      — summary (total frames, dimensions, etc.)
 */
class DataLogger(private val baseDir: File) {

    private var sessionDir: File? = null
    private var jsonlWriter: OutputStreamWriter? = null
    private var frameCount = 0
    @Volatile var isRecording = false
        private set

    fun startRecording(): String {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val dir = File(baseDir, "recordings/$timestamp")
        dir.mkdirs()
        sessionDir = dir
        frameCount = 0
        jsonlWriter = OutputStreamWriter(
            BufferedOutputStream(FileOutputStream(File(dir, "frames.jsonl"))),
            Charsets.UTF_8
        )
        isRecording = true
        Log.i(TAG, "Recording started: ${dir.absolutePath}")
        return dir.absolutePath
    }

    /**
     * Log one frame of depth + pose data.
     * Call from the depth processing thread (not UI thread).
     */
    fun logFrame(
        depth: FloatArray,
        width: Int,
        height: Int,
        intrinsics: CameraIntrinsics?,
        pose: CameraPose?,
        relativePose: RelativePose?,
        isRefined: Boolean
    ) {
        if (!isRecording) return
        val dir = sessionDir ?: return

        val frameIdx = frameCount++

        // Write raw depth binary
        val binFile = File(dir, "frame_%06d.bin".format(frameIdx))
        val buf = ByteBuffer.allocate(depth.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        buf.asFloatBuffer().put(depth)
        FileOutputStream(binFile).use { it.write(buf.array()) }

        // Write JSONL entry
        val json = JSONObject()
        json.put("frame", frameIdx)
        json.put("timestamp", System.currentTimeMillis())
        json.put("width", width)
        json.put("height", height)
        json.put("refined", isRefined)

        if (intrinsics != null) {
            val k = JSONObject()
            k.put("fx", intrinsics.fx.toDouble())
            k.put("fy", intrinsics.fy.toDouble())
            k.put("cx", intrinsics.cx.toDouble())
            k.put("cy", intrinsics.cy.toDouble())
            k.put("width", intrinsics.width)
            k.put("height", intrinsics.height)
            json.put("intrinsics", k)
        }

        if (pose != null) {
            val p = JSONObject()
            p.put("rotation", floatArrayToJsonArray(pose.rotation))
            p.put("translation", floatArrayToJsonArray(pose.translation))
            p.put("quaternion", floatArrayToJsonArray(pose.quaternion))
            p.put("timestamp", pose.timestamp)
            json.put("pose", p)
        }

        if (relativePose != null) {
            val rp = JSONObject()
            rp.put("R", floatArrayToJsonArray(relativePose.R))
            rp.put("t", floatArrayToJsonArray(relativePose.t))
            rp.put("baseline", relativePose.baseline.toDouble())
            json.put("relative_pose", rp)
        }

        jsonlWriter?.apply {
            write(json.toString())
            write("\n")
            flush()
        }
    }

    fun stopRecording(): String? {
        if (!isRecording) return null
        isRecording = false
        val dir = sessionDir ?: return null

        jsonlWriter?.close()
        jsonlWriter = null

        // Write metadata
        val meta = JSONObject()
        meta.put("total_frames", frameCount)
        meta.put("format", "float32_le")
        File(dir, "metadata.json").writeText(meta.toString(2))

        val path = dir.absolutePath
        Log.i(TAG, "Recording stopped: $frameCount frames saved to $path")
        sessionDir = null
        frameCount = 0
        return path
    }

    private fun floatArrayToJsonArray(arr: FloatArray): org.json.JSONArray {
        val ja = org.json.JSONArray()
        for (v in arr) ja.put(v.toDouble())
        return ja
    }

    companion object {
        private const val TAG = "DataLogger"
    }
}
