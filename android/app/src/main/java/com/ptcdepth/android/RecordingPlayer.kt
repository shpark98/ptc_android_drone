package com.ptcdepth.android

import android.os.Handler
import android.os.Looper
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Plays back recorded depth data from disk.
 * Reads frame_NNNNNN.bin + frames.jsonl from a recording session directory.
 */
class RecordingPlayer {

    data class RecordingInfo(
        val dir: File,
        val timestamp: String,       // e.g., "20260220_153045"
        val displayDate: String,     // e.g., "2026-02-20 15:30:45"
        val totalFrames: Int,
        val isRefined: Boolean
    )

    data class FrameData(
        val depth: FloatArray,
        val width: Int,
        val height: Int,
        val intrinsics: CameraIntrinsics?,
        val frameIndex: Int
    )

    var isPlaying = false
        private set
    var currentFrame = 0
        private set
    var totalFrames = 0
        private set

    private var sessionDir: File? = null
    private var frameMetadata: List<JSONObject> = emptyList()
    private val handler = Handler(Looper.getMainLooper())
    private var playRunnable: Runnable? = null
    var onFrame: ((FrameData) -> Unit)? = null
    var onPlaybackFinished: (() -> Unit)? = null
    var onFrameChanged: ((Int, Int) -> Unit)? = null  // (current, total)
    var playbackFps = 10

    /**
     * List all available recordings, sorted newest first.
     */
    fun listRecordings(baseDir: File): List<RecordingInfo> {
        val recordingsDir = File(baseDir, "recordings")
        if (!recordingsDir.exists()) return emptyList()

        val dateFormat = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
        val displayFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)

        return recordingsDir.listFiles()
            ?.filter { it.isDirectory }
            ?.mapNotNull { dir ->
                val metaFile = File(dir, "metadata.json")
                if (!metaFile.exists()) return@mapNotNull null
                try {
                    val meta = JSONObject(metaFile.readText())
                    val frames = meta.optInt("total_frames", 0)
                    if (frames == 0) return@mapNotNull null

                    val timestamp = dir.name
                    val displayDate = try {
                        val date = dateFormat.parse(timestamp)
                        displayFormat.format(date ?: Date())
                    } catch (e: Exception) {
                        timestamp
                    }

                    // Check first frame's jsonl to see if refined
                    val jsonlFile = File(dir, "frames.jsonl")
                    val isRefined = if (jsonlFile.exists()) {
                        val firstLine = jsonlFile.bufferedReader().readLine()
                        firstLine?.let { JSONObject(it).optBoolean("refined", false) } ?: false
                    } else false

                    RecordingInfo(dir, timestamp, displayDate, frames, isRefined)
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to read recording: ${dir.name}", e)
                    null
                }
            }
            ?.sortedByDescending { it.timestamp }
            ?: emptyList()
    }

    /**
     * Load a recording session for playback.
     */
    fun load(recording: RecordingInfo): Boolean {
        stop()
        sessionDir = recording.dir
        totalFrames = recording.totalFrames
        currentFrame = 0

        // Parse frames.jsonl
        val jsonlFile = File(recording.dir, "frames.jsonl")
        frameMetadata = if (jsonlFile.exists()) {
            jsonlFile.bufferedReader().readLines()
                .filter { it.isNotBlank() }
                .map { JSONObject(it) }
        } else {
            emptyList()
        }

        Log.i(TAG, "Loaded recording: ${recording.dir.name}, $totalFrames frames")
        return true
    }

    /**
     * Read a single frame by index.
     */
    fun readFrame(index: Int): FrameData? {
        val dir = sessionDir ?: return null
        if (index < 0 || index >= totalFrames) return null

        val binFile = File(dir, "frame_%06d.bin".format(index))
        if (!binFile.exists()) return null

        // Get dimensions from JSONL metadata
        val meta = frameMetadata.getOrNull(index)
        val width = meta?.optInt("width", 480) ?: 480
        val height = meta?.optInt("height", 640) ?: 640

        // Read binary depth
        val bytes = FileInputStream(binFile).use { it.readBytes() }
        val floatCount = bytes.size / 4
        val depth = FloatArray(floatCount)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(depth)

        // Parse intrinsics if available
        val intrinsics = meta?.optJSONObject("intrinsics")?.let { k ->
            CameraIntrinsics(
                k.getDouble("fx").toFloat(),
                k.getDouble("fy").toFloat(),
                k.getDouble("cx").toFloat(),
                k.getDouble("cy").toFloat(),
                k.getInt("width"),
                k.getInt("height")
            )
        }

        return FrameData(depth, width, height, intrinsics, index)
    }

    fun play() {
        if (isPlaying) return
        if (totalFrames == 0) return
        isPlaying = true
        scheduleNextFrame()
    }

    fun pause() {
        isPlaying = false
        playRunnable?.let { handler.removeCallbacks(it) }
        playRunnable = null
    }

    fun stop() {
        pause()
        currentFrame = 0
        onFrameChanged?.invoke(0, totalFrames)
    }

    fun seekTo(frame: Int) {
        val wasPlaying = isPlaying
        if (wasPlaying) pause()
        currentFrame = frame.coerceIn(0, (totalFrames - 1).coerceAtLeast(0))
        val data = readFrame(currentFrame)
        if (data != null) {
            onFrame?.invoke(data)
        }
        onFrameChanged?.invoke(currentFrame, totalFrames)
        if (wasPlaying) play()
    }

    private fun scheduleNextFrame() {
        if (!isPlaying) return
        val runnable = Runnable {
            if (!isPlaying) return@Runnable
            val data = readFrame(currentFrame)
            if (data != null) {
                onFrame?.invoke(data)
                onFrameChanged?.invoke(currentFrame, totalFrames)
            }
            currentFrame++
            if (currentFrame >= totalFrames) {
                currentFrame = 0
                isPlaying = false
                onPlaybackFinished?.invoke()
                return@Runnable
            }
            scheduleNextFrame()
        }
        playRunnable = runnable
        handler.postDelayed(runnable, 1000L / playbackFps)
    }

    fun release() {
        stop()
        sessionDir = null
        frameMetadata = emptyList()
        totalFrames = 0
        onFrame = null
        onPlaybackFinished = null
        onFrameChanged = null
    }

    /**
     * Delete a recording from disk.
     */
    fun deleteRecording(recording: RecordingInfo): Boolean {
        return try {
            recording.dir.deleteRecursively()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to delete recording: ${recording.dir.name}", e)
            false
        }
    }

    companion object {
        private const val TAG = "RecordingPlayer"
    }
}
