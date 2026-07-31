package com.ptcdepth.android

import android.content.Context
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Persistent FLIR trace that survives swapping the phone's only USB port. */
class FlirDiagnosticLog(context: Context) {
    private val lock = Any()
    private val formatter = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)
    private val directory = File(context.getExternalFilesDir(null) ?: context.filesDir, "diagnostics")
    private val file = File(directory, "flir_trace.log")
    private val previousFile = File(directory, "flir_trace.previous.log")

    init {
        write("APP", "diagnostic logger initialized: ${file.absolutePath}")
    }

    fun write(stage: String, detail: String = "") {
        synchronized(lock) {
            try {
                directory.mkdirs()
                rotateIfNeeded()
                val timestamp = formatter.format(Date())
                file.appendText("$timestamp [$stage] $detail\n", Charsets.UTF_8)
            } catch (error: Throwable) {
                Log.w(TAG, "Unable to write FLIR diagnostic log", error)
            }
        }
    }

    private fun rotateIfNeeded() {
        if (!file.exists() || file.length() < MAX_LOG_BYTES) return
        if (previousFile.exists()) previousFile.delete()
        file.renameTo(previousFile)
    }

    companion object {
        private const val TAG = "FlirDiagnosticLog"
        private const val MAX_LOG_BYTES = 1_000_000L
    }
}
