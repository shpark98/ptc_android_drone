package com.ptcdepth.android

import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Small in-app log buffer shared by every screen and subsystem. */
object AppLog {
    private const val MAX_LINES = 400
    private val formatter = SimpleDateFormat("HH:mm:ss.SSS", Locale.US)
    private val lock = Any()
    private val lines = ArrayDeque<String>()
    private val _text = MutableStateFlow("")
    val text: StateFlow<String> = _text

    fun d(tag: String, message: String) = write("D", tag, message).also { Log.d(tag, message) }
    fun i(tag: String, message: String) = write("I", tag, message).also { Log.i(tag, message) }
    fun w(tag: String, message: String) = write("W", tag, message).also { Log.w(tag, message) }
    fun e(tag: String, message: String, error: Throwable? = null) {
        write("E", tag, if (error == null) message else "$message: ${error.message}")
        Log.e(tag, message, error)
    }

    fun clear() = synchronized(lock) {
        lines.clear()
        _text.value = ""
    }

    fun snapshot(): String = synchronized(lock) { lines.joinToString("\n") }

    private fun write(level: String, tag: String, message: String) = synchronized(lock) {
        lines.addLast("${formatter.format(Date())} $level/$tag: $message")
        while (lines.size > MAX_LINES) lines.removeFirst()
        _text.value = lines.joinToString("\n")
    }
}
