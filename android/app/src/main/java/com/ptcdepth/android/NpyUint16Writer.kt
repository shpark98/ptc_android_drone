package com.ptcdepth.android

import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream

/** Legacy writer retained for source compatibility; FLIR capture is disabled. */
object NpyUint16Writer {
    fun write(file: File, values: DoubleArray, width: Int, height: Int) {
        require(values.size == width * height)
        file.parentFile?.mkdirs()
        var header = "{'descr': '<u2', 'fortran_order': False, 'shape': ($height, $width), }"
        val padding = (64 - ((10 + 2 + header.length + 1) % 64)) % 64
        header += " ".repeat(padding) + "\n"
        val bytes = header.toByteArray(Charsets.US_ASCII)
        BufferedOutputStream(FileOutputStream(file)).use { out ->
            out.write(byteArrayOf(0x93.toByte(), 'N'.code.toByte(), 'U'.code.toByte(), 'M'.code.toByte(), 'P'.code.toByte(), 'Y'.code.toByte(), 1, 0))
            out.write(bytes.size and 0xFF)
            out.write((bytes.size ushr 8) and 0xFF)
            out.write(bytes)
            for (value in values) {
                val encoded = if (value.isFinite()) (value * 100.0).toLong().coerceIn(0L, 65535L).toInt() else 0
                out.write(encoded and 0xFF)
                out.write((encoded ushr 8) and 0xFF)
            }
        }
    }
}
