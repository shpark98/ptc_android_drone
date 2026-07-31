package com.ptcdepth.android

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.hardware.usb.UsbConstants
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbDeviceConnection
import android.hardware.usb.UsbInterface
import android.hardware.usb.UsbManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.ByteArrayOutputStream
import android.app.PendingIntent
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.ArrayDeque

/**
 * Boson USB/UVC device discovery. This is deliberately independent of the
 * removed FLIR One Mobile SDK path. Frame streaming will be attached after the
 * connected Boson descriptor exposes its actual UVC format/endpoints.
 */
class BosonUvcController(
    context: Context,
    private val onStatus: (String) -> Unit,
    private val onFrame: (Bitmap) -> Unit = {},
) {
    private val appContext = context.applicationContext
    private val usbManager = appContext.getSystemService(Context.USB_SERVICE) as UsbManager
    private var registered = false
    private var openedConnection: UsbDeviceConnection? = null
    @Volatile private var streaming = false
    private var streamThread: Thread? = null
    private val logLines = ArrayDeque<String>()
    private val timeFormat = SimpleDateFormat("HH:mm:ss.SSS", Locale.US)

    /** Returns a compact diagnostic log suitable for displaying in the app. */
    fun diagnosticLog(): String = synchronized(logLines) {
        if (logLines.isEmpty()) "Boson USB log\n(no events)" else logLines.joinToString("\n")
    }

    private fun log(message: String) {
        val line = "${timeFormat.format(Date())} $message"
        synchronized(logLines) {
            logLines.addLast(line)
            while (logLines.size > 120) logLines.removeFirst()
        }
        android.util.Log.i("BosonUvc", message)
        onStatus(message)
    }

    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action == ACTION_USB_PERMISSION) {
                val device = intent.usbDeviceCompat() ?: return
                val granted = intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false)
                log("permission: ${device.deviceName} granted=$granted")
                if (granted) inspectDevice(device) else log("permission denied by Android")
                return
            }
            log("USB event: ${intent?.action ?: "unknown"}")
            refresh()
        }
    }

    fun start() {
        log("start: USB host=${usbManager.deviceList.isNotEmpty()}")
        if (!registered) {
            val filter = IntentFilter().apply {
                addAction(UsbManager.ACTION_USB_DEVICE_ATTACHED)
                addAction(UsbManager.ACTION_USB_DEVICE_DETACHED)
                addAction(ACTION_USB_PERMISSION)
            }
            appContext.registerReceiver(receiver, filter, Context.RECEIVER_NOT_EXPORTED)
            registered = true
        }
        refresh()
    }

    fun stop() {
        if (registered) {
            appContext.unregisterReceiver(receiver)
            registered = false
            log("stop: receiver unregistered")
        }
        openedConnection?.close()
        openedConnection = null
        streaming = false
        streamThread?.interrupt()
        streamThread = null
    }

    fun release() = stop()

    fun refresh() {
        val allDevices = usbManager.deviceList.values.toList()
        log("scan: ${allDevices.size} USB device(s), host devices=${allDevices.joinToString { it.deviceName }}")
        val devices = allDevices.filter { isUvcDevice(it) }
        if (devices.isEmpty()) {
            if (allDevices.isNotEmpty()) {
                val details = allDevices.joinToString("; ") { device ->
                    "${device.deviceName} VID=${device.vendorId} PID=${device.productId} " +
                        "class=${device.deviceClass} interfaces=${device.interfaceCount}"
                }
                log("result: USB device(s) found, but no UVC interface: $details")
            } else {
                log("result: no USB device (check OTG, power, and Boson video USB port)")
            }
            return
        }
        val target = devices.first()
        if (!usbManager.hasPermission(target)) {
            requestPermission(target)
            log("permission requested: ${target.deviceName}")
            return
        }
        val summary = devices.joinToString(separator = "\n") { device ->
            val interfaces = (0 until device.interfaceCount).joinToString { index ->
                val usbInterface = device.getInterface(index)
                "#${usbInterface.id}:class=${usbInterface.interfaceClass}/sub=${usbInterface.interfaceSubclass}/eps=${usbInterface.endpointCount}"
            }
            "UVC ${device.deviceName} VID=${device.vendorId} PID=${device.productId} interfaces=[$interfaces]"
        }
        log("result: Boson/UVC device detected: $summary")
        inspectDevice(target)
    }

    private fun requestPermission(device: UsbDevice) {
        val intent = PendingIntent.getBroadcast(
            appContext,
            0,
            Intent(ACTION_USB_PERMISSION).setPackage(appContext.packageName),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_MUTABLE,
        )
        usbManager.requestPermission(device, intent)
    }

    private fun inspectDevice(device: UsbDevice) {
        val candidates = (0 until device.interfaceCount)
            .map { device.getInterface(it) }
            .filter { it.interfaceClass == UsbConstants.USB_CLASS_VIDEO }
        if (candidates.isEmpty()) {
            log("open: no video interface found")
            return
        }
        val streamInterface = candidates.firstOrNull { iface ->
            (0 until iface.endpointCount).any {
                iface.getEndpoint(it).type == UsbConstants.USB_ENDPOINT_XFER_ISOC ||
                    iface.getEndpoint(it).type == UsbConstants.USB_ENDPOINT_XFER_BULK
            }
        } ?: candidates.first()
        val controlInterface = candidates.firstOrNull { it.interfaceSubclass == 1 }
        val connection = usbManager.openDevice(device)
        if (connection == null) {
            log("open failed: UsbManager.openDevice returned null")
            return
        }
        openedConnection?.close()
        openedConnection = connection
        val claimed = connection.claimInterface(streamInterface, true)
        val endpoints = (0 until streamInterface.endpointCount).joinToString { index ->
            val endpoint = streamInterface.getEndpoint(index)
            "addr=${endpoint.address} type=${endpoint.type} dir=${endpoint.direction} max=${endpoint.maxPacketSize}"
        }
        log("open: interface=${streamInterface.id} claimed=$claimed endpoints=[$endpoints]")
        if (!claimed) {
            connection.close()
            return
        }
        logUvcDescriptors(connection)
        if (controlInterface != null) {
            if (!configureUvcStream(connection, controlInterface.id, streamInterface.id)) {
                log("stream: UVC Probe/Commit rejected; no frames will be available")
            }
        } else {
            log("stream: UVC control interface not found; attempting raw endpoint")
        }
        val input = (0 until streamInterface.endpointCount)
            .map { streamInterface.getEndpoint(it) }
            .firstOrNull { it.direction == UsbConstants.USB_DIR_IN && it.type == UsbConstants.USB_ENDPOINT_XFER_BULK }
        if (input == null) {
            log("stream: no bulk IN endpoint; isochronous UVC requires native UsbRequest/libuvc path")
            return
        }
        streaming = true
        streamThread = Thread({ readBulkStream(connection, input) }, "BosonUvcStream").also { it.start() }
        log("stream: bulk reader started endpoint=${input.address}")
    }

    private fun logUvcDescriptors(connection: UsbDeviceConnection) {
        val raw = connection.rawDescriptors ?: return
        var offset = 0
        val formats = mutableListOf<String>()
        while (offset + 2 <= raw.size) {
            val length = raw[offset].toInt() and 0xFF
            if (length < 2 || offset + length > raw.size) break
            val type = raw[offset + 1].toInt() and 0xFF
            if (type == 0x24 && length >= 3) {
                val subtype = raw[offset + 2].toInt() and 0xFF
                when (subtype) {
                    0x04 -> formats += "UNCOMPRESSED formatIndex=${u8(raw, offset + 3)}"
                    0x06 -> formats += "MJPEG formatIndex=${u8(raw, offset + 3)}"
                    0x05, 0x07 -> {
                        if (length >= 8) {
                            val width = u16(raw, offset + 5)
                            val height = u16(raw, offset + 7)
                            formats += "FRAME ${width}x${height} subtype=0x${subtype.toString(16)}"
                        }
                    }
                }
            }
            offset += length
        }
        log("UVC formats: ${if (formats.isEmpty()) "no class-specific format descriptors" else formats.joinToString(" | ")}")
    }

    private fun u8(data: ByteArray, index: Int): Int = data[index].toInt() and 0xFF

    private fun u16(data: ByteArray, index: Int): Int =
        (data[index].toInt() and 0xFF) or ((data[index + 1].toInt() and 0xFF) shl 8)

    /** Start the BOSON 640x512 MJPEG/9Hz UVC stream using standard controls. */
    private fun configureUvcStream(
        connection: UsbDeviceConnection,
        controlInterfaceId: Int,
        streamInterfaceId: Int,
    ): Boolean {
        val probe = ByteArray(26)
        probe[2] = 2 // format index: BOSON MJPEG descriptor
        probe[3] = 1 // frame index: first frame descriptor
        putLe32(probe, 4, 111_111) // 9 Hz, 100 ns units
        putLe32(probe, 18, 640 * 512 * 2)
        putLe32(probe, 22, 16 * 1024) // conservative payload request
        val setProbe = connection.controlTransfer(0x21, 0x01, 0x0100, streamInterfaceId, probe, probe.size, 1000)
        if (setProbe < 0) {
            log("stream: SET_CUR(PROBE) failed=$setProbe")
            return false
        }
        val negotiated = ByteArray(26)
        val getProbe = connection.controlTransfer(0xA1, 0x81, 0x0100, streamInterfaceId, negotiated, negotiated.size, 1000)
        if (getProbe < 0) {
            log("stream: GET_CUR(PROBE) failed=$getProbe")
            return false
        }
        val setCommit = connection.controlTransfer(0x21, 0x01, 0x0200, streamInterfaceId, negotiated, negotiated.size, 1000)
        if (setCommit < 0) {
            log("stream: SET_CUR(COMMIT) failed=$setCommit")
            return false
        }
        log("stream: UVC Probe/Commit accepted control=$controlInterfaceId stream=$streamInterfaceId frame=${negotiated.size}B")
        return true
    }

    private fun putLe32(buffer: ByteArray, offset: Int, value: Int) {
        buffer[offset] = (value and 0xFF).toByte()
        buffer[offset + 1] = ((value ushr 8) and 0xFF).toByte()
        buffer[offset + 2] = ((value ushr 16) and 0xFF).toByte()
        buffer[offset + 3] = ((value ushr 24) and 0xFF).toByte()
    }

    private fun readBulkStream(connection: UsbDeviceConnection, endpoint: android.hardware.usb.UsbEndpoint) {
        // Keep one USB packet per read. Reading a large multiple here can
        // concatenate several UVC payload headers and corrupt the grayscale
        // image during frame assembly.
        val transfer = ByteArray(endpoint.maxPacketSize.coerceAtLeast(64))
        val frame = ByteArrayOutputStream(640 * 512 * 2)
        var lastFid = -1
        var reads = 0L
        var bytesRead = 0L
        var timeouts = 0L
        var emitted = 0L
        var lastReport = System.currentTimeMillis()
        try {
            while (streaming && !Thread.currentThread().isInterrupted) {
                val count = connection.bulkTransfer(endpoint, transfer, transfer.size, 1000)
                reads++
                if (count <= 0) {
                    timeouts++
                    if (System.currentTimeMillis() - lastReport > 2000) {
                        log("stream stats: reads=$reads bytes=$bytesRead timeouts=$timeouts frames=$emitted")
                        lastReport = System.currentTimeMillis()
                    }
                    continue
                }
                bytesRead += count
                if (count <= 2) continue
                var offset = 0
                while (offset + 2 <= count) {
                    val headerLength = transfer[offset].toInt() and 0xFF
                    val flags = transfer[offset + 1].toInt() and 0xFF
                    if (headerLength < 2 || offset + headerLength > count) {
                        // Some Boson firmware revisions expose a raw Y16 bulk
                        // stream without a UVC payload header.
                        frame.write(transfer, 0, count)
                        if (frame.size() >= 640 * 512 * 2) {
                            emitFrame(frame.toByteArray())
                            emitted++
                            frame.reset()
                        }
                        break
                    }
                    val fid = flags and 1
                    if (lastFid >= 0 && fid != lastFid && frame.size() > 0) {
                        if (emitFrame(frame.toByteArray())) {
                            emitted++
                            frame.reset()
                        }
                    }
                    lastFid = fid
                    val payloadStart = offset + headerLength
                    val payloadLength = count - payloadStart
                    if (payloadLength > 0) frame.write(transfer, payloadStart, payloadLength)
                    if ((flags and 2) != 0 && frame.size() > 0) {
                        if (emitFrame(frame.toByteArray())) {
                            emitted++
                            frame.reset()
                        }
                    }
                    offset = count
                }
                if (System.currentTimeMillis() - lastReport > 2000) {
                    log("stream stats: reads=$reads bytes=$bytesRead timeouts=$timeouts frames=$emitted")
                    lastReport = System.currentTimeMillis()
                }
            }
        } catch (t: Throwable) {
            if (streaming) log("stream read failed: ${t.message}")
        }
    }

    private fun emitFrame(bytes: ByteArray): Boolean {
        val jpeg = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        if (jpeg != null) {
            onFrame(jpeg)
            return true
        }
        // Some UVC payloads contain padding or metadata around the JPEG.
        // Extract the JPEG marker range before giving up on decoding.
        var jpegStart = -1
        var jpegEnd = -1
        for (i in 0 until bytes.size - 1) {
            val a = bytes[i].toInt() and 0xFF
            val b = bytes[i + 1].toInt() and 0xFF
            if (jpegStart < 0 && a == 0xFF && b == 0xD8) jpegStart = i
            if (jpegStart >= 0 && a == 0xFF && b == 0xD9) {
                jpegEnd = i + 2
                break
            }
        }
        if (jpegStart >= 0 && jpegEnd > jpegStart) {
            val extracted = BitmapFactory.decodeByteArray(bytes, jpegStart, jpegEnd - jpegStart)
            if (extracted != null) {
                onFrame(extracted)
                return true
            }
        }
        // Boson Y16 fallback: render the high byte as an 8-bit grayscale preview.
        val pixelCount = 640 * 512
        if (bytes.size >= pixelCount) {
            val pixels = IntArray(pixelCount)
            val twoByte = bytes.size >= pixelCount * 2
            for (i in pixels.indices) {
                val gray = if (twoByte) {
                    bytes[i * 2 + 1].toInt() and 0xFF
                } else {
                    bytes[i].toInt() and 0xFF
                }
                pixels[i] = (0xFF shl 24) or (gray shl 16) or (gray shl 8) or gray
            }
            val bitmap = Bitmap.createBitmap(640, 512, Bitmap.Config.ARGB_8888)
            bitmap.setPixels(pixels, 0, 640, 0, 0, 640, 512)
            onFrame(bitmap)
            return true
        }
        return false
    }

    private fun isUvcDevice(device: UsbDevice): Boolean {
        if (device.deviceClass == UsbConstants.USB_CLASS_VIDEO) return true
        for (index in 0 until device.interfaceCount) {
            if (device.getInterface(index).interfaceClass == UsbConstants.USB_CLASS_VIDEO) return true
        }
        return false
    }

    private fun Intent.usbDeviceCompat(): UsbDevice? =
        if (android.os.Build.VERSION.SDK_INT >= 33) {
            getParcelableExtra(UsbManager.EXTRA_DEVICE, UsbDevice::class.java)
        } else {
            @Suppress("DEPRECATION") getParcelableExtra(UsbManager.EXTRA_DEVICE)
        }

    companion object {
        private const val ACTION_USB_PERMISSION = "com.ptcdepth.android.BOSON_USB_PERMISSION"
    }
}
