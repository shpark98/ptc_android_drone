package com.ptcdepth.android

import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.hardware.usb.UsbConstants
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbDeviceConnection
import android.hardware.usb.UsbEndpoint
import android.hardware.usb.UsbInterface
import android.hardware.usb.UsbManager
import android.util.Log
import androidx.core.content.ContextCompat
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs

/**
 * Receive-only Android USB host implementation of the gs_usb protocol used by
 * the CAN-to-USB setup in ugv_sdk. It configures channel 0 for classic CAN at
 * 500 kbit/s and emits raw CAN frames; it never transmits a CAN frame.
 */
class GsUsbCanReceiver(
    context: Context,
    private val listener: Listener,
    private val bitrate: Int = AGILEX_CAN_BITRATE,
) {
    interface Listener {
        fun onStatus(status: String)
        fun onFrame(frame: CanFrame)
    }

    private val appContext = context.applicationContext
    private val usbManager = appContext.getSystemService(Context.USB_SERVICE) as UsbManager
    private val executor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "AgilexGsUsbRx")
    }
    private val running = AtomicBoolean(false)
    private var receiverRegistered = false
    @Volatile private var connection: UsbDeviceConnection? = null
    @Volatile private var connectedDeviceId: Int? = null

    private val permissionIntent by lazy {
        PendingIntent.getBroadcast(
            appContext,
            0,
            Intent(ACTION_USB_PERMISSION).setPackage(appContext.packageName),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_MUTABLE,
        )
    }

    private val usbReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            when (intent.action) {
                ACTION_USB_PERMISSION -> {
                    val device = intent.usbDevice() ?: return
                    if (intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false)) {
                        connect(device)
                    } else {
                        listener.onStatus("USB permission denied")
                    }
                }
                UsbManager.ACTION_USB_DEVICE_ATTACHED -> scanAndConnect()
                UsbManager.ACTION_USB_DEVICE_DETACHED -> {
                    val detached = intent.usbDevice()
                    if (detached != null && connectedDeviceId == detached.deviceId) {
                        disconnect("USB-CAN detached")
                    }
                }
            }
        }
    }

    fun start() {
        if (!receiverRegistered) {
            val filter = IntentFilter(ACTION_USB_PERMISSION).apply {
                addAction(UsbManager.ACTION_USB_DEVICE_ATTACHED)
                addAction(UsbManager.ACTION_USB_DEVICE_DETACHED)
            }
            ContextCompat.registerReceiver(
                appContext,
                usbReceiver,
                filter,
                ContextCompat.RECEIVER_NOT_EXPORTED,
            )
            receiverRegistered = true
        }
        scanAndConnect()
    }

    fun stop() {
        disconnect("USB-CAN stopped")
        if (receiverRegistered) {
            appContext.unregisterReceiver(usbReceiver)
            receiverRegistered = false
        }
        executor.shutdownNow()
    }

    private fun scanAndConnect() {
        if (running.get()) return
        val compatible = usbManager.deviceList.values.firstOrNull(::isSupportedGsUsbDevice)
        if (compatible == null) {
            val usbCount = usbManager.deviceList.size
            listener.onStatus(
                if (usbCount == 0) "No gs_usb-style USB-CAN adapter found"
                else "No gs_usb-style USB-CAN adapter found ($usbCount other USB device(s))"
            )
            return
        }
        if (usbManager.hasPermission(compatible)) {
            connect(compatible)
        } else {
            listener.onStatus("USB-CAN permission required")
            usbManager.requestPermission(compatible, permissionIntent)
        }
    }

    private fun connect(device: UsbDevice) {
        if (running.get()) return
        executor.execute {
            var claimedInterface: UsbInterface? = null
            var opened: UsbDeviceConnection? = null
            try {
                val usbInterface = findGsUsbInterface(device)
                    ?: error("bulk IN/OUT interface not found")
                val bulkIn = findBulkInEndpoint(usbInterface)
                    ?: error("bulk IN endpoint not found")

                opened = usbManager.openDevice(device) ?: error("openDevice returned null")
                if (!opened.claimInterface(usbInterface, true)) error("claimInterface failed")
                claimedInterface = usbInterface
                configure(opened, usbInterface.id)

                connection = opened
                connectedDeviceId = device.deviceId
                running.set(true)
                listener.onStatus(
                    "USB-CAN connected %04X:%04X @ %dk".format(
                        device.vendorId,
                        device.productId,
                        bitrate / 1000,
                    )
                )
                receiveLoop(opened, bulkIn)
            } catch (t: Throwable) {
                Log.e(TAG, "gs_usb connection failed", t)
                listener.onStatus("USB-CAN error: ${t.message ?: t.javaClass.simpleName}")
            } finally {
                running.set(false)
                connection = null
                connectedDeviceId = null
                if (claimedInterface != null) opened?.releaseInterface(claimedInterface)
                opened?.close()
            }
        }
    }

    private fun configure(connection: UsbDeviceConnection, interfaceId: Int) {
        controlOut(connection, BREQ_HOST_FORMAT, 1, interfaceId, leInts(0x0000beef))

        val deviceConfig = ByteArray(12)
        controlIn(connection, BREQ_DEVICE_CONFIG, 1, interfaceId, deviceConfig)
        val channelCount = (deviceConfig[3].toInt() and 0xff) + 1
        require(channelCount > 0) { "adapter reports no CAN channel" }

        // Always put channel 0 into reset before changing its timing.
        controlOut(connection, BREQ_MODE, CHANNEL, 0, leInts(MODE_RESET, 0))

        val constants = ByteArray(40)
        controlIn(connection, BREQ_BT_CONST, CHANNEL, 0, constants)
        val timing = calculateBitTiming(constants, bitrate)
        controlOut(
            connection,
            BREQ_BITTIMING,
            CHANNEL,
            0,
            leInts(timing.propSeg, timing.phaseSeg1, timing.phaseSeg2, timing.sjw, timing.brp),
        )
        controlOut(connection, BREQ_MODE, CHANNEL, 0, leInts(MODE_START, MODE_NORMAL_FLAGS))
        Log.i(TAG, "gs_usb timing: $timing")
    }

    private fun receiveLoop(connection: UsbDeviceConnection, endpoint: UsbEndpoint) {
        val buffer = ByteArray(256)
        while (running.get()) {
            val length = connection.bulkTransfer(endpoint, buffer, buffer.size, USB_READ_TIMEOUT_MS)
            if (length < 0) continue
            if (length == 0) continue

            var offset = 0
            while (offset + GS_USB_HEADER_SIZE <= length) {
                val echoId = buffer.int32Le(offset)
                val rawCanId = buffer.int32Le(offset + 4)
                val dlc = (buffer[offset + 8].toInt() and 0xff).coerceAtMost(8)
                val frameSize = GS_USB_HEADER_SIZE + dlc
                if (offset + frameSize > length) break

                if (echoId == GS_USB_RX_ECHO_ID && rawCanId and CAN_ERROR_FLAG == 0) {
                    val canId = rawCanId and CAN_EFF_MASK
                    listener.onFrame(
                        CanFrame(canId, buffer.copyOfRange(offset + GS_USB_HEADER_SIZE, offset + frameSize))
                    )
                }

                // Classic gs_usb frames normally occupy 20 bytes even when DLC is shorter.
                offset += if (length - offset >= GS_USB_CLASSIC_FRAME_SIZE) {
                    GS_USB_CLASSIC_FRAME_SIZE
                } else {
                    frameSize
                }
            }
        }
    }

    private fun disconnect(status: String) {
        running.set(false)
        connection?.close() // Unblocks bulkTransfer.
        connection = null
        listener.onStatus(status)
    }

    private fun controlOut(
        connection: UsbDeviceConnection,
        request: Int,
        value: Int,
        index: Int,
        data: ByteArray,
    ) {
        val result = connection.controlTransfer(
            USB_VENDOR_INTERFACE_OUT,
            request,
            value,
            index,
            data,
            data.size,
            USB_CONTROL_TIMEOUT_MS,
        )
        check(result == data.size) { "USB request $request wrote $result/${data.size} bytes" }
    }

    private fun controlIn(
        connection: UsbDeviceConnection,
        request: Int,
        value: Int,
        index: Int,
        data: ByteArray,
    ) {
        val result = connection.controlTransfer(
            USB_VENDOR_INTERFACE_IN,
            request,
            value,
            index,
            data,
            data.size,
            USB_CONTROL_TIMEOUT_MS,
        )
        check(result == data.size) { "USB request $request read $result/${data.size} bytes" }
    }

    private fun calculateBitTiming(raw: ByteArray, requestedBitrate: Int): CanBitTiming {
        val clock = raw.int32Le(4).toLong() and 0xffffffffL
        val tseg1Min = raw.int32Le(8)
        val tseg1Max = raw.int32Le(12)
        val tseg2Min = raw.int32Le(16)
        val tseg2Max = raw.int32Le(20)
        val sjwMax = raw.int32Le(24)
        val brpMin = raw.int32Le(28)
        val brpMax = raw.int32Le(32)
        val brpInc = raw.int32Le(36).coerceAtLeast(1)
        require(clock > 0 && tseg1Min > 0 && tseg2Min > 0 && brpMin > 0) {
            "invalid bit-timing constants"
        }

        var best: CanBitTiming? = null
        var bestBitrateError = Long.MAX_VALUE
        var bestSampleError = Double.MAX_VALUE
        var brp = brpMin
        while (brp <= brpMax) {
            for (tseg1 in tseg1Min..tseg1Max) {
                for (tseg2 in tseg2Min..tseg2Max) {
                    val quanta = 1L + tseg1 + tseg2
                    val actual = clock / (brp.toLong() * quanta)
                    val bitrateError = abs(actual - requestedBitrate.toLong())
                    val samplePoint = (1.0 + tseg1) / quanta
                    val sampleError = abs(samplePoint - TARGET_SAMPLE_POINT)
                    if (bitrateError < bestBitrateError ||
                        (bitrateError == bestBitrateError && sampleError < bestSampleError)
                    ) {
                        val propSeg = (tseg1 / 2).coerceAtLeast(1)
                        best = CanBitTiming(
                            propSeg = propSeg,
                            phaseSeg1 = tseg1 - propSeg,
                            phaseSeg2 = tseg2,
                            sjw = minOf(sjwMax, tseg2).coerceAtLeast(1),
                            brp = brp,
                            actualBitrate = actual.toInt(),
                        )
                        bestBitrateError = bitrateError
                        bestSampleError = sampleError
                    }
                }
            }
            brp += brpInc
        }
        val result = best ?: error("no valid CAN bit timing")
        require(bestBitrateError * 100 <= requestedBitrate.toLong()) {
            "cannot produce 500 kbit/s from adapter clock"
        }
        return result
    }

    private fun isSupportedGsUsbDevice(device: UsbDevice): Boolean =
        KNOWN_DEVICES.any { it.first == device.vendorId && it.second == device.productId } &&
            findGsUsbInterface(device) != null

    private fun findGsUsbInterface(device: UsbDevice): UsbInterface? {
        for (i in 0 until device.interfaceCount) {
            val candidate = device.getInterface(i)
            var hasBulkIn = false
            var hasBulkOut = false
            for (e in 0 until candidate.endpointCount) {
                val endpoint = candidate.getEndpoint(e)
                if (endpoint.type != UsbConstants.USB_ENDPOINT_XFER_BULK) continue
                if (endpoint.direction == UsbConstants.USB_DIR_IN) hasBulkIn = true
                if (endpoint.direction == UsbConstants.USB_DIR_OUT) hasBulkOut = true
            }
            if (hasBulkIn && hasBulkOut) return candidate
        }
        return null
    }

    private fun findBulkInEndpoint(usbInterface: UsbInterface): UsbEndpoint? {
        for (i in 0 until usbInterface.endpointCount) {
            val endpoint = usbInterface.getEndpoint(i)
            if (endpoint.type == UsbConstants.USB_ENDPOINT_XFER_BULK &&
                endpoint.direction == UsbConstants.USB_DIR_IN
            ) {
                return endpoint
            }
        }
        return null
    }

    @Suppress("DEPRECATION")
    private fun Intent.usbDevice(): UsbDevice? =
        if (android.os.Build.VERSION.SDK_INT >= 33) {
            getParcelableExtra(UsbManager.EXTRA_DEVICE, UsbDevice::class.java)
        } else {
            getParcelableExtra(UsbManager.EXTRA_DEVICE)
        }

    private fun leInts(vararg values: Int): ByteArray =
        ByteBuffer.allocate(values.size * Int.SIZE_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
            .apply { values.forEach(::putInt) }
            .array()

    private fun ByteArray.int32Le(offset: Int): Int =
        (this[offset].toInt() and 0xff) or
            ((this[offset + 1].toInt() and 0xff) shl 8) or
            ((this[offset + 2].toInt() and 0xff) shl 16) or
            ((this[offset + 3].toInt() and 0xff) shl 24)

    private data class CanBitTiming(
        val propSeg: Int,
        val phaseSeg1: Int,
        val phaseSeg2: Int,
        val sjw: Int,
        val brp: Int,
        val actualBitrate: Int,
    )

    companion object {
        private const val TAG = "GsUsbCanReceiver"
        private const val ACTION_USB_PERMISSION = "com.ptcdepth.android.USB_CAN_PERMISSION"
        private const val AGILEX_CAN_BITRATE = 500_000
        private const val CHANNEL = 0

        private const val BREQ_HOST_FORMAT = 0
        private const val BREQ_BITTIMING = 1
        private const val BREQ_MODE = 2
        private const val BREQ_BT_CONST = 4
        private const val BREQ_DEVICE_CONFIG = 5
        private const val MODE_RESET = 0
        private const val MODE_START = 1
        private const val MODE_NORMAL_FLAGS = 0

        private const val USB_VENDOR_INTERFACE_OUT = 0x41
        private const val USB_VENDOR_INTERFACE_IN = 0xC1
        private const val USB_CONTROL_TIMEOUT_MS = 1_000
        private const val USB_READ_TIMEOUT_MS = 500

        private const val GS_USB_HEADER_SIZE = 12
        private const val GS_USB_CLASSIC_FRAME_SIZE = 20
        private const val GS_USB_RX_ECHO_ID = -1
        private const val CAN_ERROR_FLAG = 0x20000000
        private const val CAN_EFF_MASK = 0x1fffffff
        private const val TARGET_SAMPLE_POINT = 0.875

        // Same gs_usb devices matched by the upstream Linux driver.
        private val KNOWN_DEVICES = listOf(
            0x1d50 to 0x606f,
            0x1209 to 0x2323,
            0x1cd2 to 0x606f,
            0x16d0 to 0x10b8,
            0x16d0 to 0x0f30,
            0x1209 to 0xca01,
        )
    }
}
