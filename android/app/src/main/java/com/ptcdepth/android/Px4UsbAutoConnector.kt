package com.ptcdepth.android

import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.hardware.usb.UsbConstants
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import android.os.Handler
import android.os.Looper
import androidx.core.content.ContextCompat

/** Detects PX4-like USB serial devices and completes the Android permission flow. */
class Px4UsbAutoConnector(
    private val context: Context,
    private val onDeviceReady: (UsbDevice) -> Unit,
    private val onStatus: (String) -> Unit
) {
    companion object {
        private const val ACTION_PERMISSION = "com.ptcdepth.android.USB_PX4_PERMISSION"
        private const val PX4_VENDOR_ID = 0x26AC
        private const val RESCAN_INTERVAL_MS = 1_000L
    }

    private val usbManager = context.getSystemService(Context.USB_SERVICE) as UsbManager
    private val mainHandler = Handler(Looper.getMainLooper())
    @Volatile private var permissionRequestInFlight = false
    @Volatile private var started = false
    private var deliveredDeviceId: Int? = null
    private var deniedDeviceId: Int? = null
    private val rescanRunnable = object : Runnable {
        override fun run() {
            if (!started) return
            scanAndRequest()
            // USB attach broadcasts are not reliable on every vendor build.
            // A lightweight device-list check makes hot-plug detection robust.
            mainHandler.postDelayed(this, RESCAN_INTERVAL_MS)
        }
    }
    private val permissionIntent by lazy {
        PendingIntent.getBroadcast(
            context,
            1001,
            Intent(ACTION_PERMISSION).setPackage(context.packageName),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_MUTABLE
        )
    }
    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action == ACTION_PERMISSION) {
                permissionRequestInFlight = false
                val device = intent.usbDeviceExtra() ?: return
                if (intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false)) {
                    deniedDeviceId = null
                    deliveredDeviceId = device.deviceId
                    onStatus("PX4 USB permission granted: ${device.deviceName}")
                    onDeviceReady(device)
                } else {
                    deniedDeviceId = device.deviceId
                    onStatus("PX4 USB permission denied")
                }
            } else if (intent.action == UsbManager.ACTION_USB_DEVICE_ATTACHED) {
                intent.usbDeviceExtra()?.let {
                    deniedDeviceId = null
                    deliveredDeviceId = null
                    if (permissionRequestInFlight) {
                        onStatus("Ignoring duplicate USB attach while permission dialog is open")
                    } else {
                        inspectAndRequest(it)
                    }
                }
            } else if (intent.action == UsbManager.ACTION_USB_DEVICE_DETACHED) {
                intent.usbDeviceExtra()?.let { device ->
                    if (deliveredDeviceId == device.deviceId) deliveredDeviceId = null
                    if (deniedDeviceId == device.deviceId) deniedDeviceId = null
                    permissionRequestInFlight = false
                    onStatus("PX4 USB detached: ${device.deviceName}")
                }
            }
        }
    }

    fun start() {
        if (started) return
        started = true
        val filter = IntentFilter().apply {
            addAction(ACTION_PERMISSION)
            addAction(UsbManager.ACTION_USB_DEVICE_ATTACHED)
            addAction(UsbManager.ACTION_USB_DEVICE_DETACHED)
        }
        ContextCompat.registerReceiver(context, receiver, filter, ContextCompat.RECEIVER_NOT_EXPORTED)
        mainHandler.post(rescanRunnable)
    }

    fun stop() {
        started = false
        mainHandler.removeCallbacks(rescanRunnable)
        permissionRequestInFlight = false
        runCatching { context.unregisterReceiver(receiver) }
    }

    /** Explicit retry used when the PX4 tab is tapped while disconnected. */
    fun retryConnection() {
        deniedDeviceId = null
        deliveredDeviceId = null
        permissionRequestInFlight = false
        scanAndRequest(userInitiated = true)
    }

    private fun scanAndRequest(userInitiated: Boolean = false) {
        val devices = usbManager.deviceList.values
        val target = devices.firstOrNull(::isPx4Candidate)
        if (target == null) {
            if (deliveredDeviceId != null) deliveredDeviceId = null
            if (userInitiated) onStatus("PX4 USB retry: no compatible USB device detected")
            return
        }
        if (permissionRequestInFlight || deliveredDeviceId == target.deviceId) return
        if (!userInitiated && deniedDeviceId == target.deviceId) return
        inspectAndRequest(target)
    }

    private fun inspectAndRequest(device: UsbDevice) {
        onStatus("PX4 USB detected: ${device.deviceName} VID=${device.vendorId.toString(16)} PID=${device.productId.toString(16)}")
        for (i in 0 until device.interfaceCount) {
            val iface = device.getInterface(i)
            val endpoints = (0 until iface.endpointCount).joinToString(",") {
                val ep = iface.getEndpoint(it)
                "${ep.address.toString(16)}:${ep.type}/${ep.direction}/${ep.maxPacketSize}"
            }
            onStatus("USB iface=${iface.id} class=${iface.interfaceClass} endpoints=[$endpoints]")
        }
        if (usbManager.hasPermission(device)) {
            if (deliveredDeviceId == device.deviceId) return
            deliveredDeviceId = device.deviceId
            onDeviceReady(device)
        } else {
            if (permissionRequestInFlight) {
                onStatus("Permission request already open; duplicate request skipped")
                return
            }
            permissionRequestInFlight = true
            onStatus("Requesting PX4 USB permission")
            try {
                usbManager.requestPermission(device, permissionIntent)
            } catch (error: Throwable) {
                permissionRequestInFlight = false
                onStatus("USB permission request failed: ${error.message}")
            }
        }
    }

    private fun isPx4Candidate(device: UsbDevice): Boolean {
        if (device.vendorId == PX4_VENDOR_ID) return true
        // Generic CDC/bulk fallback for boards whose VID is changed by a bootloader.
        for (i in 0 until device.interfaceCount) {
            val iface = device.getInterface(i)
            val hasBulk = (0 until iface.endpointCount).any {
                iface.getEndpoint(it).type == UsbConstants.USB_ENDPOINT_XFER_BULK
            }
            if (hasBulk && (iface.interfaceClass == UsbConstants.USB_CLASS_COMM ||
                    iface.interfaceClass == UsbConstants.USB_CLASS_CDC_DATA)) return true
        }
        return false
    }

    @Suppress("DEPRECATION")
    private fun Intent.usbDeviceExtra(): UsbDevice? =
        if (android.os.Build.VERSION.SDK_INT >= 33) getParcelableExtra(UsbManager.EXTRA_DEVICE, UsbDevice::class.java)
        else getParcelableExtra(UsbManager.EXTRA_DEVICE)

}
