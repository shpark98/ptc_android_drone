package com.ptcdepth.android

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import com.flir.thermalsdk.ErrorCode
import com.flir.thermalsdk.androidsdk.image.BitmapAndroid
import com.flir.thermalsdk.androidsdk.live.connectivity.UsbPermissionHandler
import com.flir.thermalsdk.image.PaletteManager
import com.flir.thermalsdk.image.Rectangle
import com.flir.thermalsdk.live.Camera
import com.flir.thermalsdk.live.CommunicationInterface
import com.flir.thermalsdk.live.ConnectParameters
import com.flir.thermalsdk.live.discovery.DiscoveredCamera
import com.flir.thermalsdk.live.discovery.DiscoveryEventListener
import com.flir.thermalsdk.live.discovery.DiscoveryFactory
import com.flir.thermalsdk.live.streaming.ThermalImageStreamListener
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

/** Owns the FLIR USB discovery, connection, and thermal-image stream. */
class FlirCameraController(
    context: Context,
    private val onStatus: (String) -> Unit,
    private val onFrame: (Bitmap) -> Unit,
    private val onFps: (Float, Boolean) -> Unit,
    private val onFieldscaleFrame: ((Long, IntArray, DoubleArray, Int, Int, Int) -> Unit)? = null,
) {
    private val appContext = context.applicationContext
    private val diagnosticLog = FlirDiagnosticLog(appContext)
    private val worker = Executors.newSingleThreadExecutor()
    private val fieldscaleWorker = Executors.newSingleThreadExecutor()
    private val retryWorker = Executors.newSingleThreadScheduledExecutor()
    private val usbPermissionHandler = UsbPermissionHandler()
    private val palettes = PaletteManager.getDefaultPalettes()
    @Volatile private var paletteIndex = palettes.indexOfFirst {
        it.name.contains("iron", ignoreCase = true)
    }.takeIf { it >= 0 } ?: 0
    private val session = AtomicInteger(0)
    private val cameraClaimed = AtomicBoolean(false)
    private val framePending = AtomicBoolean(false)
    private val fieldscaleWorkerRunning = AtomicBoolean(false)
    private val latestFieldscaleRequest = AtomicReference<FieldscaleRequest?>(null)
    private val lastFieldscaleCaptureMs = AtomicLong(0L)
    private val fieldscaleRequestCounter = AtomicInteger(0)
    @Volatile private var retryFuture: ScheduledFuture<*>? = null

    @Volatile private var running = false
    @Volatile private var camera: Camera? = null
    @Volatile private var rotationDegrees = 0
    // Keep the thermal-processing path enabled by default so entering the FLIR
    // tab immediately produces the left image and DAV2/PTC result side by side.
    @Volatile private var useFieldscale = true
    @Volatile private var displayFieldscaleOutput = false
    @Volatile private var paletteDirty = true
    private var inputFpsWindowStartMs = 0L
    private var inputFpsFrames = 0
    private var outputFpsWindowStartMs = 0L
    private var outputFpsFrames = 0

    private data class FieldscaleRequest(
        val id: Int,
        val timestampNs: Long,
        val temperatures: DoubleArray,
        val width: Int,
        val height: Int,
        val rotation: Int,
        val session: Int,
    )

    private val streamListener = object : ThermalImageStreamListener {
        override fun onImageReceived() {
            // FLIR's stream callback does not expose a capture timestamp.  Capture
            // the callback time immediately and keep it on the Android monotonic
            // clock so it can be matched against ARCore timestamps.
            val callbackTimestampNs = SystemClock.elapsedRealtimeNanos()
            val connectedCamera = camera ?: return
            reportInputFrame()
            if (!running || !framePending.compareAndSet(false, true)) return

            try {
                if (useFieldscale) {
                    val nowMs = SystemClock.elapsedRealtime()
                    val previousMs = lastFieldscaleCaptureMs.get()
                    if (FIELDSCALE_CAPTURE_INTERVAL_MS > 0L &&
                        nowMs - previousMs < FIELDSCALE_CAPTURE_INTERVAL_MS ||
                        !lastFieldscaleCaptureMs.compareAndSet(previousMs, nowMs)
                    ) {
                        framePending.set(false)
                        return
                    }

                    val requestId = fieldscaleRequestCounter.incrementAndGet()
                    var request: FieldscaleRequest? = null
                    val captureStartMs = SystemClock.elapsedRealtime()
                    diagnosticLog.write("WITH_IMAGE_START", "id=$requestId listener=stream")
                    connectedCamera.withImage(this) { thermalImage ->
                        // The SDK updates the rendered live image independently
                        // from its radiometric cache. Force the current live frame
                        // to be decoded before reading its temperature field.
                        val refreshedImage = thermalImage.image
                        diagnosticLog.write(
                            "IMAGE_REFRESH",
                            "id=$requestId size=${refreshedImage.width}x${refreshedImage.height} " +
                                "format=${refreshedImage.format}",
                        )
                        val width = thermalImage.width
                        val height = thermalImage.height
                        diagnosticLog.write("GET_VALUES_START", "id=$requestId size=${width}x$height")
                        val temperatures = thermalImage.getValues(Rectangle(0, 0, width, height))
                        var fingerprint = 1L
                        for (index in temperatures.indices step 997) {
                            fingerprint = fingerprint * 31L +
                                java.lang.Double.doubleToLongBits(temperatures[index])
                        }
                        diagnosticLog.write(
                            "GET_VALUES_END",
                            "id=$requestId count=${temperatures.size} fingerprint=$fingerprint",
                        )
                        request = FieldscaleRequest(
                            id = requestId,
                            timestampNs = callbackTimestampNs,
                            temperatures = temperatures,
                            width = width,
                            height = height,
                            rotation = rotationDegrees,
                            session = session.get(),
                        )
                    }
                    diagnosticLog.write(
                        "WITH_IMAGE_END",
                        "id=$requestId ms=${SystemClock.elapsedRealtime() - captureStartMs}",
                    )
                    framePending.set(false)
                    latestFieldscaleRequest.set(request)
                    diagnosticLog.write("FS_QUEUED", "id=$requestId")
                    scheduleFieldscaleDrain()
                    return
                }

                connectedCamera.withImage(this) { thermalImage ->
                    var sdkBitmap: BitmapAndroid? = null
                    try {
                        // Applying a palette can trigger SDK-side image work;
                        // do it only on the first frame or after the user
                        // changes the selected palette, not on every frame.
                        if (paletteDirty) {
                            palettes.getOrNull(paletteIndex)?.let { thermalImage.palette = it }
                            paletteDirty = false
                        }
                        val source = BitmapAndroid.createBitmap(thermalImage.image).also {
                            sdkBitmap = it
                        }.bitMap
                        val frame = if (rotationDegrees == 0) {
                            source.copy(Bitmap.Config.ARGB_8888, false)
                        } else {
                            Bitmap.createBitmap(
                                source,
                                0,
                                0,
                                source.width,
                                source.height,
                                Matrix().apply { postRotate(rotationDegrees.toFloat()) },
                                true,
                            )
                        }
                        onFrame(frame)
                    } catch (error: Throwable) {
                        Log.e(TAG, "Failed to render FLIR frame", error)
                        onStatus("FLIR frame error: ${error.message ?: error.javaClass.simpleName}")
                    } finally {
                        sdkBitmap?.recycle()
                        framePending.set(false)
                    }
                }
            } catch (error: Throwable) {
                framePending.set(false)
                Log.e(TAG, "Failed to acquire FLIR frame", error)
                onStatus("FLIR stream error: ${error.message ?: error.javaClass.simpleName}")
            }
        }
    }

    private fun scheduleFieldscaleDrain() {
        if (!fieldscaleWorkerRunning.compareAndSet(false, true)) return
        fieldscaleWorker.execute {
            try {
                while (true) {
                    val request = latestFieldscaleRequest.getAndSet(null) ?: break
                    if (!running || !useFieldscale || request.session != session.get()) continue
                    try {
                        diagnosticLog.write("FS_BEGIN", "id=${request.id}")
                        val processStartMs = SystemClock.elapsedRealtime()
                        diagnosticLog.write("FIELDSCALE_START", "id=${request.id}")
                        val pixels = FieldscaleProcessor.process(
                            request.temperatures,
                            request.width,
                            request.height,
                        ) ?: throw IllegalStateException("Fieldscale returned no image")
                        onFieldscaleFrame?.invoke(
                            request.timestampNs,
                            pixels,
                            request.temperatures,
                            request.width,
                            request.height,
                            request.rotation,
                        )
                        diagnosticLog.write(
                            "FIELDSCALE_END",
                            "id=${request.id} ms=${SystemClock.elapsedRealtime() - processStartMs}",
                        )
                        val source = Bitmap.createBitmap(
                            request.width,
                            request.height,
                            Bitmap.Config.ARGB_8888,
                        ).also {
                            it.setPixels(
                                pixels,
                                0,
                                request.width,
                                0,
                                0,
                                request.width,
                                request.height,
                            )
                        }
                        val frame = if (request.rotation == 0) {
                            source
                        } else {
                            Bitmap.createBitmap(
                                source,
                                0,
                                0,
                                source.width,
                                source.height,
                                Matrix().apply { postRotate(request.rotation.toFloat()) },
                                true,
                            ).also { source.recycle() }
                        }
                        if (running && useFieldscale && displayFieldscaleOutput && request.session == session.get()) {
                            onFrame(frame)
                            reportOutputFrame()
                            diagnosticLog.write("FS_OUTPUT", "id=${request.id}")
                        } else {
                            frame.recycle()
                        }
                    } catch (error: Throwable) {
                        diagnosticLog.write(
                            "FS_ERROR",
                            "id=${request.id} ${error.javaClass.simpleName}: ${error.message}",
                        )
                        Log.e(TAG, "Failed to render Fieldscale frame", error)
                        onStatus("Fieldscale error: ${error.message ?: error.javaClass.simpleName}")
                    }
                }
            } finally {
                fieldscaleWorkerRunning.set(false)
                if (latestFieldscaleRequest.get() != null) scheduleFieldscaleDrain()
            }
        }
    }

    private fun reportInputFrame() {
        val now = SystemClock.elapsedRealtime()
        if (inputFpsWindowStartMs == 0L) inputFpsWindowStartMs = now
        inputFpsFrames++
        val elapsed = now - inputFpsWindowStartMs
        if (elapsed >= FPS_REPORT_INTERVAL_MS) {
            onFps(inputFpsFrames * 1000f / elapsed, false)
            inputFpsWindowStartMs = now
            inputFpsFrames = 0
        }
    }

    private fun reportOutputFrame() {
        val now = SystemClock.elapsedRealtime()
        if (outputFpsWindowStartMs == 0L) outputFpsWindowStartMs = now
        outputFpsFrames++
        val elapsed = now - outputFpsWindowStartMs
        if (elapsed >= FPS_REPORT_INTERVAL_MS) {
            onFps(outputFpsFrames * 1000f / elapsed, true)
            outputFpsWindowStartMs = now
            outputFpsFrames = 0
        }
    }

    private val discoveryListener = object : DiscoveryEventListener {
        override fun onCameraFound(discoveredCamera: DiscoveredCamera) {
            if (!running || !cameraClaimed.compareAndSet(false, true)) return
            diagnosticLog.write("DISCOVERY_FOUND", discoveredCamera.displayName)
            cancelDiscoveryRetry()
            DiscoveryFactory.getInstance().stop(CommunicationInterface.USB)
            connectAfterUsbPermission(discoveredCamera)
        }

        override fun onDiscoveryError(
            communicationInterface: CommunicationInterface,
            error: ErrorCode,
        ) {
            diagnosticLog.write("DISCOVERY_ERROR", "$communicationInterface $error")
            Log.e(TAG, "FLIR discovery error on $communicationInterface: $error")
            cameraClaimed.set(false)
            if (running) scheduleDiscoveryRetry(session.get())
            onStatus("FLIR search error: ${error.message}")
        }

        override fun onDiscoveryFinished(communicationInterface: CommunicationInterface) {
            diagnosticLog.write("DISCOVERY_FINISHED", communicationInterface.toString())
            if (running && !cameraClaimed.get()) {
                scheduleDiscoveryRetry(session.get())
                onStatus("No FLIR USB camera found · tap to retry")
            }
        }
    }

    fun start() {
        if (running) return
        diagnosticLog.write("START", "session=${session.get() + 1}")
        running = true
        cameraClaimed.set(false)
        val currentSession = session.incrementAndGet()
        onStatus("Searching for a FLIR USB camera…")

        try {
            DiscoveryFactory.getInstance().scan(
                discoveryListener,
                CommunicationInterface.USB,
            )
        } catch (error: Throwable) {
            if (currentSession == session.get()) {
                Log.e(TAG, "Unable to start FLIR discovery", error)
                onStatus("FLIR search failed: ${error.message ?: error.javaClass.simpleName} · tap to retry")
            }
        }
    }

    fun restart() {
        stop()
        start()
    }

    fun setRotation(degrees: Int) {
        rotationDegrees = ((degrees % 360) + 360) % 360
    }

    fun setDisplayFieldscaleOutput(enabled: Boolean) {
        displayFieldscaleOutput = enabled
    }

    fun currentPaletteName(): String =
        palettes.getOrNull(paletteIndex)?.name ?: "Camera default"

    fun selectNextPalette(): String {
        if (palettes.isNotEmpty()) {
            paletteIndex = (paletteIndex + 1) % palettes.size
            paletteDirty = true
        }
        return currentPaletteName()
    }

    fun toggleFieldscale(): Boolean {
        useFieldscale = !useFieldscale
        diagnosticLog.write("MODE", if (useFieldscale) "Fieldscale" else "FLIR")
        latestFieldscaleRequest.set(null)
        lastFieldscaleCaptureMs.set(if (useFieldscale) SystemClock.elapsedRealtime() else 0L)
        outputFpsWindowStartMs = 0L
        outputFpsFrames = 0
        return useFieldscale
    }

    fun isFieldscaleEnabled(): Boolean = useFieldscale

    fun stop() {
        if (!running && camera == null) return
        diagnosticLog.write("STOP", "session=${session.get()}")
        running = false
        session.incrementAndGet()
        cameraClaimed.set(false)
        framePending.set(false)
        latestFieldscaleRequest.set(null)
        lastFieldscaleCaptureMs.set(0L)
        cancelDiscoveryRetry()
        try {
            DiscoveryFactory.getInstance().stop(CommunicationInterface.USB)
        } catch (error: Throwable) {
            Log.w(TAG, "Failed to stop FLIR discovery", error)
        }

        val oldCamera = camera
        camera = null
        if (oldCamera != null) {
            worker.execute { closeCamera(oldCamera) }
        }
    }

    fun release() {
        stop()
        retryWorker.shutdown()
        fieldscaleWorker.shutdown()
        worker.shutdown()
    }

    private fun connect(discoveredCamera: DiscoveredCamera) {
        val requestedSession = session.get()
        diagnosticLog.write("CONNECT_START", "session=$requestedSession ${discoveredCamera.displayName}")
        onStatus("Connecting to ${discoveredCamera.displayName}…")
        worker.execute {
            val newCamera = Camera()
            try {
                newCamera.connect(
                    discoveredCamera.identity,
                    { error ->
                        diagnosticLog.write("DISCONNECTED", error?.toString() ?: "connection closed")
                        Log.w(TAG, "FLIR camera disconnected: $error")
                        if (camera === newCamera) camera = null
                        if (running && requestedSession == session.get()) {
                            cameraClaimed.set(false)
                            scheduleDiscoveryRetry(requestedSession)
                            try {
                                worker.execute { closeCamera(newCamera) }
                            } catch (_: Throwable) {
                            }
                            val reason = error?.message ?: "connection closed"
                            onStatus("FLIR disconnected: $reason · tap to retry")
                        }
                    },
                    ConnectParameters(CONNECT_TIMEOUT_MS),
                )

                if (!running || requestedSession != session.get()) {
                    closeCamera(newCamera)
                    return@execute
                }

                camera = newCamera
                paletteDirty = true
                newCamera.subscribeStream(streamListener)
                diagnosticLog.write("CONNECTED", "session=$requestedSession ${discoveredCamera.displayName}")
                onStatus("${discoveredCamera.displayName} · live")
                Log.i(TAG, "FLIR camera connected: ${discoveredCamera.displayName}")
            } catch (error: Throwable) {
                diagnosticLog.write(
                    "CONNECT_ERROR",
                    "${error.javaClass.simpleName}: ${error.message}",
                )
                closeCamera(newCamera)
                Log.e(TAG, "Failed to connect FLIR camera", error)
                if (running && requestedSession == session.get()) {
                    cameraClaimed.set(false)
                    scheduleDiscoveryRetry(requestedSession)
                    onStatus("FLIR connection failed: ${error.message ?: error.javaClass.simpleName} · tap to retry")
                }
            }
        }
    }

    private fun connectAfterUsbPermission(discoveredCamera: DiscoveredCamera) {
        val identity = discoveredCamera.identity
        val needsPermission = UsbPermissionHandler.isFlirOne(identity) &&
            !UsbPermissionHandler.hasFlirOnePermission(identity, appContext)
        if (!needsPermission) {
            connect(discoveredCamera)
            return
        }

        onStatus("Allow USB access for ${discoveredCamera.displayName}…")
        try {
            usbPermissionHandler.requestFlirOnePermisson(
                identity,
                appContext,
                object : UsbPermissionHandler.UsbPermissionListener {
                    override fun permissionGranted(grantedIdentity: com.flir.thermalsdk.live.Identity) {
                        if (running && grantedIdentity == identity) {
                            connect(discoveredCamera)
                        }
                    }

                    override fun permissionDenied(deniedIdentity: com.flir.thermalsdk.live.Identity) {
                        cameraClaimed.set(false)
                        if (running && deniedIdentity == identity) {
                            onStatus("FLIR USB permission denied · tap to retry")
                        }
                    }

                    override fun error(
                        errorType: UsbPermissionHandler.UsbPermissionListener.ErrorType?,
                        errorIdentity: com.flir.thermalsdk.live.Identity?,
                    ) {
                        cameraClaimed.set(false)
                        Log.e(TAG, "FLIR USB permission error: $errorType")
                        if (running && (errorIdentity == null || errorIdentity == identity)) {
                            onStatus("FLIR USB permission error: $errorType · tap to retry")
                        }
                    }
                },
            )
        } catch (error: Throwable) {
            cameraClaimed.set(false)
            Log.e(TAG, "Unable to request FLIR USB permission", error)
            onStatus("FLIR USB permission request failed · tap to retry")
        }
    }

    private fun closeCamera(target: Camera) {
        try {
            target.unsubscribeStream(streamListener)
        } catch (_: Throwable) {
        }
        try {
            target.disconnect()
        } catch (_: Throwable) {
        }
        try {
            target.close()
        } catch (_: Throwable) {
        }
    }

    private fun beginDiscovery(requestedSession: Int) {
        if (!running || requestedSession != session.get() || cameraClaimed.get()) return
        cancelDiscoveryRetry()
        onStatus("Searching for a FLIR USB camera...")
        try {
            DiscoveryFactory.getInstance().scan(
                discoveryListener,
                CommunicationInterface.USB,
            )
        } catch (error: Throwable) {
            if (running && requestedSession == session.get()) {
                Log.e(TAG, "Unable to retry FLIR discovery", error)
                scheduleDiscoveryRetry(requestedSession)
            }
        }
    }

    private fun scheduleDiscoveryRetry(requestedSession: Int) {
        if (!running || requestedSession != session.get() || cameraClaimed.get()) return
        cancelDiscoveryRetry()
        try {
            retryFuture = retryWorker.schedule(
                { beginDiscovery(requestedSession) },
                DISCOVERY_RETRY_DELAY_MS,
                TimeUnit.MILLISECONDS,
            )
        } catch (_: Throwable) {
            // The controller is being released.
        }
    }

    private fun cancelDiscoveryRetry() {
        retryFuture?.cancel(false)
        retryFuture = null
    }

    companion object {
        private const val TAG = "FlirCameraController"
        private const val CONNECT_TIMEOUT_MS = 10_000L
        // No app-side throttle: let Fieldscale consume the camera's ~8.7 Hz
        // stream. The latest-request queue still drops stale work if native
        // processing falls behind, preventing latency from growing unbounded.
        private const val FIELDSCALE_CAPTURE_INTERVAL_MS = 0L
        private const val FPS_REPORT_INTERVAL_MS = 1000L
        private const val DISCOVERY_RETRY_DELAY_MS = 1_500L
    }
}
