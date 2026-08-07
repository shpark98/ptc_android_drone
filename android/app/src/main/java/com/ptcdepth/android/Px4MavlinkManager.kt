package com.ptcdepth.android

import android.content.Context
import android.hardware.usb.UsbConstants
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbDeviceConnection
import android.hardware.usb.UsbEndpoint
import android.hardware.usb.UsbInterface
import android.hardware.usb.UsbManager
import com.hoho.android.usbserial.driver.UsbSerialPort
import com.hoho.android.usbserial.driver.UsbSerialProber
import com.hoho.android.usbserial.driver.ProbeTable
import com.hoho.android.usbserial.driver.CdcAcmSerialDriver
import com.hoho.android.usbserial.util.SerialInputOutputManager
import io.dronefleet.mavlink.MavlinkConnection
import io.dronefleet.mavlink.common.Heartbeat
import io.dronefleet.mavlink.common.MavAutopilot
import io.dronefleet.mavlink.common.MavState
import io.dronefleet.mavlink.common.MavType
import io.dronefleet.mavlink.common.Timesync
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

/**
 * PX4 link facade. It deliberately does not arm or enter Offboard by itself.
 * The caller opts into individual command messages through [sendMessage].
 */
class Px4MavlinkManager(
    private val context: Context,
    private val scope: CoroutineScope = CoroutineScope(Dispatchers.IO),
    private val listener: Listener
) {
    companion object {
        // Keep the first hardware test receive-only. Sending resumes after
        // PX4 -> phone MAVLink reception has been confirmed.
        private const val ENABLE_AUTOMATIC_HEARTBEAT = true
        // The phone is a VIO component mounted on the same vehicle as PX4.
        // Components share the vehicle system ID and use a unique component ID.
        private const val APP_SYSTEM_ID = 1
        private const val APP_COMPONENT_ID = 197 // MAV_COMP_ID_VISUAL_INERTIAL_ODOMETRY
    }
    interface Listener {
        fun onLinkState(state: State, detail: String = "")
        fun onMessage(message: io.dronefleet.mavlink.MavlinkMessage<*>)
        fun onError(error: Throwable)
    }

    enum class State { DISCONNECTED, WAITING_USB_PERMISSION, CONNECTING, CONNECTED, STOPPED, ERROR }

    private var connection: MavlinkConnection? = null
    private var transport: CloseableTransport? = null
    private var receiveJob: Job? = null
    private var heartbeatJob: Job? = null
    private var rawDiagnosticJob: Job? = null
    private var timesyncJob: Job? = null
    private val running = AtomicBoolean(false)
    private val rawMavlinkSequence = AtomicInteger(0)
    @Volatile private var px4TimeOffsetUs: Long = 0L
    @Volatile private var timesyncLocked: Boolean = false
    @Volatile private var timesyncRxCount: Long = 0L
    @Volatile private var timesyncRttUs: Long = 0L
    @Volatile private var bestTimesyncRttNs: Long = Long.MAX_VALUE
    @Volatile private var lastTimesyncRequestUs: Long = 0L
    private val timesyncTxCount = AtomicLong(0)
    private val timesyncTxErrors = AtomicLong(0)
    private val rawFrameCount = AtomicLong(0)
    private val rawMessageCounts = ConcurrentHashMap<Int, AtomicLong>()
    private val mavlinkParseErrors = AtomicLong(0)

    /** Connect to a PX4 USB CDC ACM interface already granted by Android. */
    fun connectUsb(device: UsbDevice): Boolean {
        stop()
        listener.onLinkState(State.CONNECTING, "USB ${device.deviceName}")
        return try {
            val usb = context.getSystemService(Context.USB_SERVICE) as UsbManager
            val pair = try {
                openUsbSerialTransport(usb, device)
            } catch (serialError: Throwable) {
                listener.onLinkState(State.CONNECTING, "QGC-style serial driver unavailable; using raw USB fallback: ${serialError.message}")
                openUsbBulkTransport(usb, device)
            }
            transport = pair
            startConnection(pair.input, pair.output, pair.diagnostic)
            true
        } catch (t: Throwable) {
            listener.onLinkState(State.ERROR, t.message ?: "USB open failed")
            listener.onError(t)
            false
        }
    }

    /** Connect to the PX4 MAVLink UDP endpoint (typically 14540/14550). */
    fun connectUdp(host: String, port: Int = 14540, localPort: Int = 0): Boolean {
        stop()
        listener.onLinkState(State.CONNECTING, "UDP $host:$port")
        return try {
            val socket = DatagramSocket(localPort)
            val address = InetAddress.getByName(host)
            val transport = UdpTransport(socket, address, port)
            this.transport = transport
            startConnection(transport.input, transport.output)
            true
        } catch (t: Throwable) {
            listener.onLinkState(State.ERROR, t.message ?: "UDP open failed")
            listener.onError(t)
            false
        }
    }

    @Synchronized
    fun sendMessage(message: Any) {
        val current = connection ?: throw IllegalStateException("PX4 link is not connected")
        // Identify as the vehicle's VIO component. Using QGC's 255/190 address
        // makes PX4 route replies to the telemetry GCS instead of this USB link.
        @Suppress("UNCHECKED_CAST")
        current.send(io.dronefleet.mavlink.MavlinkMessage(APP_SYSTEM_ID, APP_COMPONENT_ID, message))
    }

    fun isConnected(): Boolean = running.get() && connection != null

    fun toPx4TimestampUs(androidTimestampUs: Long): Long = androidTimestampUs + px4TimeOffsetUs
    fun timesyncStatus(): String = if (timesyncLocked) "TS OK %.1fms".format(timesyncRttUs / 1_000.0) else "TS --/${timesyncRxCount}"

    private fun acceptTimesyncResponse(tc1: Long, ts1: Long, nowNs: Long) {
        // Request: tc1=0, ts1=our send time.
        // Response: tc1=PX4 receive time, ts1=our original send time.
        val rttNs = nowNs - ts1
        if (rttNs < 0L || rttNs > 100_000_000L) return
        val sampleOffsetUs = (tc1 - ((ts1 + nowNs) / 2L)) / 1_000L
        // Prefer the lowest-latency USB samples and smooth only nearby samples.
        if (!timesyncLocked || rttNs < bestTimesyncRttNs || rttNs <= bestTimesyncRttNs + 2_000_000L) {
            px4TimeOffsetUs = if (!timesyncLocked) sampleOffsetUs else
                (px4TimeOffsetUs * 9L + sampleOffsetUs) / 10L
            bestTimesyncRttNs = minOf(bestTimesyncRttNs, rttNs)
            timesyncRttUs = rttNs / 1_000L
            timesyncLocked = true
        }
    }

    @Synchronized
    private fun sendRawTimesync(tc1: Long, ts1: Long) {
        try {
            val out = transport?.output ?: return
            val payload = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN).putLong(tc1).putLong(ts1).array()
            val packet = ByteArray(28)
            packet[0] = 0xFD.toByte(); packet[1] = 16; packet[2] = 0; packet[3] = 0
            packet[4] = (rawMavlinkSequence.getAndIncrement() and 0xFF).toByte()
            packet[5] = APP_SYSTEM_ID.toByte(); packet[6] = APP_COMPONENT_ID.toByte(); packet[7] = 111; packet[8] = 0; packet[9] = 0
            payload.copyInto(packet, 10)
            var crc = 0xFFFF
            for (i in 1 until 10) crc = crcAccumulate(packet[i].toInt() and 0xFF, crc)
            for (b in payload) crc = crcAccumulate(b.toInt() and 0xFF, crc)
            crc = crcAccumulate(34, crc)
            packet[26] = (crc and 0xFF).toByte(); packet[27] = (crc ushr 8).toByte()
            out.write(packet); out.flush()
            timesyncTxCount.incrementAndGet()
        } catch (error: Throwable) {
            timesyncTxErrors.incrementAndGet()
            throw error
        }
    }

    private fun handleRawTimesyncFrame(frame: ByteArray) {
        if (frame.size < 8) return
        val magic = frame[0].toInt() and 0xFF
        val msgId = if (magic == 0xFD && frame.size >= 12) {
            (frame[7].toInt() and 0xFF) or ((frame[8].toInt() and 0xFF) shl 8) or
                ((frame[9].toInt() and 0xFF) shl 16)
        } else if (magic == 0xFE) frame[5].toInt() and 0xFF else return
        rawFrameCount.incrementAndGet()
        rawMessageCounts.computeIfAbsent(msgId) { AtomicLong(0) }.incrementAndGet()
        if (msgId != 111) return
        val payloadOffset = if (magic == 0xFD) 10 else 6
        if (frame.size < payloadOffset + 16) return
        val payload = ByteBuffer.wrap(frame, payloadOffset, 16).order(ByteOrder.LITTLE_ENDIAN)
        val tc1 = payload.long
        val ts1 = payload.long
        val nowNs = System.nanoTime()
        timesyncRxCount++
        if (tc1 == 0L) sendRawTimesync(nowNs, ts1)
        else {
            val wasLocked = timesyncLocked
            acceptTimesyncResponse(tc1, ts1, nowNs)
            // The HUD continuously shows TS health. Keep only the first lock in
            // the terminal instead of adding a line for every 2 Hz response.
            if (!wasLocked && timesyncLocked) {
                listener.onLinkState(State.CONNECTED, "TIMESYNC locked; offset=${px4TimeOffsetUs}us rtt=${timesyncRttUs}us")
            }
        }
    }

    private fun handleTimesync(message: io.dronefleet.mavlink.MavlinkMessage<*>) {
        if (!message.payload.javaClass.simpleName.contains("Timesync", ignoreCase = true)) return
        try {
            val payload = message.payload
            timesyncRxCount++
            fun read(name: String): Long {
                val method = payload.javaClass.methods.firstOrNull { it.name == name || it.name.equals("get${name.replaceFirstChar { it.uppercase() }}", true) }
                    ?: error("Timesync field $name unavailable")
                return (method.invoke(payload) as Number).toLong()
            }
            val tc1 = read("tc1")
            val ts1 = read("ts1")
            val nowNs = System.nanoTime()
            if (tc1 == 0L) sendRawTimesync(nowNs, ts1)
            else {
                px4TimeOffsetUs = (tc1 - ((ts1 + nowNs) / 2L)) / 1_000L
                timesyncLocked = true
                listener.onLinkState(State.CONNECTED, "TIMESYNC offset=${px4TimeOffsetUs}us")
            }
        } catch (error: Throwable) {
            listener.onLinkState(State.CONNECTED, "TIMESYNC RX parse failed: ${error.message}")
        }
    }

    /** Send MAVLink ODOMETRY (331) directly, bypassing Dronefleet's array serializer. */
    @Synchronized
    fun sendRawOdometry(
        timestampUs: Long,
        x: Float, y: Float, z: Float,
        qw: Float, qx: Float, qy: Float, qz: Float,
        vx: Float, vy: Float, vz: Float,
        rollSpeed: Float, pitchSpeed: Float, yawSpeed: Float,
    ) {
        val out = transport?.output ?: throw IllegalStateException("PX4 link is not connected")
        // MAVLink ODOMETRY payload (233 bytes), in wire-field order.
        // MAVLink serialization reorders fields by type size: uint64, floats,
        // then the uint8 frame IDs, reset counter, estimator type, and quality.
        val payload = ByteBuffer.allocate(233).order(ByteOrder.LITTLE_ENDIAN)
        payload.putLong(timestampUs)
        payload.putFloat(x).putFloat(y).putFloat(z)
        payload.putFloat(qw).putFloat(qx).putFloat(qy).putFloat(qz)
        payload.putFloat(vx).putFloat(vy).putFloat(vz)
        payload.putFloat(rollSpeed).putFloat(pitchSpeed).putFloat(yawSpeed)
        repeat(42) { payload.putFloat(Float.NaN) }
        payload.put(20) // MAV_FRAME_LOCAL_FRD
        payload.put(12) // MAV_FRAME_BODY_FRD
        payload.put(0) // reset_counter
        payload.put(3) // estimator_type: MAV_ESTIMATOR_TYPE_VIO
        payload.put(100) // quality (valid/high-quality VIO sample)

        val packet = ByteArray(12 + payload.array().size)
        packet[0] = 0xFD.toByte()
        packet[1] = payload.array().size.toByte()
        packet[2] = 0 // incompat flags
        packet[3] = 0 // compat flags
        packet[4] = (rawMavlinkSequence.getAndIncrement() and 0xFF).toByte()
        packet[5] = APP_SYSTEM_ID.toByte()
        packet[6] = APP_COMPONENT_ID.toByte()
        packet[7] = 331.toByte()
        packet[8] = (331 shr 8).toByte()
        packet[9] = 0
        payload.array().copyInto(packet, 10)
        var crc = 0xFFFF
        for (i in 1 until 10) crc = crcAccumulate(packet[i].toInt() and 0xFF, crc)
        for (b in payload.array()) crc = crcAccumulate(b.toInt() and 0xFF, crc)
        crc = crcAccumulate(91, crc) // ODOMETRY CRC_EXTRA
        packet[packet.lastIndex - 1] = (crc and 0xFF).toByte()
        packet[packet.lastIndex] = (crc ushr 8).toByte()
        out.write(packet)
        out.flush()
    }

    private fun crcAccumulate(value: Int, current: Int): Int {
        var tmp = (value xor (current and 0xFF)) and 0xFF
        // MAVLink's reference implementation stores tmp in uint8_t. Kotlin's
        // Int does not overflow at 8 bits, so explicitly truncate here. Without
        // this mask every manually-built frame has a bad checksum and PX4
        // silently discards it before dispatching the message.
        tmp = (tmp xor (tmp shl 4)) and 0xFF
        return ((current ushr 8) xor (tmp shl 8) xor (tmp shl 3) xor (tmp ushr 4)) and 0xFFFF
    }

    fun stop() {
        running.set(false)
        receiveJob?.cancel()
        heartbeatJob?.cancel()
        rawDiagnosticJob?.cancel()
        timesyncJob?.cancel()
        receiveJob = null
        heartbeatJob = null
        timesyncJob = null
        connection = null
        transport?.close()
        transport = null
        listener.onLinkState(State.STOPPED)
    }

    private fun startConnection(input: InputStream, output: OutputStream, diagnostic: String = "") {
        timesyncTxCount.set(0); timesyncTxErrors.set(0); rawFrameCount.set(0)
        mavlinkParseErrors.set(0); rawMessageCounts.clear(); timesyncRxCount = 0
        timesyncLocked = false; px4TimeOffsetUs = 0L; timesyncRttUs = 0L
        bestTimesyncRttNs = Long.MAX_VALUE
        val countedInput = CountingInputStream(input, ::handleRawTimesyncFrame)
        val link = MavlinkConnection.create(countedInput, output)
        connection = link
        running.set(true)
        listener.onLinkState(
            State.CONNECTED,
            "USB opened; $diagnostic; RX test active; VIO component 1/197"
        )
        // lifecycleScope defaults to the main thread. MAVLink next() blocks
        // while waiting for USB/UDP bytes, so both loops must stay on IO.
        receiveJob = scope.launch(Dispatchers.IO) {
            try {
                listener.onLinkState(State.CONNECTED, "RX waiting: waiting for PX4 MAVLink frames")
                var firstFrame = true
                while (isActive && running.get()) {
                    val message = try {
                        link.next()
                    } catch (parseError: Throwable) {
                        mavlinkParseErrors.incrementAndGet()
                        // A malformed/partial frame must not terminate the USB RX loop.
                        delay(5L)
                        null
                    }
                    if (message != null) {
                        if (firstFrame) {
                            firstFrame = false
                            listener.onLinkState(State.CONNECTED, "RX SUCCESS: ${message.payload.javaClass.simpleName} received; receive path verified")
                        }
                        listener.onMessage(message)
                    }
                }
            } catch (t: Throwable) {
                if (running.get()) {
                    listener.onLinkState(State.ERROR, t.message ?: "MAVLink receive failed")
                    listener.onError(t)
                }
            }
        }
        rawDiagnosticJob = scope.launch(Dispatchers.IO) {
            while (isActive && running.get()) {
                // Detailed healthy-link diagnostics remain available without
                // flooding the in-app terminal.
                delay(10_000)
                val topIds = rawMessageCounts.entries
                    .sortedByDescending { it.value.get() }
                    .take(8)
                    .joinToString(",") { "${it.key}:${it.value.get()}" }
                listener.onLinkState(
                    State.CONNECTED,
                    "DIAG RXbytes=${countedInput.totalBytes} rawFrames=${rawFrameCount.get()} " +
                        "ids=[$topIds] parseErr=${mavlinkParseErrors.get()} " +
                        "TS(tx=${timesyncTxCount.get()},txErr=${timesyncTxErrors.get()},rx=$timesyncRxCount," +
                        "locked=$timesyncLocked,offset=$px4TimeOffsetUs,rttUs=$timesyncRttUs)"
                )
            }
        }
        timesyncJob = scope.launch(Dispatchers.IO) {
            while (isActive && running.get()) {
                val nowNs = System.nanoTime()
                lastTimesyncRequestUs = nowNs / 1_000L
                try {
                    sendMessage(Timesync.builder().tc1(0L).ts1(nowNs).build())
                    timesyncTxCount.incrementAndGet()
                } catch (_: Throwable) {
                    timesyncTxErrors.incrementAndGet()
                }
                delay(500L)
            }
        }
        if (!ENABLE_AUTOMATIC_HEARTBEAT) return
        heartbeatJob = scope.launch(Dispatchers.IO) {
            try {
                // Give CDC ACM a moment to settle before the first host packet.
                delay(1000)
                listener.onLinkState(State.CONNECTED, "TX starting: VIO component Heartbeat")
                var firstHeartbeat = true
                while (isActive && running.get()) {
                    sendMessage(
                        Heartbeat.builder()
                            .type(MavType.MAV_TYPE_ONBOARD_CONTROLLER)
                            .autopilot(MavAutopilot.MAV_AUTOPILOT_INVALID)
                            .customMode(0)
                            .systemStatus(MavState.MAV_STATE_ACTIVE)
                            .mavlinkVersion(3)
                            .build()
                    )
                    if (firstHeartbeat) {
                        firstHeartbeat = false
                        listener.onLinkState(State.CONNECTED, "TX SUCCESS: VIO Heartbeat active")
                    }
                    delay(1000)
                }
            } catch (t: Throwable) {
                if (running.get()) {
                    listener.onLinkState(State.ERROR, "TX FAILED: cannot send GCS Heartbeat; ${t.message}")
                    listener.onError(t)
                }
            }
        }
    }

    private fun openUsbBulkTransport(usb: UsbManager, device: UsbDevice): UsbBulkTransport {
        val connection = usb.openDevice(device) ?: error("USB permission missing or device busy")
        var selected: UsbInterface? = null
        var controlId = 0
        var input: UsbEndpoint? = null
        var output: UsbEndpoint? = null
        for (i in 0 until device.interfaceCount) {
            val iface = device.getInterface(i)
            var inEp: UsbEndpoint? = null
            var outEp: UsbEndpoint? = null
            for (j in 0 until iface.endpointCount) {
                val ep = iface.getEndpoint(j)
                if (ep.type != UsbConstants.USB_ENDPOINT_XFER_BULK) continue
                if (ep.direction == UsbConstants.USB_DIR_IN) inEp = ep else outEp = ep
            }
            if (inEp != null && outEp != null) {
                selected = iface
                input = inEp
                output = outEp
                controlId = (0 until device.interfaceCount)
                    .map { device.getInterface(it) }
                    .firstOrNull { it.interfaceClass == UsbConstants.USB_CLASS_COMM }
                    ?.id ?: iface.id
                break
            }
        }
        if (selected == null || input == null || output == null) {
            connection.close()
            error("No USB bulk IN/OUT interface found")
        }
        if (!connection.claimInterface(selected, true)) {
            connection.close()
            error("Unable to claim PX4 USB interface")
        }
        return UsbBulkTransport(connection, selected, input, output, controlId)
    }

    private fun openUsbSerialTransport(usb: UsbManager, device: UsbDevice): UsbSerialTransport {
        // Match QGroundControl's explicit CDC probe table. The default prober
        // does not reliably associate all PX4/Holybro composite VID/PID pairs.
        val probeTable = UsbSerialProber.getDefaultProbeTable().apply {
            addProduct(0x26AC, 0x0010, CdcAcmSerialDriver::class.java)
            addProduct(0x26AC, 0x0011, CdcAcmSerialDriver::class.java)
            addProduct(0x26AC, 0x0012, CdcAcmSerialDriver::class.java)
            addProduct(0x26AC, 0x0032, CdcAcmSerialDriver::class.java)
            addProduct(0x26AC, 0x0033, CdcAcmSerialDriver::class.java)
            addProduct(0x26AC, 0x0035, CdcAcmSerialDriver::class.java)
            addProduct(0x26AC, 0x0036, CdcAcmSerialDriver::class.java)
            addProduct(0x3162, 0x0047, CdcAcmSerialDriver::class.java)
            addProduct(0x3162, 0x0049, CdcAcmSerialDriver::class.java)
            addProduct(0x3162, 0x004B, CdcAcmSerialDriver::class.java)
            // KoaFC re-enumeration ID observed on this phone.
            addProduct(0x1B8C, 0x0036, CdcAcmSerialDriver::class.java)
        }
        val driver = UsbSerialProber(probeTable)
            .findAllDrivers(usb)
            .firstOrNull { it.device == device }
            ?: error("No CDC/USB-serial driver for ${device.deviceName}")
        val port = driver.ports.firstOrNull() ?: error("USB serial device has no port")
        val connection = usb.openDevice(device) ?: error("USB permission missing or device busy")
        try {
            port.open(connection)
            port.setParameters(115200, 8, UsbSerialPort.STOPBITS_1, UsbSerialPort.PARITY_NONE)
            return UsbSerialTransport(port, connection, "usb-serial driver port=${port.portNumber}")
        } catch (error: Throwable) {
            runCatching { port.close() }
            connection.close()
            throw error
        }
    }

    private interface CloseableTransport {
        val input: InputStream
        val output: OutputStream
        val diagnostic: String
        fun close()
    }

    private class UsbSerialTransport(
        private val port: UsbSerialPort,
        private val connection: UsbDeviceConnection,
        override val diagnostic: String
    ) : CloseableTransport {
        private val rxQueue = LinkedBlockingQueue<ByteArray>()
        private var pending = ByteArray(0)
        private var pendingOffset = 0
        private val ioManager = SerialInputOutputManager(port, object : SerialInputOutputManager.Listener {
            override fun onNewData(data: ByteArray) {
                if (data.isNotEmpty()) rxQueue.offer(data.copyOf())
            }

            override fun onRunError(error: Exception) {
                rxQueue.offer(ByteArray(0))
            }
        })

        init {
            // QGroundControl uses SerialInputOutputManager rather than calling
            // UsbSerialPort.read()/write() synchronously from MAVLink threads.
            // A larger TX buffer absorbs short Heartbeat/TIMESYNC bursts while
            // preserving every fresh 30 Hz ODOMETRY frame in wire order.
            ioManager.setWriteBufferSize(16 * 1024)
            ioManager.setWriteTimeout(1000)
            ioManager.start()
        }

        override val input = object : InputStream() {
            override fun read(): Int { val b = ByteArray(1); return if (read(b) == 1) b[0].toInt() and 0xff else -1 }
            override fun read(buffer: ByteArray, offset: Int, length: Int): Int {
                if (length <= 0) return 0
                while (pendingOffset >= pending.size) {
                    pending = rxQueue.poll(1500, TimeUnit.MILLISECONDS) ?: return 0
                    pendingOffset = 0
                    if (pending.isEmpty()) return -1
                }
                val count = minOf(length, pending.size - pendingOffset)
                pending.copyInto(buffer, offset, pendingOffset, pendingOffset + count)
                pendingOffset += count
                return count
            }
        }
        override val output = object : OutputStream() {
            override fun write(b: Int) = write(byteArrayOf(b.toByte()))
            override fun write(buffer: ByteArray, offset: Int, length: Int) {
                // writeAsync copies/enqueues the frame for the dedicated USB
                // I/O thread, so ARCore pose delivery never waits for a bulk
                // transfer to finish.
                val source = buffer.copyOfRange(offset, offset + length)
                ioManager.writeAsync(source)
            }
        }
        override fun close() {
            runCatching { ioManager.stop() }
            runCatching { port.close() }
            connection.close()
        }
    }

    private class CountingInputStream(
        private val source: InputStream,
        private val frameListener: (ByteArray) -> Unit
    ) : InputStream() {
        @Volatile var totalBytes: Long = 0
            private set
        private var frame = ByteArray(300)
        private var frameLength = 0
        private var expectedLength = -1

        private fun feed(value: Int) {
            val b = value and 0xFF
            if (frameLength == 0) {
                if (b != 0xFD && b != 0xFE) return
                frame[0] = b.toByte(); frameLength = 1; expectedLength = -1; return
            }
            if (frameLength >= frame.size) { frameLength = 0; expectedLength = -1; return }
            frame[frameLength++] = b.toByte()
            if (frameLength == 2) {
                val payloadLength = frame[1].toInt() and 0xFF
                expectedLength = if ((frame[0].toInt() and 0xFF) == 0xFD) -1 else payloadLength + 8
            }
            if (frameLength == 3 && (frame[0].toInt() and 0xFF) == 0xFD) {
                val payloadLength = frame[1].toInt() and 0xFF
                val signatureLength = if ((frame[2].toInt() and 0x01) != 0) 13 else 0
                expectedLength = payloadLength + 12 + signatureLength
            }
            if (expectedLength > 0 && frameLength == expectedLength) {
                frameListener(frame.copyOf(frameLength))
                frameLength = 0; expectedLength = -1
            }
        }

        override fun read(): Int {
            val value = source.read()
            if (value >= 0) { totalBytes++; feed(value) }
            return value
        }

        override fun read(buffer: ByteArray, offset: Int, length: Int): Int {
            val count = source.read(buffer, offset, length)
            if (count > 0) {
                totalBytes += count
                for (i in 0 until count) feed(buffer[offset + i].toInt())
            }
            return count
        }

        override fun close() = source.close()
    }

    private class UsbBulkTransport(
        private val connection: UsbDeviceConnection,
        private val iface: UsbInterface,
        inEndpoint: UsbEndpoint,
        outEndpoint: UsbEndpoint,
        controlInterfaceId: Int
    ) : CloseableTransport {
        override val diagnostic: String
        init {
            // PX4 USB serial devices commonly expose CDC ACM. Configure the
            // line before the first MAVLink heartbeat is written.
            val lineCoding = byteArrayOf(
                0x00, 0xC2.toByte(), 0x01, 0x00, // 115200 baud, little endian
                0x00, // 1 stop bit
                0x00, // no parity
                0x08  // 8 data bits
            )
            val codingResult = connection.controlTransfer(0x21, 0x20, 0, controlInterfaceId, lineCoding, lineCoding.size, 1000)
            // Assert DTR only. Some PX4 CDC ACM implementations treat RTS as
            // a hardware-flow-control request and do not start MAVLink when
            // both bits are asserted by an Android host.
            val stateResult = connection.controlTransfer(0x21, 0x22, 0x0001, controlInterfaceId, null, 0, 1000)
            diagnostic = "CDC control iface=$controlInterfaceId lineCoding=$codingResult DTR=$stateResult OUT=${outEndpoint.address} IN=${inEndpoint.address}"
        }
        override val input = UsbInputStream(connection, inEndpoint)
        override val output = UsbOutputStream(connection, outEndpoint)
        override fun close() { runCatching { connection.releaseInterface(iface) }; connection.close() }
    }

    private class UsbInputStream(private val connection: UsbDeviceConnection, private val endpoint: UsbEndpoint) : InputStream() {
        override fun read(): Int { val b = ByteArray(1); return if (read(b) == 1) b[0].toInt() and 0xff else -1 }
        override fun read(buffer: ByteArray, offset: Int, length: Int): Int {
            val target = if (offset == 0) buffer else ByteArray(length)
            while (true) {
                val count = connection.bulkTransfer(endpoint, target, length, 1000)
                // Android returns -1 for a USB transfer timeout. Keep the MAVLink
                // stream alive and let the next transfer wait for a frame.
                if (count >= 0) {
                    if (offset != 0 && count > 0) target.copyInto(buffer, offset, 0, count)
                    return count
                }
            }
        }
    }

    private class UsbOutputStream(private val connection: UsbDeviceConnection, private val endpoint: UsbEndpoint) : OutputStream() {
        override fun write(b: Int) = write(byteArrayOf(b.toByte()))
        override fun write(buffer: ByteArray, offset: Int, length: Int) {
            val source = if (offset == 0) buffer else buffer.copyOfRange(offset, offset + length)
            var result = -1
            repeat(3) {
                result = connection.bulkTransfer(endpoint, source, length, 2000)
                if (result == length) return
            }
            throw IOException("USB bulk write failed: $result/$length endpoint=${endpoint.address}")
        }
    }

    private class UdpTransport(private val socket: DatagramSocket, private val address: InetAddress, private val port: Int) : CloseableTransport {
        override val diagnostic: String = "udp $address:$port"
        override val input: InputStream = object : InputStream() {
            private var current = ByteArray(0)
            private var index = 0
            override fun read(): Int { val b = ByteArray(1); return if (read(b) == 1) b[0].toInt() and 0xff else -1 }
            override fun read(buffer: ByteArray, offset: Int, length: Int): Int {
                if (index >= current.size) {
                    val packet = DatagramPacket(ByteArray(65535), 65535)
                    socket.receive(packet)
                    current = packet.data.copyOf(packet.length)
                    index = 0
                }
                val count = minOf(length, current.size - index)
                current.copyInto(buffer, offset, index, index + count)
                index += count
                return count
            }
        }
        override val output: OutputStream = object : OutputStream() {
            override fun write(b: Int) = write(byteArrayOf(b.toByte()))
            override fun write(buffer: ByteArray, offset: Int, length: Int) {
                socket.send(DatagramPacket(buffer, length, address, port))
            }
        }
        override fun close() = socket.close()
    }
}
