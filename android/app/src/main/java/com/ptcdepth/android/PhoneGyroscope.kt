package com.ptcdepth.android

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread

/** High-rate, bias-compensated phone gyroscope sampled on its own thread. */
class PhoneGyroscope(context: Context) : SensorEventListener {
    data class Sample(val timestampNs: Long, val x: Float, val y: Float, val z: Float)
    data class BodyRates(val roll: Float, val pitch: Float, val yaw: Float)

    private val sensorManager = context.applicationContext
        .getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val sensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val lock = Any()
    private val samples = ArrayDeque<Sample>(256)
    private var thread: HandlerThread? = null

    val sensorName: String get() = sensor?.name ?: "unavailable"

    fun start(): Boolean {
        if (sensor == null) return false
        if (thread != null) return true
        val sensorThread = HandlerThread("PhoneGyroscope").also { it.start() }
        thread = sensorThread
        val handler = Handler(sensorThread.looper)
        val registered = try {
            sensorManager.registerListener(
                this,
                sensor,
                SensorManager.SENSOR_DELAY_FASTEST,
                0,
                handler
            )
        } catch (_: SecurityException) {
            // A vendor build may still restrict high-rate access. Keep the app
            // alive and fall back to the unrestricted game sampling period.
            sensorManager.registerListener(
                this,
                sensor,
                SensorManager.SENSOR_DELAY_GAME,
                0,
                handler
            )
        }
        return registered.also { success ->
            if (!success) stop()
        }
    }

    fun stop() {
        sensorManager.unregisterListener(this)
        thread?.quitSafely()
        thread = null
        synchronized(lock) { samples.clear() }
    }

    /**
     * Fixed landscape mount:
     * - rear camera points vehicle-forward
     * - USB charging port points vehicle-right
     *
     * Android native sensor axes then map to body FRD as:
     * body X(forward)=-phone Z, body Y(right)=-phone Y,
     * body Z(down)=-phone X.
     */
    fun bodyRatesClosestTo(timestampNs: Long): BodyRates? {
        val closest = synchronized(lock) {
            var best: Sample? = null
            var bestDelta = Long.MAX_VALUE
            for (sample in samples) {
                val delta = kotlin.math.abs(sample.timestampNs - timestampNs)
                if (delta < bestDelta) {
                    best = sample
                    bestDelta = delta
                }
            }
            best
        } ?: return null
        return BodyRates(roll = -closest.z, pitch = -closest.y, yaw = -closest.x)
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type != Sensor.TYPE_GYROSCOPE || event.values.size < 3) return
        val sample = Sample(event.timestamp, event.values[0], event.values[1], event.values[2])
        synchronized(lock) {
            samples.addLast(sample)
            while (samples.size > 256) samples.removeFirst()
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
}
