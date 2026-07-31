package com.ptcdepth.android

/** A classic CAN frame after transport-specific USB framing has been removed. */
data class CanFrame(
    val id: Int,
    val data: ByteArray,
    val receivedAtNanos: Long = System.nanoTime(),
)

data class AgilexMotorEncoder(
    val motorId: Int,
    val rpm: Int,
    val currentAmps: Float,
    val pulseCount: Int,
)

data class AgilexWheelEncoderSnapshot(
    val leftOdometry: Int? = null,
    val rightOdometry: Int? = null,
    val motors: List<AgilexMotorEncoder> = emptyList(),
    val receivedFrames: Long = 0,
    val lastCanId: Int? = null,
    val updatedAtNanos: Long = 0,
)

/**
 * Decodes the wheel-related Protocol V2 frames used by agilexrobotics/ugv_sdk.
 *
 * - 0x311: left/right 32-bit wheel odometry counts
 * - 0x251..0x258: motor RPM, current and 32-bit encoder pulse count
 *
 * AgileX serializes all multi-byte fields most-significant byte first on CAN.
 */
class AgilexCanDecoder {
    private var leftOdometry: Int? = null
    private var rightOdometry: Int? = null
    private val motors = arrayOfNulls<AgilexMotorEncoder>(MAX_MOTORS)
    private var receivedFrames = 0L
    private var lastCanId: Int? = null
    private var updatedAtNanos = 0L

    @Synchronized
    fun accept(frame: CanFrame): AgilexWheelEncoderSnapshot? {
        val decoded = when {
            frame.id == ODOMETRY_CAN_ID && frame.data.size >= 8 -> {
                leftOdometry = frame.data.int32Be(0)
                rightOdometry = frame.data.int32Be(4)
                true
            }
            frame.id in ACTUATOR_HS_FIRST_ID..ACTUATOR_HS_LAST_ID && frame.data.size >= 8 -> {
                val motorId = frame.id - ACTUATOR_HS_FIRST_ID
                motors[motorId] = AgilexMotorEncoder(
                    motorId = motorId + 1,
                    rpm = frame.data.int16Be(0),
                    currentAmps = frame.data.int16Be(2) * 0.1f,
                    pulseCount = frame.data.int32Be(4),
                )
                true
            }
            else -> false
        }
        if (!decoded) return null

        receivedFrames++
        lastCanId = frame.id
        updatedAtNanos = frame.receivedAtNanos
        return snapshotLocked()
    }

    @Synchronized
    fun snapshot(): AgilexWheelEncoderSnapshot = snapshotLocked()

    private fun snapshotLocked() = AgilexWheelEncoderSnapshot(
        leftOdometry = leftOdometry,
        rightOdometry = rightOdometry,
        motors = motors.filterNotNull(),
        receivedFrames = receivedFrames,
        lastCanId = lastCanId,
        updatedAtNanos = updatedAtNanos,
    )

    private fun ByteArray.int16Be(offset: Int): Int {
        val unsigned = ((this[offset].toInt() and 0xff) shl 8) or
            (this[offset + 1].toInt() and 0xff)
        return unsigned.toShort().toInt()
    }

    private fun ByteArray.int32Be(offset: Int): Int =
        ((this[offset].toInt() and 0xff) shl 24) or
            ((this[offset + 1].toInt() and 0xff) shl 16) or
            ((this[offset + 2].toInt() and 0xff) shl 8) or
            (this[offset + 3].toInt() and 0xff)

    companion object {
        const val ODOMETRY_CAN_ID = 0x311
        const val ACTUATOR_HS_FIRST_ID = 0x251
        const val ACTUATOR_HS_LAST_ID = 0x258
        private const val MAX_MOTORS = 8
    }
}
