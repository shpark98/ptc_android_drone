package com.ptcdepth.android

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class AgilexCanDecoderTest {
    @Test
    fun decodesProtocolV2OdometryAsSignedBigEndianCounts() {
        val decoder = AgilexCanDecoder()

        val snapshot = decoder.accept(
            CanFrame(
                id = 0x311,
                data = byteArrayOf(0x00, 0x01, 0x02, 0x03, 0xff.toByte(), 0xff.toByte(), 0xff.toByte(), 0xfe.toByte()),
                receivedAtNanos = 123,
            )
        )!!

        assertEquals(0x00010203, snapshot.leftOdometry)
        assertEquals(-2, snapshot.rightOdometry)
        assertEquals(1, snapshot.receivedFrames)
        assertEquals(0x311, snapshot.lastCanId)
    }

    @Test
    fun decodesProtocolV2MotorHighSpeedState() {
        val decoder = AgilexCanDecoder()

        val snapshot = decoder.accept(
            CanFrame(
                id = 0x253,
                data = byteArrayOf(0xff.toByte(), 0x9c.toByte(), 0x00, 0x7b, 0x01, 0x02, 0x03, 0x04),
            )
        )!!

        val motor = snapshot.motors.single()
        assertEquals(3, motor.motorId)
        assertEquals(-100, motor.rpm)
        assertEquals(12.3f, motor.currentAmps, 0.0001f)
        assertEquals(0x01020304, motor.pulseCount)
    }

    @Test
    fun ignoresCanFramesUnrelatedToWheelEncoders() {
        val decoder = AgilexCanDecoder()

        assertNull(decoder.accept(CanFrame(0x221, ByteArray(8))))
        assertEquals(0, decoder.snapshot().receivedFrames)
    }
}
