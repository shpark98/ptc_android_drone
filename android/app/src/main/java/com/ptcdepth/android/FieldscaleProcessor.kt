package com.ptcdepth.android

/** Compatibility no-op; thermal Fieldscale processing was removed. */
object FieldscaleProcessor {
    fun configure(maxDiffC: Float, minDiffC: Float, iterations: Int, gamma: Float, clahe: Boolean) = Unit
    fun process(temperatures: DoubleArray, width: Int, height: Int): IntArray? = null
}
