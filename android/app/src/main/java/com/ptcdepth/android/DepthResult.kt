package com.ptcdepth.android

/**
 * Result from the PTC-Depth refinement pipeline.
 */
data class DepthResult(
    val refinedDepth: FloatArray,     // H×W refined depth
    val triDepth: FloatArray,         // H×W triangulated depth
    val R: FloatArray,                // 3×3 estimated rotation (row-major)
    val t: FloatArray,                // 3×1 estimated translation
    val baseline: Float,              // Baseline distance (meters)
    val numMatches: Int,              // Number of matches
    val numValidTri: Int,             // Number of valid triangulated points
    val usedGTPose: Boolean,          // Whether GT pose was used
    val rotationAngleDeg: Float,      // Rotation angle in degrees
)
