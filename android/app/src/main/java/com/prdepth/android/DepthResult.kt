package com.prdepth.android

/**
 * Result from depth refinement pipeline
 */
data class DepthResult(
    val refinedDepth: FloatArray,     // H×W refined depth
    val triDepth: FloatArray,         // H×W triangulated depth
    val confidence: FloatArray,       // H×W confidence map
    val R: FloatArray,                // 3×3 estimated rotation (row-major)
    val t: FloatArray,                // 3×1 estimated translation
    val baseline: Float,              // Baseline distance (meters)
    val numMatches: Int,              // Number of matches
    val numValidTri: Int,             // Number of valid triangulated points
    val usedGTPose: Boolean,          // Whether GT pose was used
    val rotationAngleDeg: Float,      // Rotation angle in degrees
    val flowVizPixels: IntArray?,     // Flow viz ARGB pixels (Middlebury colormap)
    val flowWidth: Int,               // Flow visualization width
    val flowHeight: Int               // Flow visualization height
)
