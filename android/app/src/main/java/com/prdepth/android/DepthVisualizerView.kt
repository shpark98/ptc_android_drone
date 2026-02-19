package com.prdepth.android

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * View for visualizing depth maps with colormap.
 * Letterbox: shows full depth image without cropping, matching camera preview.
 * Touch shows metric depth at that point.
 */
class DepthVisualizerView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var depthBitmap: Bitmap? = null
    private val paint = Paint().apply {
        isAntiAlias = false
        isFilterBitmap = true
    }
    private val drawMatrix = Matrix()

    // Pre-allocated colormap LUT (256 entries for fast lookup)
    private val colormapLUT = IntArray(256).also { lut ->
        for (i in 0 until 256) {
            lut[i] = spectralRColormap(i / 255f)
        }
    }

    // Pre-allocated pixel buffer to avoid per-frame allocation
    private var pixelBuffer: IntArray? = null

    // Metric depth storage for touch-to-measure
    private var metricDepth: FloatArray? = null
    private var depthWidth = 0
    private var depthHeight = 0

    // Touch marker state
    private var touchX = -1f
    private var touchY = -1f
    private var touchDepthM = Float.NaN
    private var showMarker = false
    private var touchDepthPixelX = -1  // Depth pixel coordinate for live update
    private var touchDepthPixelY = -1
    private val markerPaint = Paint().apply {
        color = Color.WHITE
        textSize = 40f
        isAntiAlias = true
        setShadowLayer(4f, 2f, 2f, Color.BLACK)
    }
    private val crosshairPaint = Paint().apply {
        color = Color.WHITE
        strokeWidth = 2f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    /**
     * Update depth map (normalized [0,1] for colormap display).
     */
    fun updateDepth(depth: FloatArray, width: Int, height: Int) {
        val pixelCount = width * height
        var pixels = pixelBuffer
        if (pixels == null || pixels.size != pixelCount) {
            pixels = IntArray(pixelCount)
            pixelBuffer = pixels
        }

        // Batch convert depth to colormap using LUT
        for (i in 0 until pixelCount) {
            val d = depth[i]
            if (d.isFinite()) {
                val idx = (d.coerceIn(0f, 1f) * 255f).toInt()
                pixels[i] = colormapLUT[idx]
            } else {
                pixels[i] = Color.WHITE
            }
        }

        val bitmap = depthBitmap
        if (bitmap != null && bitmap.width == width && bitmap.height == height) {
            bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        } else {
            val newBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            newBitmap.setPixels(pixels, 0, width, 0, 0, width, height)
            depthBitmap = newBitmap
        }
        postInvalidate()
    }

    /**
     * Display pre-colored ARGB pixels (e.g., flow visualization).
     * Bypasses the depth colormap — pixels are already colored.
     */
    fun updatePixels(pixels: IntArray, width: Int, height: Int) {
        val bitmap = depthBitmap
        if (bitmap != null && bitmap.width == width && bitmap.height == height) {
            bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        } else {
            val newBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            newBitmap.setPixels(pixels, 0, width, 0, 0, width, height)
            depthBitmap = newBitmap
        }
        postInvalidate()
    }

    /**
     * Set raw metric depth data (meters) for touch-to-measure.
     * Call this separately from updateDepth.
     */
    fun setMetricDepth(depth: FloatArray, width: Int, height: Int) {
        metricDepth = depth.copyOf()
        depthWidth = width
        depthHeight = height
        // Live-update marker depth if still visible
        if (showMarker && touchDepthPixelX in 0 until width && touchDepthPixelY in 0 until height) {
            touchDepthM = depth[touchDepthPixelY * width + touchDepthPixelX]
            postInvalidate()
        }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.action) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_MOVE -> {
                touchX = event.x
                touchY = event.y

                // Map touch position to depth pixel
                val bitmap = depthBitmap
                val metric = metricDepth
                if (bitmap != null && metric != null && depthWidth > 0) {
                    val mapping = computeDrawMapping(bitmap)
                    if (mapping != null) {
                        // Invert: screen → depth pixel
                        val depthX = ((touchX - mapping.dx) / mapping.scaleX + mapping.srcLeft).roundToInt()
                        val depthY = ((touchY - mapping.dy) / mapping.scaleY + mapping.srcTop).roundToInt()

                        if (depthX in 0 until depthWidth && depthY in 0 until depthHeight) {
                            touchDepthPixelX = depthX
                            touchDepthPixelY = depthY
                            val idx = depthY * depthWidth + depthX
                            touchDepthM = metric[idx]
                            showMarker = true
                        } else {
                            showMarker = false
                            touchDepthPixelX = -1
                            touchDepthPixelY = -1
                        }
                    }
                }
                invalidate()
                return true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                // Keep marker visible for 5 seconds after release (live-updates with new frames)
                postDelayed({ showMarker = false; touchDepthPixelX = -1; touchDepthPixelY = -1; invalidate() }, 5000)
                return true
            }
        }
        return super.onTouchEvent(event)
    }

    private data class DrawMapping(
        val dx: Float, val dy: Float,
        val scaleX: Float, val scaleY: Float,
        val srcLeft: Float, val srcTop: Float
    )

    /**
     * Compute letterbox mapping: fit full bitmap into view without cropping.
     */
    private fun computeDrawMapping(bitmap: Bitmap): DrawMapping? {
        val viewW = width.toFloat()
        val viewH = height.toFloat()
        if (viewW <= 0 || viewH <= 0) return null

        val bmpW = bitmap.width.toFloat()
        val bmpH = bitmap.height.toFloat()

        val scale = min(viewW / bmpW, viewH / bmpH)
        val scaledW = bmpW * scale
        val scaledH = bmpH * scale
        return DrawMapping(
            dx = (viewW - scaledW) / 2f,
            dy = (viewH - scaledH) / 2f,
            scaleX = scale, scaleY = scale,
            srcLeft = 0f, srcTop = 0f
        )
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val bitmap = depthBitmap ?: return
        val mapping = computeDrawMapping(bitmap) ?: return

        drawMatrix.reset()
        drawMatrix.postTranslate(-mapping.srcLeft, -mapping.srcTop)
        drawMatrix.postScale(mapping.scaleX, mapping.scaleY)
        drawMatrix.postTranslate(mapping.dx, mapping.dy)

        canvas.drawBitmap(bitmap, drawMatrix, paint)

        // Draw touch marker with depth value
        if (showMarker && touchDepthM.isFinite() && touchDepthM > 0f) {
            val viewW = width.toFloat()
            // Crosshair
            val r = 20f
            canvas.drawCircle(touchX, touchY, r, crosshairPaint)
            canvas.drawLine(touchX - r * 1.5f, touchY, touchX + r * 1.5f, touchY, crosshairPaint)
            canvas.drawLine(touchX, touchY - r * 1.5f, touchX, touchY + r * 1.5f, crosshairPaint)

            // Depth text
            val text = "%.2fm".format(touchDepthM)
            val textWidth = markerPaint.measureText(text)

            // Position text above crosshair, clamped to view bounds
            var textX = touchX - textWidth / 2f
            var textY = touchY - r - 12f
            if (textY < markerPaint.textSize) textY = touchY + r + markerPaint.textSize + 4f
            if (textX < 4f) textX = 4f
            if (textX + textWidth > viewW - 4f) textX = viewW - textWidth - 4f

            // Background
            canvas.drawRoundRect(
                textX - 6f, textY - markerPaint.textSize,
                textX + textWidth + 6f, textY + 8f,
                8f, 8f,
                Paint().apply { color = 0xCC000000.toInt() }
            )
            canvas.drawText(text, textX, textY, markerPaint)
        }
    }

    /**
     * Spectral_r colormap (matplotlib).
     * 0.0 = deep purple/blue, 1.0 = deep red/magenta.
     * 0.0 = close (blue/purple), 1.0 = far (red).
     */
    private fun spectralRColormap(value: Float): Int {
        val v = value.coerceIn(0f, 1f)

        // 11 control points sampled from matplotlib Spectral_r
        val stops = floatArrayOf(0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f)
        val rStops = floatArrayOf(0.369f, 0.199f, 0.400f, 0.665f, 0.902f, 1.000f, 0.996f, 0.991f, 0.957f, 0.831f, 0.620f)
        val gStops = floatArrayOf(0.310f, 0.529f, 0.761f, 0.865f, 0.961f, 0.998f, 0.878f, 0.677f, 0.427f, 0.238f, 0.004f)
        val bStops = floatArrayOf(0.635f, 0.739f, 0.647f, 0.643f, 0.596f, 0.745f, 0.545f, 0.378f, 0.263f, 0.309f, 0.259f)

        // Find segment and interpolate
        var idx = 0
        for (i in 0 until stops.size - 1) {
            if (v >= stops[i] && v <= stops[i + 1]) { idx = i; break }
        }
        if (v >= stops.last()) idx = stops.size - 2

        val t = ((v - stops[idx]) / (stops[idx + 1] - stops[idx])).coerceIn(0f, 1f)
        val r = rStops[idx] + t * (rStops[idx + 1] - rStops[idx])
        val g = gStops[idx] + t * (gStops[idx + 1] - gStops[idx])
        val b = bStops[idx] + t * (bStops[idx + 1] - bStops[idx])

        return Color.rgb(
            (r * 255).toInt().coerceIn(0, 255),
            (g * 255).toInt().coerceIn(0, 255),
            (b * 255).toInt().coerceIn(0, 255)
        )
    }
}
