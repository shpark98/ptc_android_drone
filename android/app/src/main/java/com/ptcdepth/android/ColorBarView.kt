package com.ptcdepth.android

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View

/**
 * Vertical colorbar showing Spectral_r gradient with labels.
 * Near (bottom) → Far (top). In metric mode, shows −/+ buttons for vmax.
 */
class ColorBarView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var isMetric = false
    private var vmax = 20f

    /** Called when user taps −/+ to change vmax */
    var onVmaxChanged: ((Float) -> Unit)? = null

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 28f
        setShadowLayer(4f, 1f, 1f, Color.BLACK)
    }
    private val modePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xBBFFFFFF.toInt()
        textSize = 22f
        setShadowLayer(3f, 1f, 1f, Color.BLACK)
    }
    private val btnPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x55FFFFFF.toInt()
        style = Paint.Style.FILL
    }
    private val btnTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 22f
        textAlign = Paint.Align.CENTER
        setShadowLayer(2f, 1f, 1f, Color.BLACK)
    }
    private val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x66FFFFFF.toInt()
        style = Paint.Style.STROKE
        strokeWidth = 1f
    }

    private val barRect = RectF()
    private val minusBtnRect = RectF()
    private val plusBtnRect = RectF()
    private var colormapBitmap: Bitmap? = null

    // Spectral_r LUT (256 entries)
    private val lut = IntArray(256)

    init {
        buildLUT()
    }

    private fun buildLUT() {
        val stops = floatArrayOf(0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f)
        val rS = floatArrayOf(0.369f, 0.199f, 0.400f, 0.665f, 0.902f, 1.000f, 0.996f, 0.991f, 0.957f, 0.831f, 0.620f)
        val gS = floatArrayOf(0.310f, 0.529f, 0.761f, 0.865f, 0.961f, 0.998f, 0.878f, 0.677f, 0.427f, 0.238f, 0.004f)
        val bS = floatArrayOf(0.635f, 0.724f, 0.647f, 0.463f, 0.596f, 0.645f, 0.322f, 0.380f, 0.263f, 0.322f, 0.259f)
        for (i in 0..255) {
            val v = i / 255f
            var idx = 0
            for (j in 0..9) { if (v >= stops[j]) idx = j }
            val t = if (idx < 10) (v - stops[idx]) / (stops[idx + 1] - stops[idx]) else 0f
            val r = rS[idx] + t * (rS[idx + 1] - rS[idx])
            val g = gS[idx] + t * (gS[idx + 1] - gS[idx])
            val b = bS[idx] + t * (bS[idx + 1] - bS[idx])
            lut[i] = Color.rgb((r * 255).toInt().coerceIn(0, 255),
                (g * 255).toInt().coerceIn(0, 255),
                (b * 255).toInt().coerceIn(0, 255))
        }
    }

    fun setMode(metric: Boolean, vmaxValue: Float) {
        isMetric = metric
        vmax = vmaxValue
        invalidate()
    }

    fun setVmax(v: Float) {
        vmax = v.coerceAtLeast(1f)
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()

        val barW = 18f
        val barLeft = 6f
        val barRight = barLeft + barW
        val textX = barRight + 6f

        // Layout: top label → gradient → bottom label → buttons → mode label
        val topLabelY = 24f
        val barTop = 32f
        val barBottom = h - (if (isMetric) 100f else 50f)
        val bottomLabelY = barBottom + 20f
        barRect.set(barLeft, barTop, barRight, barBottom)

        // Draw gradient (top=far/lut[255], bottom=near/lut[0])
        val barH = (barBottom - barTop).toInt()
        if (barH > 0) {
            if (colormapBitmap == null || colormapBitmap!!.height != barH) {
                colormapBitmap = Bitmap.createBitmap(1, barH, Bitmap.Config.ARGB_8888)
            }
            val bmp = colormapBitmap!!
            for (y in 0 until barH) {
                val norm = y.toFloat() / (barH - 1) // 0=top(far), 1=bottom(near)
                val lutIdx = ((1f - norm) * 255f).toInt().coerceIn(0, 255)
                bmp.setPixel(0, y, lut[lutIdx])
            }
            canvas.drawBitmap(bmp, Rect(0, 0, 1, barH), barRect, null)
        }

        // Border
        canvas.drawRoundRect(barRect, 2f, 2f, borderPaint)

        // Top label: far / Xm
        textPaint.textSize = 24f
        textPaint.textAlign = Paint.Align.LEFT
        val topLabel = if (isMetric) "${vmax.toInt()}m" else "far"
        canvas.drawText(topLabel, textX, topLabelY, textPaint)

        // Bottom label: near / 0m
        val bottomLabel = if (isMetric) "0m" else "near"
        canvas.drawText(bottomLabel, textX, bottomLabelY, textPaint)

        // −/+ buttons and mode label
        if (isMetric) {
            val btnSize = 36f
            val btnGap = 8f
            val btnY = bottomLabelY + 10f

            // − button
            minusBtnRect.set(textX, btnY, textX + btnSize, btnY + btnSize)
            canvas.drawRoundRect(minusBtnRect, 6f, 6f, btnPaint)
            btnTextPaint.textSize = 24f
            canvas.drawText("−", minusBtnRect.centerX(), minusBtnRect.centerY() + 8f, btnTextPaint)

            // + button
            plusBtnRect.set(textX + btnSize + btnGap, btnY, textX + btnSize * 2 + btnGap, btnY + btnSize)
            canvas.drawRoundRect(plusBtnRect, 6f, 6f, btnPaint)
            canvas.drawText("+", plusBtnRect.centerX(), plusBtnRect.centerY() + 8f, btnTextPaint)

            // Mode label below buttons
            modePaint.textSize = 18f
            modePaint.textAlign = Paint.Align.LEFT
            canvas.drawText("Metric", textX, btnY + btnSize + 18f, modePaint)
        } else {
            // Mode label below bottom label
            modePaint.textSize = 18f
            modePaint.textAlign = Paint.Align.LEFT
            canvas.drawText("Relative", textX, bottomLabelY + 22f, modePaint)
        }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (event.action == MotionEvent.ACTION_DOWN && isMetric) {
            val x = event.x
            val y = event.y
            // Generous touch areas
            if (minusBtnRect.contains(x, y) ||
                (x >= minusBtnRect.left - 8f && x <= minusBtnRect.right + 8f &&
                 y >= minusBtnRect.top - 8f && y <= minusBtnRect.bottom + 8f)) {
                vmax = (vmax - 5f).coerceAtLeast(1f)
                onVmaxChanged?.invoke(vmax)
                invalidate()
                return true
            }
            if (plusBtnRect.contains(x, y) ||
                (x >= plusBtnRect.left - 8f && x <= plusBtnRect.right + 8f &&
                 y >= plusBtnRect.top - 8f && y <= plusBtnRect.bottom + 8f)) {
                vmax = (vmax + 5f).coerceAtMost(200f)
                onVmaxChanged?.invoke(vmax)
                invalidate()
                return true
            }
        }
        return super.onTouchEvent(event)
    }
}
