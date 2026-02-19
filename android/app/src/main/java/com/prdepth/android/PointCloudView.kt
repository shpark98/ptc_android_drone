package com.prdepth.android

import android.content.Context
import android.opengl.GLSurfaceView
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.ScaleGestureDetector

/**
 * GLSurfaceView for 3D point cloud with touch rotation and pinch zoom.
 */
class PointCloudView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : GLSurfaceView(context, attrs) {

    val renderer = PointCloudRenderer()

    private var previousX = 0f
    private var previousY = 0f
    private val scaleDetector: ScaleGestureDetector

    init {
        setEGLContextClientVersion(2)
        setRenderer(renderer)
        renderMode = RENDERMODE_CONTINUOUSLY

        scaleDetector = ScaleGestureDetector(context,
            object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
                override fun onScale(detector: ScaleGestureDetector): Boolean {
                    renderer.zoom /= detector.scaleFactor
                    renderer.zoom = renderer.zoom.coerceIn(0.5f, 50f)
                    return true
                }
            })
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        scaleDetector.onTouchEvent(event)

        if (event.pointerCount == 1) {
            when (event.action) {
                MotionEvent.ACTION_MOVE -> {
                    val dx = event.x - previousX
                    val dy = event.y - previousY

                    renderer.rotationX += dy * 0.3f
                    renderer.rotationY += dx * 0.3f
                }
            }
            previousX = event.x
            previousY = event.y
        }

        return true
    }
}
