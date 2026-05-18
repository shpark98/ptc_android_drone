package com.ptcdepth.android

import android.content.Context
import android.opengl.GLSurfaceView
import android.util.AttributeSet
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector

/**
 * GLSurfaceView for 3D point cloud with touch rotation, pinch zoom, and double-tap reset.
 */
class PointCloudView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : GLSurfaceView(context, attrs) {

    val renderer = PointCloudRenderer()
    var rotationLocked = false

    private var previousX = 0f
    private var previousY = 0f
    private val scaleDetector: ScaleGestureDetector
    private val gestureDetector: GestureDetector

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

        gestureDetector = GestureDetector(context,
            object : GestureDetector.SimpleOnGestureListener() {
                override fun onDoubleTap(e: MotionEvent): Boolean {
                    resetView()
                    return true
                }
            })
    }

    fun resetView() {
        renderer.rotationX = 0f
        renderer.rotationY = 0f
        renderer.zoom = 3f
        renderer.modelOffsetZ = 0f
    }

    fun setViewPreset(rotX: Float, rotY: Float, zoom: Float, offsetZ: Float = 0f) {
        renderer.rotationX = rotX
        renderer.rotationY = rotY
        renderer.zoom = zoom
        renderer.modelOffsetZ = offsetZ
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        gestureDetector.onTouchEvent(event)
        scaleDetector.onTouchEvent(event)

        if (event.pointerCount == 1 && !rotationLocked) {
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
