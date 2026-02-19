package com.prdepth.android

import android.app.Activity
import android.content.Context
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.util.Log
import com.google.ar.core.Frame
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * Minimal OpenGL Renderer for ARCore session.
 * Only maintains the ARCore session and delivers frames — no camera rendering.
 * Camera display is handled via bitmap ImageViews for pixel-perfect depth alignment.
 */
class ARCoreRenderer(
    private val context: Context,
    private val arCoreManager: ARCoreManager
) : GLSurfaceView.Renderer {

    private val tag = "ARCoreRenderer"

    @Volatile
    var currentFrame: Frame? = null
        private set

    // Callback invoked on GL thread with fresh frame for depth processing
    @Volatile
    var onNewFrame: ((Frame) -> Unit)? = null

    private var cameraTextureId = -1

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0.0f, 0.0f, 0.0f, 1.0f)

        // Create camera texture for ARCore (required even if not rendered)
        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        cameraTextureId = textures[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTextureId)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)

        // Initialize ARCore session
        try {
            arCoreManager.resume()
            arCoreManager.session?.setCameraTextureName(cameraTextureId)
            Log.i(tag, "ARCore session resumed with texture ID: $cameraTextureId")
        } catch (e: Exception) {
            Log.e(tag, "Failed to resume ARCore in GL context", e)
        }
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        val display = (context as Activity).windowManager.defaultDisplay
        arCoreManager.session?.setDisplayGeometry(display.rotation, width, height)
        Log.i(tag, "Surface: ${width}x${height}")
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)

        val session = arCoreManager.session ?: return

        try {
            val frame = session.update()
            currentFrame = frame
            arCoreManager.currentFrame = frame

            if (frame.camera.trackingState == com.google.ar.core.TrackingState.TRACKING) {
                onNewFrame?.invoke(frame)
            }
        } catch (e: Exception) {
            Log.w(tag, "Exception in onDrawFrame", e)
        }
    }
}
