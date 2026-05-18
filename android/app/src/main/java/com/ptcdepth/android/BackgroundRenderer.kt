package com.ptcdepth.android

import android.opengl.GLES11Ext
import android.opengl.GLES20
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Background renderer for ARCore camera texture.
 * Supports per-frame UV updates via frame.transformDisplayUvCoords().
 */
class BackgroundRenderer {
    private var quadProgram = 0
    private var quadPositionAttrib = 0
    private var quadTexCoordAttrib = 0
    private var positionBuffer: FloatBuffer? = null
    private var texCoordBuffer: FloatBuffer? = null

    // Vertex shader for camera background
    private val vertexShaderCode = """
        attribute vec4 a_Position;
        attribute vec2 a_TexCoord;
        varying vec2 v_TexCoord;

        void main() {
            gl_Position = a_Position;
            v_TexCoord = a_TexCoord;
        }
    """.trimIndent()

    // Fragment shader for external OES texture
    private val fragmentShaderCode = """
        #extension GL_OES_EGL_image_external : require
        precision mediump float;

        varying vec2 v_TexCoord;
        uniform samplerExternalOES u_Texture;

        void main() {
            gl_FragColor = texture2D(u_Texture, v_TexCoord);
        }
    """.trimIndent()

    fun createOnGlThread() {
        // Full screen quad positions (triangle strip: BL, BR, TL, TR)
        val positions = floatArrayOf(
            -1.0f, -1.0f,  // Bottom-left
             1.0f, -1.0f,  // Bottom-right
            -1.0f,  1.0f,  // Top-left
             1.0f,  1.0f   // Top-right
        )
        positionBuffer = ByteBuffer.allocateDirect(positions.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(positions)
        positionBuffer?.position(0)

        // Default texcoords (will be overridden by transformDisplayUvCoords)
        val texCoords = floatArrayOf(
            0.0f, 1.0f,  // Bottom-left
            1.0f, 1.0f,  // Bottom-right
            0.0f, 0.0f,  // Top-left
            1.0f, 0.0f   // Top-right
        )
        texCoordBuffer = ByteBuffer.allocateDirect(texCoords.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(texCoords)
        texCoordBuffer?.position(0)

        // Load shaders
        val vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vertexShaderCode)
        val fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentShaderCode)

        quadProgram = GLES20.glCreateProgram()
        GLES20.glAttachShader(quadProgram, vertexShader)
        GLES20.glAttachShader(quadProgram, fragmentShader)
        GLES20.glLinkProgram(quadProgram)

        // Get attribute locations
        quadPositionAttrib = GLES20.glGetAttribLocation(quadProgram, "a_Position")
        quadTexCoordAttrib = GLES20.glGetAttribLocation(quadProgram, "a_TexCoord")
    }

    /**
     * Update texture coordinates from frame.transformDisplayUvCoords() output.
     * Must be called on GL thread before draw().
     */
    fun updateTexCoords(transformedCoords: FloatBuffer) {
        texCoordBuffer?.clear()
        val arr = FloatArray(8)
        transformedCoords.get(arr)
        transformedCoords.rewind()
        texCoordBuffer?.put(arr)
        texCoordBuffer?.position(0)
    }

    fun draw(cameraTextureId: Int) {
        if (positionBuffer == null || texCoordBuffer == null) {
            return
        }

        // Use shader program
        GLES20.glUseProgram(quadProgram)

        // Bind camera texture
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTextureId)

        // Set vertex positions (stride=0, tightly packed)
        positionBuffer?.position(0)
        GLES20.glVertexAttribPointer(
            quadPositionAttrib, 2, GLES20.GL_FLOAT, false, 0, positionBuffer
        )
        GLES20.glEnableVertexAttribArray(quadPositionAttrib)

        // Set texture coordinates (stride=0, tightly packed)
        texCoordBuffer?.position(0)
        GLES20.glVertexAttribPointer(
            quadTexCoordAttrib, 2, GLES20.GL_FLOAT, false, 0, texCoordBuffer
        )
        GLES20.glEnableVertexAttribArray(quadTexCoordAttrib)

        // Draw full screen quad
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)

        // Cleanup
        GLES20.glDisableVertexAttribArray(quadPositionAttrib)
        GLES20.glDisableVertexAttribArray(quadTexCoordAttrib)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, 0)
    }

    private fun loadShader(type: Int, shaderCode: String): Int {
        val shader = GLES20.glCreateShader(type)
        GLES20.glShaderSource(shader, shaderCode)
        GLES20.glCompileShader(shader)

        // Check compilation
        val compiled = IntArray(1)
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0)
        if (compiled[0] == 0) {
            android.util.Log.e("BackgroundRenderer", "Shader compilation failed: " +
                GLES20.glGetShaderInfoLog(shader))
            GLES20.glDeleteShader(shader)
            return 0
        }

        return shader
    }
}
