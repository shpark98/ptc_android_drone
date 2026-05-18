package com.ptcdepth.android

import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.abs
import kotlin.math.max

/**
 * OpenGL ES 2.0 point cloud renderer.
 * Generates 3D points from depth map + camera intrinsics.
 * Features: voxel grid filtering, edge filtering, round point sprites.
 */
class PointCloudRenderer : GLSurfaceView.Renderer {

    // Touch-controlled camera
    var rotationX = 0f
    var rotationY = 0f
    var zoom = 3.0f
    var modelOffsetZ = 0f  // Model-space Z offset (shifts scene forward for top view)

    // Matrices
    private val projMatrix = FloatArray(16)
    private val viewMatrix = FloatArray(16)
    private val mvpMatrix = FloatArray(16)
    private val modelMatrix = FloatArray(16)
    private val tempMatrix = FloatArray(16)

    // Shader
    private var program = 0
    private var posHandle = 0
    private var colorHandle = 0
    private var mvpHandle = 0
    private var pointSizeHandle = 0

    // Point cloud data
    private var vertexBuffer: FloatBuffer? = null
    private var pointCount = 0
    private val COORDS_PER_VERTEX = 6  // x, y, z, r, g, b
    private val VERTEX_STRIDE = COORDS_PER_VERTEX * 4  // bytes

    // Depth data to process (set from UI thread, consumed on GL thread)
    @Volatile private var pendingDepth: FloatArray? = null
    @Volatile private var pendingWidth = 0
    @Volatile private var pendingHeight = 0
    @Volatile private var pendingFx = 400f
    @Volatile private var pendingFy = 400f
    @Volatile private var pendingCx = 259f
    @Volatile private var pendingCy = 259f
    @Volatile private var pendingMaxDepth = 80f
    @Volatile private var pendingRgbPixels: IntArray? = null  // ARGB packed, same size as depth
    @Volatile var useRGBColor = false

    // Viewport size for GL capture
    var viewportWidth = 0
        private set
    var viewportHeight = 0
        private set

    // Spectral_r colormap LUT (matplotlib)
    private val colormapLUT = Array(256) { i ->
        spectralRColor(i / 255f)
    }

    fun updatePointCloud(
        depth: FloatArray, width: Int, height: Int,
        fx: Float, fy: Float, cx: Float, cy: Float,
        maxDepth: Float = 80f,
        rgbPixels: IntArray? = null
    ) {
        pendingDepth = depth
        pendingWidth = width
        pendingHeight = height
        pendingFx = fx
        pendingFy = fy
        pendingCx = cx
        pendingCy = cy
        pendingMaxDepth = maxDepth
        pendingRgbPixels = rgbPixels
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0.05f, 0.05f, 0.1f, 1.0f)
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)

        val vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, VERTEX_SHADER)
        val fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, FRAGMENT_SHADER)

        program = GLES20.glCreateProgram().also {
            GLES20.glAttachShader(it, vertexShader)
            GLES20.glAttachShader(it, fragmentShader)
            GLES20.glLinkProgram(it)
        }

        posHandle = GLES20.glGetAttribLocation(program, "aPosition")
        colorHandle = GLES20.glGetAttribLocation(program, "aColor")
        mvpHandle = GLES20.glGetUniformLocation(program, "uMVPMatrix")
        pointSizeHandle = GLES20.glGetUniformLocation(program, "uPointSize")
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        viewportWidth = width
        viewportHeight = height
        val ratio = width.toFloat() / height.toFloat()
        Matrix.perspectiveM(projMatrix, 0, 60f, ratio, 0.1f, 200f)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)

        // Process pending depth data
        val depth = pendingDepth
        if (depth != null) {
            pendingDepth = null
            val rgb = pendingRgbPixels
            pendingRgbPixels = null
            generatePoints(depth, pendingWidth, pendingHeight,
                pendingFx, pendingFy, pendingCx, pendingCy,
                pendingMaxDepth, rgb)
        }

        val buf = vertexBuffer ?: return
        if (pointCount == 0) return

        // Camera: look at origin from distance 'zoom'
        Matrix.setLookAtM(viewMatrix, 0,
            0f, 0f, zoom,   // eye
            0f, 0f, 0f,     // center
            0f, -1f, 0f)    // up (Y-down to match depth convention)

        // Model: translate in model space (for top view centering), then rotate
        Matrix.setIdentityM(modelMatrix, 0)
        Matrix.rotateM(modelMatrix, 0, rotationX, 1f, 0f, 0f)
        Matrix.rotateM(modelMatrix, 0, rotationY, 0f, 1f, 0f)
        if (modelOffsetZ != 0f) {
            Matrix.translateM(modelMatrix, 0, 0f, 0f, modelOffsetZ)
        }

        // MVP = Projection * View * Model
        Matrix.multiplyMM(tempMatrix, 0, viewMatrix, 0, modelMatrix, 0)
        Matrix.multiplyMM(mvpMatrix, 0, projMatrix, 0, tempMatrix, 0)

        GLES20.glUseProgram(program)
        GLES20.glUniformMatrix4fv(mvpHandle, 1, false, mvpMatrix, 0)
        GLES20.glUniform1f(pointSizeHandle, max(1.5f, 3.5f - zoom * 0.2f))

        buf.position(0)
        GLES20.glEnableVertexAttribArray(posHandle)
        GLES20.glVertexAttribPointer(posHandle, 3, GLES20.GL_FLOAT, false, VERTEX_STRIDE, buf)

        buf.position(3)
        GLES20.glEnableVertexAttribArray(colorHandle)
        GLES20.glVertexAttribPointer(colorHandle, 3, GLES20.GL_FLOAT, false, VERTEX_STRIDE, buf)

        GLES20.glDrawArrays(GLES20.GL_POINTS, 0, pointCount)

        GLES20.glDisableVertexAttribArray(posHandle)
        GLES20.glDisableVertexAttribArray(colorHandle)
    }

    private fun generatePoints(
        depth: FloatArray, width: Int, height: Int,
        fx: Float, fy: Float, cx: Float, cy: Float,
        maxDepth: Float = 80f,
        rgbPixels: IntArray? = null
    ) {
        val step = 1
        val maxPoints = (width / step) * (height / step)
        val data = FloatArray(maxPoints * COORDS_PER_VERTEX)
        var count = 0
        val clampZ = maxDepth
        val useRgb = useRGBColor && rgbPixels != null && rgbPixels.size == width * height

        for (v in 0 until height step step) {
            for (u in 0 until width step step) {
                val pixIdx = v * width + u
                val z = depth[pixIdx]
                if (z <= 0f || !z.isFinite() || z > clampZ) continue

                // Edge filter: skip flying pixels at depth discontinuities
                val edgeThresh = z * 0.12f
                val dL = if (u >= 2) depth[v * width + u - 2] else z
                val dR = if (u + 2 < width) depth[v * width + u + 2] else z
                val dU = if (v >= 2) depth[(v - 2) * width + u] else z
                val dD = if (v + 2 < height) depth[(v + 2) * width + u] else z
                if ((dL > 0f && abs(z - dL) > edgeThresh) ||
                    (dR > 0f && abs(z - dR) > edgeThresh) ||
                    (dU > 0f && abs(z - dU) > edgeThresh) ||
                    (dD > 0f && abs(z - dD) > edgeThresh)) continue

                val x = (u - cx) * z / fx
                val y = (v - cy) * z / fy

                val r: Float; val g: Float; val b: Float
                if (useRgb) {
                    val argb = rgbPixels!![pixIdx]
                    r = ((argb shr 16) and 0xFF) / 255f
                    g = ((argb shr 8) and 0xFF) / 255f
                    b = (argb and 0xFF) / 255f
                } else {
                    val normalized = (z / clampZ).coerceIn(0f, 1f)
                    val colorIdx = (normalized * 255f).toInt().coerceIn(0, 255)
                    val color = colormapLUT[colorIdx]
                    r = color[0]; g = color[1]; b = color[2]
                }

                val idx = count * COORDS_PER_VERTEX
                data[idx] = -x
                data[idx + 1] = y
                data[idx + 2] = -z
                data[idx + 3] = r
                data[idx + 4] = g
                data[idx + 5] = b
                count++
            }
        }

        pointCount = count
        vertexBuffer = ByteBuffer.allocateDirect(count * VERTEX_STRIDE)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .apply {
                put(data, 0, count * COORDS_PER_VERTEX)
                position(0)
            }
    }

    /**
     * Voxel grid downsampling: partition 3D space into cells,
     * average all points per cell → one output point per occupied cell.
     * Reduces noise and produces uniform density.
     */
    private fun voxelFilter(data: FloatArray, count: Int, cellSize: Float): Pair<FloatArray, Int> {
        if (count == 0) return Pair(FloatArray(0), 0)

        val invCell = 1f / cellSize
        // HashMap key = spatial hash, value = [sumX, sumY, sumZ, sumR, sumG, sumB, count]
        val cells = HashMap<Long, FloatArray>(count / 2)

        for (i in 0 until count) {
            val base = i * COORDS_PER_VERTEX
            val x = data[base]
            val y = data[base + 1]
            val z = data[base + 2]

            val ix = if (x >= 0) (x * invCell).toLong() else (x * invCell).toLong() - 1
            val iy = if (y >= 0) (y * invCell).toLong() else (y * invCell).toLong() - 1
            val iz = if (z >= 0) (z * invCell).toLong() else (z * invCell).toLong() - 1
            val key = ix * 73856093L xor iy * 19349663L xor iz * 83492791L

            val cell = cells.getOrPut(key) { FloatArray(7) }
            cell[0] += x
            cell[1] += y
            cell[2] += z
            cell[3] += data[base + 3]  // r
            cell[4] += data[base + 4]  // g
            cell[5] += data[base + 5]  // b
            cell[6] += 1f
        }

        val out = FloatArray(cells.size * COORDS_PER_VERTEX)
        var idx = 0
        for (cell in cells.values) {
            val n = cell[6]
            out[idx++] = cell[0] / n
            out[idx++] = cell[1] / n
            out[idx++] = cell[2] / n
            out[idx++] = cell[3] / n
            out[idx++] = cell[4] / n
            out[idx++] = cell[5] / n
        }
        return Pair(out, cells.size)
    }

    private fun spectralRColor(value: Float): FloatArray {
        val v = value.coerceIn(0f, 1f)
        val stops = floatArrayOf(0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f)
        val rS = floatArrayOf(0.369f, 0.199f, 0.400f, 0.665f, 0.902f, 1.000f, 0.996f, 0.991f, 0.957f, 0.831f, 0.620f)
        val gS = floatArrayOf(0.310f, 0.529f, 0.761f, 0.865f, 0.961f, 0.998f, 0.878f, 0.677f, 0.427f, 0.238f, 0.004f)
        val bS = floatArrayOf(0.635f, 0.739f, 0.647f, 0.643f, 0.596f, 0.745f, 0.545f, 0.378f, 0.263f, 0.309f, 0.259f)

        var idx = 0
        for (i in 0 until stops.size - 1) {
            if (v >= stops[i] && v <= stops[i + 1]) { idx = i; break }
        }
        if (v >= stops.last()) idx = stops.size - 2
        val t = ((v - stops[idx]) / (stops[idx + 1] - stops[idx])).coerceIn(0f, 1f)
        return floatArrayOf(
            rS[idx] + t * (rS[idx + 1] - rS[idx]),
            gS[idx] + t * (gS[idx + 1] - gS[idx]),
            bS[idx] + t * (bS[idx + 1] - bS[idx])
        )
    }

    private fun loadShader(type: Int, code: String): Int {
        return GLES20.glCreateShader(type).also { shader ->
            GLES20.glShaderSource(shader, code)
            GLES20.glCompileShader(shader)
        }
    }

    companion object {
        private const val VERTEX_SHADER = """
            uniform mat4 uMVPMatrix;
            uniform float uPointSize;
            attribute vec3 aPosition;
            attribute vec3 aColor;
            varying vec3 vColor;
            void main() {
                gl_Position = uMVPMatrix * vec4(aPosition, 1.0);
                gl_PointSize = uPointSize;
                vColor = aColor;
            }
        """

        private const val FRAGMENT_SHADER = """
            precision mediump float;
            varying vec3 vColor;
            void main() {
                gl_FragColor = vec4(vColor, 1.0);
            }
        """
    }
}
