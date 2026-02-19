package com.prdepth.android

import android.app.Activity
import android.util.Log
import com.google.ar.core.ArCoreApk
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Session
import com.google.ar.core.Pose
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.UnavailableException
import kotlin.math.sqrt

/**
 * Manages ARCore session and extracts camera poses
 */
class ARCoreManager(private val activity: Activity) {

    internal var session: Session? = null
    private var trackingState: TrackingState = TrackingState.PAUSED

    @Volatile
    internal var currentFrame: com.google.ar.core.Frame? = null

    /**
     * Check if ARCore is available and up-to-date
     */
    fun checkAvailability(): Boolean {
        return try {
            when (ArCoreApk.getInstance().checkAvailability(activity)) {
                ArCoreApk.Availability.SUPPORTED_INSTALLED -> true
                ArCoreApk.Availability.SUPPORTED_APK_TOO_OLD,
                ArCoreApk.Availability.SUPPORTED_NOT_INSTALLED -> {
                    ArCoreApk.getInstance().requestInstall(activity, true)
                    false
                }
                else -> false
            }
        } catch (e: Exception) {
            Log.e(TAG, "ARCore check failed", e)
            false
        }
    }

    /**
     * Initialize ARCore session
     */
    fun initialize(): Boolean {
        return try {
            session = Session(activity).apply {
                // Select lowest resolution camera config (e.g. 640x480 instead of 1920x1080)
                val filter = CameraConfigFilter(this)
                val configs = getSupportedCameraConfigs(filter)
                val smallest = configs.minByOrNull { it.imageSize.width * it.imageSize.height }
                if (smallest != null) {
                    cameraConfig = smallest
                    Log.i(TAG, "Camera config: ${smallest.imageSize.width}x${smallest.imageSize.height}")
                }

                configure(config.apply {
                    planeFindingMode = com.google.ar.core.Config.PlaneFindingMode.HORIZONTAL_AND_VERTICAL
                    depthMode = com.google.ar.core.Config.DepthMode.AUTOMATIC
                })
            }
            Log.i(TAG, "ARCore session initialized")
            true
        } catch (e: UnavailableException) {
            Log.e(TAG, "Failed to initialize ARCore", e)
            false
        }
    }

    /**
     * Get current camera pose from cached frame
     * Note: currentFrame is updated by ARCoreRenderer in GL thread
     */
    fun getCurrentPose(): CameraPose? {
        val frame = this.currentFrame ?: return null

        return try {
            val camera = frame.camera

            trackingState = camera.trackingState

            if (trackingState != TrackingState.TRACKING) {
                return null
            }

            val pose = camera.pose

            // Extract rotation matrix (3x3) and translation (3x1)
            val rotation = FloatArray(9)
            val translation = FloatArray(3)

            // ARCore's toMatrix() returns column-major 4x4 (OpenGL convention):
            //   [ m[0] m[4] m[8]  m[12] ]
            //   [ m[1] m[5] m[9]  m[13] ]
            //   [ m[2] m[6] m[10] m[14] ]
            //   [ m[3] m[7] m[11] m[15] ]
            // Extract 3x3 rotation as row-major: rotation[row*3+col]
            val matrix = FloatArray(16)
            pose.toMatrix(matrix, 0)
            rotation[0] = matrix[0]; rotation[1] = matrix[4]; rotation[2] = matrix[8]
            rotation[3] = matrix[1]; rotation[4] = matrix[5]; rotation[5] = matrix[9]
            rotation[6] = matrix[2]; rotation[7] = matrix[6]; rotation[8] = matrix[10]

            // Translation = last column of 4x4 matrix
            translation[0] = matrix[12]
            translation[1] = matrix[13]
            translation[2] = matrix[14]

            // Extract quaternion (xyzw) for use in relative pose computation
            val quat = FloatArray(4)
            pose.getRotationQuaternion(quat, 0)

            CameraPose(
                rotation = rotation,
                translation = translation,
                quaternion = quat,
                timestamp = frame.timestamp
            )

        } catch (e: Exception) {
            Log.e(TAG, "Failed to get current pose", e)
            null
        }
    }

    /**
     * Compute relative pose between two poses
     */
    fun computeRelativePose(prev: CameraPose, curr: CameraPose): RelativePose {
        // Create ARCore Pose objects with actual quaternions
        val prevPose = Pose(prev.translation, prev.quaternion)
        val currPose = Pose(curr.translation, curr.quaternion)

        // Compute relative transformation: T_curr^{-1} * T_prev  (prev→curr point transform)
        // Pipeline expects P_curr = R * P_prev + baseline * t
        val relativePose = currPose.inverse().compose(prevPose)

        // Extract relative rotation (3x3) as row-major from column-major 4x4
        val R_ar = FloatArray(9)
        val relMatrix = FloatArray(16)
        relativePose.toMatrix(relMatrix, 0)
        R_ar[0] = relMatrix[0]; R_ar[1] = relMatrix[4]; R_ar[2] = relMatrix[8]
        R_ar[3] = relMatrix[1]; R_ar[4] = relMatrix[5]; R_ar[5] = relMatrix[9]
        R_ar[6] = relMatrix[2]; R_ar[7] = relMatrix[6]; R_ar[8] = relMatrix[10]

        // Extract relative translation
        val t_ar = relativePose.translation

        // Convert from ARCore (OpenGL: Y-up, Z-backward) to CV (Y-down, Z-forward)
        // C = diag(1, -1, -1): R_cv = C * R_ar * C, t_cv = C * t_ar
        val R_rel = floatArrayOf(
             R_ar[0], -R_ar[1], -R_ar[2],
            -R_ar[3],  R_ar[4],  R_ar[5],
            -R_ar[6],  R_ar[7],  R_ar[8]
        )
        val t_cv = floatArrayOf(t_ar[0], -t_ar[1], -t_ar[2])

        // Compute baseline (distance)
        val baseline = sqrt(t_cv[0] * t_cv[0] + t_cv[1] * t_cv[1] + t_cv[2] * t_cv[2])

        // Normalize translation to unit vector
        val t_rel = if (baseline > 1e-6f) {
            floatArrayOf(t_cv[0] / baseline, t_cv[1] / baseline, t_cv[2] / baseline)
        } else {
            floatArrayOf(0f, 0f, 1f)  // Default: forward in CV convention (+Z)
        }

        return RelativePose(
            R = R_rel,
            t = t_rel,
            baseline = baseline
        )
    }

    /**
     * Get camera intrinsics from cached frame
     * Note: currentFrame is updated by ARCoreRenderer in GL thread
     */
    fun getCameraIntrinsics(): CameraIntrinsics? {
        val frame = this.currentFrame ?: return null

        return try {
            val camera = frame.camera

            // Use imageIntrinsics (raw CPU image, always landscape 640x480)
            val intrinsics = camera.imageIntrinsics

            // Debug: log both to verify which is rotated
            val texIntr = camera.textureIntrinsics
            Log.d(TAG, "imageIntrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f dim=%dx%d".format(
                intrinsics.focalLength[0], intrinsics.focalLength[1],
                intrinsics.principalPoint[0], intrinsics.principalPoint[1],
                intrinsics.imageDimensions[0], intrinsics.imageDimensions[1]))
            Log.d(TAG, "textureIntrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f dim=%dx%d".format(
                texIntr.focalLength[0], texIntr.focalLength[1],
                texIntr.principalPoint[0], texIntr.principalPoint[1],
                texIntr.imageDimensions[0], texIntr.imageDimensions[1]))

            CameraIntrinsics(
                fx = intrinsics.focalLength[0],
                fy = intrinsics.focalLength[1],
                cx = intrinsics.principalPoint[0],
                cy = intrinsics.principalPoint[1],
                width = intrinsics.imageDimensions[0],
                height = intrinsics.imageDimensions[1]
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get intrinsics", e)
            null
        }
    }

    /**
     * Get current camera image as Bitmap
     * Note: For now returns null - actual camera image extraction would require
     * implementing OpenGL texture reading or using ARCore's camera image API
     */
    fun getCameraImage(): android.graphics.Bitmap? {
        // TODO: Implement camera image extraction from ARCore
        // This requires either:
        // 1. Using ARCore's acquireCameraImage() (YUV format, needs conversion)
        // 2. Reading from OpenGL texture (requires GL context)
        // For now, return null and we'll use dummy data
        return null
    }

    /**
     * Pause session
     */
    fun pause() {
        session?.pause()
    }

    /**
     * Resume session
     */
    fun resume() {
        session?.resume()
    }

    /**
     * Destroy session
     */
    fun destroy() {
        session?.close()
        session = null
        Log.i(TAG, "ARCore session destroyed")
    }

    companion object {
        private const val TAG = "ARCoreManager"
    }
}

/**
 * Camera pose at a specific timestamp
 */
data class CameraPose(
    val rotation: FloatArray,      // 3x3 rotation matrix (row-major)
    val translation: FloatArray,   // 3x1 translation vector
    val quaternion: FloatArray,    // xyzw quaternion (for ARCore Pose recreation)
    val timestamp: Long
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as CameraPose

        if (!rotation.contentEquals(other.rotation)) return false
        if (!translation.contentEquals(other.translation)) return false
        if (!quaternion.contentEquals(other.quaternion)) return false
        if (timestamp != other.timestamp) return false

        return true
    }

    override fun hashCode(): Int {
        var result = rotation.contentHashCode()
        result = 31 * result + translation.contentHashCode()
        result = 31 * result + quaternion.contentHashCode()
        result = 31 * result + timestamp.hashCode()
        return result
    }
}

/**
 * Relative pose between two frames
 */
data class RelativePose(
    val R: FloatArray,             // 3x3 relative rotation
    val t: FloatArray,             // 3x1 relative translation (unit vector)
    val baseline: Float            // Distance between frames (meters)
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as RelativePose

        if (!R.contentEquals(other.R)) return false
        if (!t.contentEquals(other.t)) return false
        if (baseline != other.baseline) return false

        return true
    }

    override fun hashCode(): Int {
        var result = R.contentHashCode()
        result = 31 * result + t.contentHashCode()
        result = 31 * result + baseline.hashCode()
        return result
    }
}

/**
 * Camera intrinsics
 */
data class CameraIntrinsics(
    val fx: Float,
    val fy: Float,
    val cx: Float,
    val cy: Float,
    val width: Int,
    val height: Int
)
