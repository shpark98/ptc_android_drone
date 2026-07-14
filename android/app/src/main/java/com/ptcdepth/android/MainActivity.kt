package com.ptcdepth.android

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.util.concurrent.Executors
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

/**
 * Main activity for PTC-Depth Android app.
 *
 * Pipeline:
 *   ARCore camera (YUV 640x480)
 *   → QNN HTP (Depth Anything V2, 518x518) → inv_depth (640x480)
 *   → [optional] PTC-Depth refinement (optical flow + triangulation + fusion)
 *   → Display: Split / Overlay / 3D Point Cloud
 */
class MainActivity : AppCompatActivity() {

    private lateinit var arCoreManager: ARCoreManager
    private var depthEstimatorQNN: DepthEstimatorQNN? = null
    private var depthEstimator: DepthEstimator? = null
    private lateinit var glSurfaceView: GLSurfaceView
    private lateinit var arCoreRenderer: ARCoreRenderer

    // Views
    private lateinit var depthVisualizerSplit: DepthVisualizerView
    private lateinit var depthVisualizerOverlay: DepthVisualizerView
    private lateinit var depthVisualizerCompareModel: DepthVisualizerView
    private lateinit var depthVisualizerCompareARCore: DepthVisualizerView
    private lateinit var splitContainer: LinearLayout
    private lateinit var compareContainer: LinearLayout
    private lateinit var pointCloudView: PointCloudView
    private lateinit var cameraSplit: ImageView
    private lateinit var cameraView: ImageView
    private lateinit var btnToggleRefine: TextView
    private lateinit var btnToggleRGB: Switch
    private lateinit var labelRGB: TextView
    private lateinit var settingsOverlay: FrameLayout
    private lateinit var colorBar: ColorBarView
    private lateinit var presetButtons: LinearLayout
    private lateinit var zoomSliderContainer: LinearLayout

    // Mode tabs
    private lateinit var modeTabs: Array<TextView>

    private lateinit var switchGTPose: Switch
    private lateinit var textViewFPS: TextView
    private lateinit var textViewBaseline: TextView
    private lateinit var textViewRotation: TextView
    private lateinit var textViewMatches: TextView
    private lateinit var textViewAbsRel: TextView

    // PTC-Depth pipeline
    private var depthRefinementMgr: DepthRefinementManager? = null
    private var modelIntrinsics: CameraIntrinsics? = null

    // Data logging
    private var dataLogger: DataLogger? = null
    private lateinit var btnRecordData: TextView

    // 3D controls
    private var rotationLocked = false
    private lateinit var btnLockRotation: TextView

    // Recording playback
    private var recordingPlayer: RecordingPlayer? = null
    private var isPlaybackMode = false
    private lateinit var playbackControls: LinearLayout
    private lateinit var btnPlaybackPlay: TextView
    private lateinit var btnPlaybackClose: TextView
    private lateinit var seekPlayback: SeekBar
    private lateinit var textPlaybackFrame: TextView

    // State
    private var prevPose: CameraPose? = null
    private var useGTPose = false
    @Volatile private var isProcessing = false
    private var viewMode = VIEW_SPLIT  // 0=Split, 1=Overlay, 2=3D
    private var useRefinement = false
    private var showTriDepth = false  // Debug: show raw triangulated depth instead of fused
    private var useRGBColor = false  // 3D point cloud color mode (false=colormap, true=RGB)
    private var maxDepthParam = 80f  // Max depth for pipeline computation (triangulation, clamping)
    private var vmaxParam = 20f     // Vmax for visualization normalization (colormap, point cloud)
    private var paramsVisible = false
    private var isUIReady = false

    private var lastFrameTime = 0L
    private var fpsEma = 0.0              // EMA-smoothed FPS for stable display

    // Pre-allocated buffers for combined pipeline (reused across frames)
    private var yPlaneData: ByteArray? = null
    private var uPlaneData: ByteArray? = null
    private var vPlaneData: ByteArray? = null
    private var depthOutput: FloatArray? = null

    // Camera display bitmap (separate from depth processing buffers)
    private var camDispY: ByteArray? = null
    private var camDispU: ByteArray? = null
    private var camDispV: ByteArray? = null
    private var camDispPixels: IntArray? = null
    private var camDispBitmap: Bitmap? = null
    private val qnnExecutor = Executors.newSingleThreadExecutor()

    // Cache camera strides for PTC-Depth (set once on first frame)
    private var camW = 0
    private var camH = 0
    private var rotDeg = 90

    // One-shot request: next frame, write a colored PLY of the current depth.
    // Captured inside processFrame so depth + YUV + intrinsics are consistent.
    @Volatile private var plySaveRequested = false
    // Continuous capture: every refined frame (rate-limited) is written as a PLY.
    @Volatile private var plyContinuousRecording = false
    private var lastPlySaveMs = 0L
    private val plyMinIntervalMs = 100L  // skip frames closer than this (~10 Hz max)

    // Pre-allocated buffers
    private var normalizedBuf: FloatArray? = null
    private var resizedMonoBuf: FloatArray? = null

    // Metrics: TAE (prev depth for warp-based self-consistency)
    private var prevRefinedDepth: FloatArray? = null
    private var prevDepthW = 0
    private var prevDepthH = 0
    // ARCore depth for accuracy comparison (always fetched)
    private var latestARCoreDepth: FloatArray? = null
    private var arcoreDepthW = 0
    private var arcoreDepthH = 0

    // Background thread for depth estimation
    private lateinit var depthThread: HandlerThread
    private lateinit var depthHandler: Handler

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (!checkPermissions()) {
            requestPermissions()
            return
        }

        if (!initializeARCore()) {
            Toast.makeText(this, "ARCore initialization failed", Toast.LENGTH_LONG).show()
            finish()
            return
        }

        // Start depth processing thread
        depthThread = HandlerThread("DepthEstimation").also { it.start() }
        depthHandler = Handler(depthThread.looper)

        glSurfaceView = findViewById(R.id.gl_surface_view)
        glSurfaceView.preserveEGLContextOnPause = true
        glSurfaceView.setEGLContextClientVersion(2)
        glSurfaceView.setEGLConfigChooser(8, 8, 8, 8, 16, 0)
        arCoreRenderer = ARCoreRenderer(this, arCoreManager)
        glSurfaceView.setRenderer(arCoreRenderer)
        glSurfaceView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY

        val loadingOverlay = findViewById<View>(R.id.loading_overlay)
        val loadingProgress = findViewById<android.widget.ProgressBar>(R.id.loading_progress)
        val loadingPercent = findViewById<TextView>(R.id.loading_percent)
        loadingOverlay.visibility = View.VISIBLE

        Thread {
            try {
                val handler = Handler(mainLooper)
                var shouldContinue = true
                var progress = 0

                val progressUpdater = object : Runnable {
                    override fun run() {
                        if (shouldContinue && progress < 95) {
                            progress += 5
                            loadingProgress.progress = progress
                            loadingPercent.text = "$progress%"
                            handler.postDelayed(this, 100)
                        }
                    }
                }
                handler.post(progressUpdater)

                try {
                    Log.i(TAG, "Loading QNN HTP depth estimator...")
                    depthEstimatorQNN = DepthEstimatorQNN(this)
                    Log.i(TAG, "QNN HTP depth estimator loaded successfully")
                } catch (e: Exception) {
                    Log.w(TAG, "QNN HTP failed, falling back to ONNX Runtime CPU: ${e.message}")
                    depthEstimator = DepthEstimator(this)
                    Log.i(TAG, "ONNX Runtime CPU model loaded (fallback)")
                }

                shouldContinue = false
                handler.removeCallbacks(progressUpdater)

                runOnUiThread {
                    loadingProgress.progress = 100
                    loadingPercent.text = "100%"
                    setupUI()
                    loadingOverlay.visibility = View.GONE
                    Log.i(TAG, "App ready with depth estimation")
                }

            } catch (e: Exception) {
                Log.e(TAG, "Failed to load Depth Anything model", e)
                runOnUiThread {
                    Toast.makeText(this, "Failed to load depth model: ${e.message}", Toast.LENGTH_LONG).show()
                    finish()
                }
            }
        }.start()
    }

    private fun checkPermissions(): Boolean {
        return ContextCompat.checkSelfPermission(
            this, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestPermissions() {
        ActivityCompat.requestPermissions(
            this, arrayOf(Manifest.permission.CAMERA), PERMISSION_REQUEST_CODE
        )
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                recreate()
            } else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
                finish()
            }
        }
    }

    private fun initializeARCore(): Boolean {
        arCoreManager = ARCoreManager(this)
        if (!arCoreManager.checkAvailability()) return false
        if (!arCoreManager.initialize()) return false
        Log.i(TAG, "ARCore initialized")
        return true
    }

    // ========================================================================
    // UI Setup
    // ========================================================================

    private fun setupUI() {
        depthVisualizerSplit = findViewById(R.id.depth_visualizer_split)
        depthVisualizerOverlay = findViewById(R.id.depth_visualizer_overlay)
        depthVisualizerCompareModel = findViewById(R.id.depth_visualizer_compare_model)
        depthVisualizerCompareARCore = findViewById(R.id.depth_visualizer_compare_arcore)
        splitContainer = findViewById(R.id.split_container)
        compareContainer = findViewById(R.id.compare_container)
        pointCloudView = findViewById(R.id.point_cloud_view)
        cameraSplit = findViewById(R.id.camera_split)
        cameraView = findViewById(R.id.camera_view)
        btnToggleRefine = findViewById(R.id.btn_toggle_refine)
        btnToggleRGB = findViewById(R.id.btn_toggle_rgb)
        labelRGB = findViewById(R.id.label_rgb)
        settingsOverlay = findViewById(R.id.settings_overlay)
        colorBar = findViewById(R.id.color_bar)
        presetButtons = findViewById(R.id.preset_buttons)
        zoomSliderContainer = findViewById(R.id.zoom_slider_container)

        // 3D viewpoint presets
        findViewById<TextView>(R.id.btn_preset_front).setOnClickListener {
            pointCloudView.setViewPreset(0f, 0f, 3f)
        }
        findViewById<TextView>(R.id.btn_preset_top).setOnClickListener {
            // Offset Z by vmaxParam/2 so scene is centered in top view
            pointCloudView.setViewPreset(-90f, 0f, 3f, vmaxParam / 2f)
        }
        findViewById<TextView>(R.id.btn_preset_side).setOnClickListener {
            pointCloudView.setViewPreset(0f, -90f, 3f)
        }

        // Lock rotation toggle
        btnLockRotation = findViewById(R.id.btn_lock_rotation)
        btnLockRotation.setOnClickListener {
            rotationLocked = !rotationLocked
            pointCloudView.rotationLocked = rotationLocked
            btnLockRotation.text = if (rotationLocked) "\uD83D\uDD12" else "\uD83D\uDD13"
        }

        // Zoom +/- buttons
        findViewById<TextView>(R.id.btn_zoom_in).setOnClickListener {
            pointCloudView.renderer.zoom = (pointCloudView.renderer.zoom * 0.75f).coerceAtLeast(0.5f)
        }
        findViewById<TextView>(R.id.btn_zoom_out).setOnClickListener {
            pointCloudView.renderer.zoom = (pointCloudView.renderer.zoom * 1.33f).coerceAtMost(50f)
        }

        colorBar.onVmaxChanged = { newVmax ->
            vmaxParam = newVmax
            findViewById<SeekBar>(R.id.seek_vmax).progress = newVmax.toInt()
            findViewById<TextView>(R.id.label_vmax).text = "Vmax (viz): ${newVmax.toInt()}m"
        }

        switchGTPose = findViewById(R.id.switch_gt_pose)
        textViewFPS = findViewById(R.id.text_fps)
        textViewBaseline = findViewById(R.id.text_baseline)
        textViewRotation = findViewById(R.id.text_rotation)
        textViewMatches = findViewById(R.id.text_matches)
        textViewAbsRel = findViewById(R.id.text_absrel)

        // Mode tabs
        val tabSplit = findViewById<TextView>(R.id.tab_split)
        val tabOverlay = findViewById<TextView>(R.id.tab_overlay)
        val tabDepth = findViewById<TextView>(R.id.tab_depth)
        val tabCompare = findViewById<TextView>(R.id.tab_compare)
        val tab3d = findViewById<TextView>(R.id.tab_3d)
        modeTabs = arrayOf(tabSplit, tabOverlay, tabDepth, tabCompare, tab3d)

        modeTabs.forEachIndexed { index, tab ->
            tab.setOnClickListener {
                viewMode = index
                updateViewMode()
            }
        }

        // Refine button: toggles Mono ↔ Refined
        btnToggleRefine.setOnClickListener {
            useRefinement = !useRefinement
            btnToggleRefine.text = if (useRefinement) "Refined" else "Mono"
            if (useRefinement) {
                initPTCDepthIfNeeded()
            }
            updateColorBar()
            // Ensure RGB visibility stays correct for current mode
            val is3D = viewMode == VIEW_3D
            btnToggleRGB.visibility = if (is3D) View.VISIBLE else View.GONE
            labelRGB.visibility = if (is3D) View.VISIBLE else View.GONE
        }

        // RGB toggle: toggles RGB color mode (synced with settings switch)
        btnToggleRGB.setOnCheckedChangeListener { _, isChecked ->
            useRGBColor = isChecked
            findViewById<Switch>(R.id.switch_rgb_color).isChecked = isChecked
        }

        // Data record button
        btnRecordData = findViewById(R.id.btn_record_data)
        btnRecordData.setOnClickListener { toggleDataRecording() }

        // Browse recordings (HUD button)
        findViewById<TextView>(R.id.btn_browse_recordings_hud).setOnClickListener {
            showRecordingBrowser()
        }

        // Save current depth as colored PLY. Toggle: tap to start continuous
        // capture, tap again to stop. Long-press still does a one-shot snapshot.
        val btnSavePly = findViewById<TextView>(R.id.btn_save_ply)
        btnSavePly.setOnClickListener {
            plyContinuousRecording = !plyContinuousRecording
            Log.i(TAG, "PLY button tap: plyContinuousRecording=$plyContinuousRecording")
            if (plyContinuousRecording) {
                lastPlySaveMs = 0L
                btnSavePly.setTextColor(0xFFFF4444.toInt())
                Toast.makeText(this, "PLY 연속 저장 시작", Toast.LENGTH_SHORT).show()
            } else {
                btnSavePly.setTextColor(0xFFFFCC44.toInt())
                Toast.makeText(this, "PLY 연속 저장 중지", Toast.LENGTH_SHORT).show()
            }
        }
        btnSavePly.setOnLongClickListener {
            plySaveRequested = true
            Log.i(TAG, "PLY button long-press: one-shot save requested")
            Toast.makeText(this, "PLY 한 장 저장", Toast.LENGTH_SHORT).show()
            true
        }

        // Playback controls
        playbackControls = findViewById(R.id.playback_controls)
        btnPlaybackPlay = findViewById(R.id.btn_playback_play)
        btnPlaybackClose = findViewById(R.id.btn_playback_close)
        seekPlayback = findViewById(R.id.seek_playback)
        textPlaybackFrame = findViewById(R.id.text_playback_frame)

        btnPlaybackPlay.setOnClickListener { togglePlayback() }
        btnPlaybackClose.setOnClickListener { stopPlaybackMode() }
        seekPlayback.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                if (fromUser) {
                    recordingPlayer?.seekTo(progress)
                }
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {}
        })

        // Settings gear: open settings overlay
        findViewById<TextView>(R.id.btn_settings).setOnClickListener {
            paramsVisible = !paramsVisible
            settingsOverlay.visibility = if (paramsVisible) View.VISIBLE else View.GONE
        }

        // Close settings button
        findViewById<View>(R.id.btn_close_settings).setOnClickListener {
            paramsVisible = false
            settingsOverlay.visibility = View.GONE
        }

        // Tap outside settings dialog to close
        settingsOverlay.setOnClickListener {
            paramsVisible = false
            settingsOverlay.visibility = View.GONE
        }

        // GT Pose switch
        switchGTPose.setOnCheckedChangeListener { _, isChecked ->
            useGTPose = isChecked
            depthRefinementMgr?.setUseExternalPose(isChecked)
        }

        setupParameterPanel()
        updateViewMode()

        textViewMatches.text = "Mono"

        // Set up frame callback
        arCoreRenderer.onNewFrame = { frame -> processFrame(frame) }
        isUIReady = true
    }

    private fun setupParameterPanel() {
        // RANSAC iterations (0-500)
        val seekRansac = findViewById<SeekBar>(R.id.seek_ransac)
        val labelRansac = findViewById<TextView>(R.id.label_ransac)
        seekRansac.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                labelRansac.text = "RANSAC iters: $progress"
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) { pushConfig() }
        })

        // Max depth (0-200m) — pipeline computation parameter
        val seekMaxDepth = findViewById<SeekBar>(R.id.seek_max_depth)
        val labelMaxDepth = findViewById<TextView>(R.id.label_max_depth)
        seekMaxDepth.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                labelMaxDepth.text = "Max depth: ${progress}m"
                maxDepthParam = progress.toFloat().coerceAtLeast(1f)
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) { pushConfig() }
        })

        // Vmax (0-200m) — visualization normalization parameter
        val seekVmax = findViewById<SeekBar>(R.id.seek_vmax)
        val labelVmax = findViewById<TextView>(R.id.label_vmax)
        seekVmax.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                labelVmax.text = "Vmax (viz): ${progress}m"
                vmaxParam = progress.toFloat().coerceAtLeast(1f)
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {
                if (::colorBar.isInitialized) updateColorBar()
            }
        })

        // Lambda forget (0-100 → 0.0-1.0)
        val seekLambda = findViewById<SeekBar>(R.id.seek_lambda)
        val labelLambda = findViewById<TextView>(R.id.label_lambda)
        seekLambda.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                labelLambda.text = "Lambda forget: %.2f".format(progress / 100f)
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) { pushConfig() }
        })

        // Chi2 soft (0-200 → 0.0-20.0)
        val seekChi2 = findViewById<SeekBar>(R.id.seek_chi2)
        val labelChi2 = findViewById<TextView>(R.id.label_chi2)
        seekChi2.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                labelChi2.text = "Chi2 soft: %.1f".format(progress / 10f)
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) { pushConfig() }
        })

        // Variance floor (0-100 → 0.0-0.1)
        val seekVarFloor = findViewById<SeekBar>(R.id.seek_var_floor)
        val labelVarFloor = findViewById<TextView>(R.id.label_var_floor)
        seekVarFloor.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                labelVarFloor.text = "Var floor: %.3f".format(progress / 1000f)
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) { pushConfig() }
        })

        // Min baseline (0-100 → 0.0-0.3m)
        val seekMinBaseline = findViewById<SeekBar>(R.id.seek_min_baseline)
        val labelMinBaseline = findViewById<TextView>(R.id.label_min_baseline)
        seekMinBaseline.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                labelMinBaseline.text = "Min baseline: %.2fm".format(progress / 100f * 0.3f)
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) { pushConfig() }
        })

        // Sky mask threshold (0=off, 1-100 → 1e-8 to 1e-1 log scale)
        val seekSkyThresh = findViewById<SeekBar>(R.id.seek_sky_thresh)
        val labelSkyThresh = findViewById<TextView>(R.id.label_sky_thresh)
        seekSkyThresh.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                if (progress == 0) {
                    labelSkyThresh.text = "Sky mask: OFF"
                } else {
                    val thresh = skyThreshFromProgress(progress)
                    labelSkyThresh.text = "Sky mask: %.1e".format(thresh)
                }
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) { pushConfig() }
        })

        // Toggle switches — push config on change
        findViewById<Switch>(R.id.switch_iterative).setOnCheckedChangeListener { _, _ -> pushConfig() }
        findViewById<Switch>(R.id.switch_timing).setOnCheckedChangeListener { _, _ -> pushConfig() }
        findViewById<Switch>(R.id.switch_show_tri_depth).setOnCheckedChangeListener { _, checked -> showTriDepth = checked }
        findViewById<Switch>(R.id.switch_rgb_color).setOnCheckedChangeListener { _, checked ->
            useRGBColor = checked
            btnToggleRGB.isChecked = checked
        }

        // Browse recordings button
        findViewById<View>(R.id.btn_browse_recordings).setOnClickListener {
            paramsVisible = false
            settingsOverlay.visibility = View.GONE
            showRecordingBrowser()
        }

        // Reset pipeline button
        findViewById<View>(R.id.btn_reset_pipeline).setOnClickListener {
            depthRefinementMgr?.reset()
            prevPose = null
            Toast.makeText(this, "Pipeline reset", Toast.LENGTH_SHORT).show()
        }
    }

    /**
     * Convert sky threshold seekbar progress to actual threshold value.
     * 0 = disabled, 1-100 = log scale from 1e-8 to 1e-1
     */
    private fun skyThreshFromProgress(progress: Int): Float {
        if (progress == 0) return 0f
        // Log scale: progress 1→1e-8, progress 50→~1e-4.5, progress 100→1e-1
        val exp = -8f + (progress - 1) * 7f / 99f  // range: -8 to -1
        return Math.pow(10.0, exp.toDouble()).toFloat()
    }

    /**
     * Read all parameter UI values and push to native pipeline.
     */
    private fun pushConfig() {
        val mgr = depthRefinementMgr ?: return

        // Map UI sliders to PTC-Depth's user-facing tunable params.
        // Note: chi2_soft / fusion_var_floor / fb_consistency / sky-mask threshold
        // are pr_depth-era knobs without a PTC-Depth equivalent — kept in the UI
        // for layout stability but unused at runtime.
        val ransacIters = findViewById<SeekBar>(R.id.seek_ransac).progress
        val maxDepth = findViewById<SeekBar>(R.id.seek_max_depth).progress.toFloat()
        val lambdaForget = findViewById<SeekBar>(R.id.seek_lambda).progress / 100f
        val minBaseline = findViewById<SeekBar>(R.id.seek_min_baseline).progress / 100f * 0.3f
        val iterRefine = if (findViewById<Switch>(R.id.switch_iterative).isChecked) 1 else 0
        val verbose = findViewById<Switch>(R.id.switch_timing).isChecked
        val outdoor = true  // PTC-Depth outdoor flag (sky masking via inverse depth)

        mgr.updateConfig(
            ransacIters = ransacIters,
            maxDepth = maxDepth,
            lambdaForget = lambdaForget,
            minBaseline = minBaseline,
            outdoor = outdoor,
            iterative = iterRefine,
            verbose = verbose,
        )
    }

    // ========================================================================
    // View Mode Management
    // ========================================================================

    private fun updateViewMode() {
        when (viewMode) {
            VIEW_SPLIT -> {
                splitContainer.visibility = View.VISIBLE
                compareContainer.visibility = View.GONE
                cameraView.visibility = View.GONE
                depthVisualizerOverlay.visibility = View.GONE
                pointCloudView.visibility = View.GONE
            }
            VIEW_OVERLAY -> {
                splitContainer.visibility = View.GONE
                compareContainer.visibility = View.GONE
                cameraView.visibility = View.VISIBLE
                depthVisualizerOverlay.visibility = View.VISIBLE
                depthVisualizerOverlay.alpha = 0.7f
                pointCloudView.visibility = View.GONE
            }
            VIEW_DEPTH -> {
                splitContainer.visibility = View.GONE
                compareContainer.visibility = View.GONE
                cameraView.visibility = View.GONE
                depthVisualizerOverlay.visibility = View.VISIBLE
                depthVisualizerOverlay.alpha = 1.0f
                pointCloudView.visibility = View.GONE
            }
            VIEW_COMPARE -> {
                splitContainer.visibility = View.GONE
                compareContainer.visibility = View.VISIBLE
                cameraView.visibility = View.GONE
                depthVisualizerOverlay.visibility = View.GONE
                pointCloudView.visibility = View.GONE
            }
            VIEW_3D -> {
                splitContainer.visibility = View.GONE
                compareContainer.visibility = View.GONE
                cameraView.visibility = View.GONE
                depthVisualizerOverlay.visibility = View.GONE
                pointCloudView.visibility = View.VISIBLE
            }
        }

        // Update tab highlights
        modeTabs.forEachIndexed { index, tab ->
            if (index == viewMode) {
                tab.setTextColor(0xFFFFFFFF.toInt())
                tab.setBackgroundResource(R.drawable.bg_tab_selected)
            } else {
                tab.setTextColor(0xFF666666.toInt())
                tab.background = null
            }
        }

        // RGB switch: only visible in 3D mode (in top HUD)
        val is3D = viewMode == VIEW_3D
        btnToggleRGB.visibility = if (is3D) View.VISIBLE else View.GONE
        labelRGB.visibility = if (is3D) View.VISIBLE else View.GONE
        presetButtons.visibility = if (is3D) View.VISIBLE else View.GONE
        zoomSliderContainer.visibility = if (is3D) View.VISIBLE else View.GONE

        // Colorbar: visible when depth is shown (not 3D, not RGB)
        updateColorBar()

        // PointCloudView lifecycle
        if (viewMode == VIEW_3D) {
            pointCloudView.onResume()
        } else {
            if (::pointCloudView.isInitialized) {
                pointCloudView.onPause()
            }
        }
    }

    private fun updateColorBar() {
        val showColorBar = viewMode == VIEW_DEPTH
        colorBar.visibility = if (showColorBar) View.VISIBLE else View.GONE
        if (showColorBar) {
            colorBar.setMode(useRefinement, vmaxParam)
        }
    }

    // ========================================================================
    // Data Recording
    // ========================================================================

    private fun toggleDataRecording() {
        val logger = dataLogger ?: DataLogger(getExternalFilesDir(null)!!).also { dataLogger = it }
        if (logger.isRecording) {
            val path = logger.stopRecording()
            btnRecordData.text = "\u25CF"  // ●
            btnRecordData.setTextColor(0xFFFF4444.toInt())
            Toast.makeText(this, "Saved: $path", Toast.LENGTH_LONG).show()
        } else {
            logger.startRecording()
            btnRecordData.text = "\u25A0"  // ■
            btnRecordData.setTextColor(0xFFFF0000.toInt())
            Toast.makeText(this, "Recording data...", Toast.LENGTH_SHORT).show()
        }
    }

    // ========================================================================
    // Recording Browser & Playback
    // ========================================================================

    private fun showRecordingBrowser() {
        val player = recordingPlayer ?: RecordingPlayer().also { recordingPlayer = it }
        val recordings = player.listRecordings(getExternalFilesDir(null)!!)

        if (recordings.isEmpty()) {
            Toast.makeText(this, "No recordings found", Toast.LENGTH_SHORT).show()
            return
        }

        val items = recordings.map { rec ->
            val label = rec.displayDate
            val detail = "${rec.totalFrames} frames" + if (rec.isRefined) " (refined)" else " (mono)"
            "$label\n$detail"
        }.toTypedArray()

        AlertDialog.Builder(this, android.R.style.Theme_DeviceDefault_Dialog)
            .setTitle("Recordings")
            .setItems(items) { _, which ->
                val rec = recordings[which]
                startPlaybackMode(rec)
            }
            .setNeutralButton("Delete...") { _, _ ->
                showDeleteRecordingDialog(recordings, player)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun showDeleteRecordingDialog(
        recordings: List<RecordingPlayer.RecordingInfo>,
        player: RecordingPlayer
    ) {
        val items = recordings.map { it.displayDate + " (${it.totalFrames} frames)" }.toTypedArray()
        val checked = BooleanArray(items.size)

        AlertDialog.Builder(this, android.R.style.Theme_DeviceDefault_Dialog)
            .setTitle("Delete Recordings")
            .setMultiChoiceItems(items, checked) { _, which, isChecked ->
                checked[which] = isChecked
            }
            .setPositiveButton("Delete") { _, _ ->
                var count = 0
                for (i in checked.indices) {
                    if (checked[i]) {
                        if (player.deleteRecording(recordings[i])) count++
                    }
                }
                Toast.makeText(this, "Deleted $count recording(s)", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun startPlaybackMode(recording: RecordingPlayer.RecordingInfo) {
        val player = recordingPlayer ?: RecordingPlayer().also { recordingPlayer = it }
        if (!player.load(recording)) {
            Toast.makeText(this, "Failed to load recording", Toast.LENGTH_SHORT).show()
            return
        }

        isPlaybackMode = true

        // Show playback controls (keep current view mode)
        playbackControls.visibility = View.VISIBLE
        seekPlayback.max = (player.totalFrames - 1).coerceAtLeast(0)
        seekPlayback.progress = 0
        textPlaybackFrame.text = "0/${player.totalFrames}"
        btnPlaybackPlay.text = "\u25B6"  // ▶

        // Wire callbacks
        player.onFrame = { frameData ->
            displayPlaybackFrame(frameData)
        }
        player.onFrameChanged = { current, total ->
            seekPlayback.progress = current
            textPlaybackFrame.text = "$current/$total"
        }
        player.onPlaybackFinished = {
            btnPlaybackPlay.text = "\u25B6"  // ▶
        }

        // Show first frame
        player.seekTo(0)

        Toast.makeText(this, "Playback: ${recording.displayDate}", Toast.LENGTH_SHORT).show()
    }

    private fun displayPlaybackFrame(frameData: RecordingPlayer.FrameData) {
        val depth = frameData.depth
        val w = frameData.width
        val h = frameData.height
        val intrinsics = frameData.intrinsics

        // Normalize for display
        val vmax = vmaxParam
        val normalized = FloatArray(depth.size)
        for (i in depth.indices) {
            val d = depth[i]
            normalized[i] = if (d.isFinite() && d > 0f && d <= vmax) d / vmax else Float.NaN
        }

        // Display in current view mode
        when (viewMode) {
            VIEW_SPLIT -> {
                depthVisualizerSplit.updateDepth(normalized, w, h)
                depthVisualizerSplit.setMetricDepth(depth, w, h)
            }
            VIEW_OVERLAY, VIEW_DEPTH -> {
                depthVisualizerOverlay.updateDepth(normalized, w, h)
                depthVisualizerOverlay.setMetricDepth(depth, w, h)
            }
            VIEW_COMPARE -> {
                depthVisualizerCompareModel.updateDepth(normalized, w, h)
                depthVisualizerCompareModel.setMetricDepth(depth, w, h)
            }
            VIEW_3D -> {
                val fx = intrinsics?.fx ?: 400f
                val fy = intrinsics?.fy ?: 400f
                val cx = intrinsics?.cx ?: (w / 2f)
                val cy = intrinsics?.cy ?: (h / 2f)
                pointCloudView.renderer.updatePointCloud(
                    depth, w, h, fx, fy, cx, cy, vmax
                )
            }
        }
    }

    private fun togglePlayback() {
        val player = recordingPlayer ?: return
        if (player.isPlaying) {
            player.pause()
            btnPlaybackPlay.text = "\u25B6"  // ▶
        } else {
            player.play()
            btnPlaybackPlay.text = "\u23F8"  // ⏸
        }
    }

    private fun stopPlaybackMode() {
        recordingPlayer?.stop()
        isPlaybackMode = false
        playbackControls.visibility = View.GONE
        btnPlaybackPlay.text = "\u25B6"  // ▶
    }

    // ========================================================================
    // PTC-Depth Initialization
    // ========================================================================

    /**
     * Initialize PTC-Depth pipeline lazily when refinement is first enabled.
     * Needs camera intrinsics from ARCore, so may defer until first frame.
     */
    private fun initPTCDepthIfNeeded() {
        if (depthRefinementMgr != null) return

        val intrinsics = arCoreManager.getCameraIntrinsics()
        if (intrinsics == null) {
            Log.w(TAG, "Camera intrinsics not yet available, deferring PTC-Depth init")
            return
        }

        // PTC-Depth operates at rotated camera resolution (480x640), NOT 518x518
        Log.d(TAG, "initPTCDepth: rotDeg=$rotDeg, cam=${intrinsics.width}x${intrinsics.height}")
        val ri = DepthRefinementManager.computeRotatedIntrinsics(
            intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
            intrinsics.width, intrinsics.height,
            rotDeg
        )
        modelIntrinsics = ri

        try {
            depthRefinementMgr = DepthRefinementManager(
                ri.fx, ri.fy, ri.cx, ri.cy, ri.width, ri.height, useGTPose
            )
            Log.i(TAG, "PTC-Depth initialized: rotated intrinsics fx=%.1f fy=%.1f cx=%.1f cy=%.1f %dx%d"
                .format(ri.fx, ri.fy, ri.cx, ri.cy, ri.width, ri.height))

            // Push current parameter values
            pushConfig()

            // Enable GT pose switch
            switchGTPose.isEnabled = true
            switchGTPose.alpha = 1.0f
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize PTC-Depth", e)
            Toast.makeText(this, "PTC-Depth init failed: ${e.message}", Toast.LENGTH_LONG).show()
            useRefinement = false
            btnToggleRefine.text = "Mono"
        }
    }

    // ========================================================================
    // Frame Processing
    // ========================================================================

    /**
     * Called on GL thread with a fresh ARCore frame.
     * Acquires camera image and starts depth estimation.
     */
    private fun processFrame(frame: com.google.ar.core.Frame) {
        // Skip live processing during recording playback
        if (isPlaybackMode) return

        // Always acquire camera image for display (even while depth is processing)
        val localCamW: Int
        val localCamH: Int
        val localYRowStride: Int
        val localUvRowStride: Int
        val localUvPixelStride: Int
        val localRotDeg: Int
        try {
            val displayRotation = windowManager.defaultDisplay.rotation
            localRotDeg = (90 - displayRotation * 90 + 360) % 360

            val image = frame.acquireCameraImage()
            localCamW = image.width
            localCamH = image.height

            val yPlane = image.planes[0]
            val uPlane = image.planes[1]
            val vPlane = image.planes[2]
            localYRowStride = yPlane.rowStride
            localUvRowStride = uPlane.rowStride
            localUvPixelStride = uPlane.pixelStride

            val yBuf = yPlane.buffer
            val uBuf = uPlane.buffer
            val vBuf = vPlane.buffer

            // Copy to depth buffers only when depth thread is free
            if (!isProcessing) {
                if (yPlaneData == null || yPlaneData!!.size != yBuf.remaining())
                    yPlaneData = ByteArray(yBuf.remaining())
                if (uPlaneData == null || uPlaneData!!.size != uBuf.remaining())
                    uPlaneData = ByteArray(uBuf.remaining())
                if (vPlaneData == null || vPlaneData!!.size != vBuf.remaining())
                    vPlaneData = ByteArray(vBuf.remaining())
                yBuf.get(yPlaneData!!); yBuf.rewind()
                uBuf.get(uPlaneData!!); uBuf.rewind()
                vBuf.get(vPlaneData!!); vBuf.rewind()
            }

            // Copy to display buffers only when camera views are visible
            val needCamBitmap = viewMode == VIEW_SPLIT || viewMode == VIEW_OVERLAY
            if (needCamBitmap) {
                if (camDispY == null || camDispY!!.size != yBuf.remaining())
                    camDispY = ByteArray(yBuf.remaining())
                if (camDispU == null || camDispU!!.size != uBuf.remaining())
                    camDispU = ByteArray(uBuf.remaining())
                if (camDispV == null || camDispV!!.size != vBuf.remaining())
                    camDispV = ByteArray(vBuf.remaining())
                yBuf.get(camDispY!!); yBuf.rewind()
                uBuf.get(camDispU!!); uBuf.rewind()
                vBuf.get(camDispV!!); vBuf.rewind()
            }
            image.close()

            camW = localCamW
            camH = localCamH
            rotDeg = localRotDeg
        } catch (e: Exception) {
            Log.d(TAG, "processFrame error: ${e.javaClass.simpleName}: ${e.message}")
            return
        }

        // Camera bitmap: only compute when camera views are visible (Split/Overlay)
        if (viewMode == VIEW_SPLIT || viewMode == VIEW_OVERLAY) {
            val camBitmap = yuvToRotatedBitmap(
                camDispY!!, camDispU!!, camDispV!!,
                localCamW, localCamH,
                localYRowStride, localUvRowStride, localUvPixelStride,
                localRotDeg
            )
            runOnUiThread {
                cameraSplit.setImageBitmap(camBitmap)
                cameraView.setImageBitmap(camBitmap)
            }
        }

        // ARCore depth: only fetch in Compare mode (expensive per-pixel rotation loop)
        if (viewMode == VIEW_COMPARE) {
            try {
                val depthImage = frame.acquireDepthImage16Bits()
                val dW = depthImage.width
                val dH = depthImage.height
                val plane = depthImage.planes[0]
                val rowStride = plane.rowStride
                val rawBuf = plane.buffer
                val isRotated = (localRotDeg == 90 || localRotDeg == 270)
                val outW = if (isRotated) dH else dW
                val outH = if (isRotated) dW else dH
                val arcoreDepth = FloatArray(outW * outH)
                for (y in 0 until dH) {
                    for (x in 0 until dW) {
                        val byteOff = y * rowStride + x * 2
                        val lo = rawBuf.get(byteOff).toInt() and 0xFF
                        val hi = rawBuf.get(byteOff + 1).toInt() and 0xFF
                        val mm = lo or (hi shl 8)
                        val meters = mm / 1000f
                        val ox: Int
                        val oy: Int
                        when (localRotDeg) {
                            90 -> { ox = y; oy = dW - 1 - x }
                            180 -> { ox = dW - 1 - x; oy = dH - 1 - y }
                            270 -> { ox = dH - 1 - y; oy = x }
                            else -> { ox = x; oy = y }
                        }
                        arcoreDepth[oy * outW + ox] = if (mm > 0) meters else Float.NaN
                    }
                }
                depthImage.close()
                latestARCoreDepth = arcoreDepth
                arcoreDepthW = outW
                arcoreDepthH = outH
                val normalizedARCore = FloatArray(outW * outH)
                val vmax = vmaxParam
                for (i in arcoreDepth.indices) {
                    val d = arcoreDepth[i]
                    normalizedARCore[i] = if (d.isFinite() && d > 0f && d <= vmax) d / vmax else Float.NaN
                }
                runOnUiThread {
                    depthVisualizerCompareARCore.updateDepth(normalizedARCore, outW, outH)
                    depthVisualizerCompareARCore.setMetricDepth(arcoreDepth, outW, outH)
                }
            } catch (e: com.google.ar.core.exceptions.NotYetAvailableException) {
                // Depth not ready yet
            } catch (e: Exception) {
                Log.d(TAG, "ARCore depth error: ${e.message}")
            }
        }

        if (isProcessing) return

        // Update pose (lightweight, OK on GL thread)
        val currentPose = arCoreManager.getCurrentPose()
        var relativePose: RelativePose? = null
        if (currentPose != null) {
            val prev = prevPose
            if (prev != null) {
                relativePose = arCoreManager.computeRelativePose(prev, currentPose)
            }
            prevPose = currentPose
        }

        // Fast path: QNN combined pipeline
        val qnn = depthEstimatorQNN
        if (qnn != null) {
            isProcessing = true
            val yData = yPlaneData!!
            val uData = uPlaneData!!
            val vData = vPlaneData!!
            val relPose = relativePose
            val capturePose = currentPose
            val modelW = qnn.inputWidth   // 518
            val modelH = qnn.inputHeight  // 518
            depthHandler.post {
                try {
                    if (depthOutput == null || depthOutput!!.size != modelW * modelH)
                        depthOutput = FloatArray(modelW * modelH)
                    val output = depthOutput!!

                    // ======== Pipelined execution: QNN (NPU) || Optical Flow (CPU) ========
                    val mgr = if (useRefinement) depthRefinementMgr else null

                    if (mgr != null) {
                        // Lazy init (needs intrinsics from first frame)
                        if (depthRefinementMgr == null) {
                            runOnUiThread { initPTCDepthIfNeeded() }
                        }

                        // Phase 1: Start QNN on cached thread (NPU) while computing flow (CPU)
                        val t1 = System.currentTimeMillis()
                        val qnnFuture = qnnExecutor.submit {
                            qnn.processFrame(
                                yData, uData, vData,
                                localCamW, localCamH, localYRowStride, localUvRowStride, localUvPixelStride,
                                localRotDeg, output, modelW, modelH
                            )
                        }

                        // Meanwhile: compute grayscale + optical flow on current thread (CPU)
                        mgr.prepareFlow(
                            yData, uData, vData,
                            localCamW, localCamH,
                            localYRowStride, localUvRowStride, localUvPixelStride,
                            localRotDeg
                        )
                        val flowDone = System.currentTimeMillis()

                        // Wait for QNN to complete
                        qnnFuture.get()
                        val qnnDone = System.currentTimeMillis()
                        Log.d(TAG, "Pipeline: flow=${flowDone - t1}ms qnn=${qnnDone - t1}ms (wait=${qnnDone - flowDone}ms)")
                    } else {
                        // No refinement — just run QNN sequentially
                        qnn.processFrame(
                            yData, uData, vData,
                            localCamW, localCamH, localYRowStride, localUvRowStride, localUvPixelStride,
                            localRotDeg, output, modelW, modelH
                        )
                    }

                    // Step 2: PTC-Depth refinement (uses precomputed flow if available)
                    var depthResult: DepthResult? = null
                    if (useRefinement && mgr != null && relPose != null) {
                        val gtR = if (useGTPose) relPose.R else null
                        val gtT = if (useGTPose) relPose.t else null

                        // === SCALE INJECTION POINT (metric baseline / travel distance) ===
                        // `relPose.baseline` is the VIO travel distance between the
                        // previous and current frame (‖ARCore relative translation‖, in
                        // metres). It is the metric scale the pipeline applies to depth.
                        // To drive scale from an EXTERNAL sensor instead of ARCore VIO,
                        // replace `relPose.baseline` below with your sensor's distance for
                        // the SAME prev→curr interval (metres). Rotation/direction stay
                        // from ARCore unless you also override gtR/gtT. See AGENTS.md /
                        // CLAUDE.md "External scale / sensor input".
                        val metricBaseline = relPose.baseline

                        depthResult = mgr.processFrameSync(
                            yData, uData, vData,
                            localCamW, localCamH,
                            localYRowStride, localUvRowStride, localUvPixelStride,
                            localRotDeg,
                            output, modelW, modelH,
                            metricBaseline,
                            gtR, gtT
                        )
                    }

                    // Log frame data if recording
                    if (dataLogger?.isRecording == true) {
                        val logDepth = depthResult?.refinedDepth?.copyOf() ?: output.copyOf()
                        val logW = if (depthResult != null) (modelIntrinsics?.width ?: 480) else modelW
                        val logH = if (depthResult != null) (modelIntrinsics?.height ?: 640) else modelH
                        dataLogger?.logFrame(
                            logDepth, logW, logH,
                            modelIntrinsics, capturePose, relPose,
                            depthResult != null
                        )
                    }

                    // PLY save (snapshot taken on depth thread for consistency).
                    // One-shot via long-press; continuous mode rate-limited to
                    // plyMinIntervalMs so I/O doesn't drown the depth thread.
                    val nowMs = System.currentTimeMillis()
                    val continuousDue = plyContinuousRecording &&
                        (nowMs - lastPlySaveMs >= plyMinIntervalMs)
                    if (plySaveRequested || continuousDue) {
                        val oneShot = plySaveRequested
                        plySaveRequested = false
                        lastPlySaveMs = nowMs
                        savePLYSnapshot(
                            depthResult?.refinedDepth, output, modelW, modelH,
                            yData, uData, vData,
                            localCamW, localCamH,
                            localYRowStride, localUvRowStride, localUvPixelStride,
                            localRotDeg,
                            showToast = oneShot,  // silent during continuous
                        )
                    }

                    // Step 3: Pre-compute 3D RGB pixels on depth thread (avoid blocking UI thread)
                    val rgbPixelsFor3D: IntArray? = if (viewMode == VIEW_3D && useRGBColor) {
                        yuvToARGBRotated(yData, uData, vData, localCamW, localCamH,
                            localYRowStride, localUvRowStride, localUvPixelStride, localRotDeg)
                    } else null

                    // Step 4: Display results
                    val result = depthResult
                    runOnUiThread {
                        if (result != null) {
                            // Refined depth is at rotated camera resolution (e.g., 480x640)
                            val ri = modelIntrinsics
                            val refW = ri?.width ?: 480
                            val refH = ri?.height ?: 640

                            // Choose depth source: raw triangulated or fused
                            val displayDepth = if (showTriDepth) result.triDepth else result.refinedDepth

                            // 2D views
                            if (viewMode != VIEW_3D) {
                                val normalized = normalizeMetricDepth(displayDepth)
                                displayDepth2D(normalized, refW, refH)
                                val activeView = when (viewMode) {
                                    VIEW_SPLIT -> depthVisualizerSplit
                                    VIEW_COMPARE -> depthVisualizerCompareModel
                                    else -> depthVisualizerOverlay
                                }
                                activeView.setMetricDepth(displayDepth, refW, refH)
                            }

                            // 3D view (RGB pixels already computed on depth thread)
                            if (viewMode == VIEW_3D && ri != null) {
                                pointCloudView.renderer.useRGBColor = useRGBColor
                                pointCloudView.renderer.updatePointCloud(
                                    displayDepth, refW, refH,
                                    ri.fx, ri.fy, ri.cx, ri.cy,
                                    vmaxParam, rgbPixelsFor3D
                                )
                            }

                            // Update stats
                            textViewBaseline.text = "B: %.3fm".format(result.baseline)
                            textViewRotation.text = "R: %.1f°".format(result.rotationAngleDeg)
                            textViewMatches.text = "M:${result.numMatches}/V:${result.numValidTri}" +
                                if (result.usedGTPose) " [GT]" else ""

                            // Metrics: only compute when HUD is visible (expensive)
                            if (textViewAbsRel.visibility == View.VISIBLE) {
                                val arDepth = latestARCoreDepth
                                val arW = arcoreDepthW; val arH = arcoreDepthH
                                if (arDepth != null && arW > 0 && arH > 0) {
                                    val metrics = computeDepthMetricsResampled(
                                        result.refinedDepth, refW, refH, arDepth, arW, arH)
                                    if (metrics != null) {
                                        textViewAbsRel.text = "AbsRel: %.3f | δ<1.25: %.1f%%".format(metrics.first, metrics.second * 100f)
                                    }
                                }

                                val prevD = prevRefinedDepth
                                val K = modelIntrinsics
                                if (prevD != null && K != null && prevDepthW == refW && prevDepthH == refH
                                    && result.baseline > 1e-4f) {
                                    val tae = computeTAE(prevD, result.refinedDepth, refW, refH,
                                        result.R, result.t, result.baseline, K)
                                    if (tae.isFinite()) {
                                        textViewAbsRel.text = (textViewAbsRel.text.toString().split("|").firstOrNull()?.trim() ?: "") + " | TAE:%.4f".format(tae)
                                    }
                                }
                            }
                            prevRefinedDepth = result.refinedDepth
                            prevDepthW = refW
                            prevDepthH = refH
                        } else {
                            // Show mono depth — resize from model (518x518) to rotated camera resolution
                            val isRotated = (localRotDeg == 90 || localRotDeg == 270)
                            val dispW = if (localCamW > 0) (if (isRotated) localCamH else localCamW) else modelW
                            val dispH = if (localCamH > 0) (if (isRotated) localCamW else localCamH) else modelH
                            val monoDisp = if (dispW != modelW || dispH != modelH)
                                resizeFloatBilinear(output, modelW, modelH, dispW, dispH)
                            else output
                            if (viewMode != VIEW_3D) {
                                displayDepth2D(monoDisp, dispW, dispH)
                            } else if (!useRefinement) {
                                // 3D with mono depth (RGB pixels already computed on depth thread)
                                val mi = modelIntrinsics
                                pointCloudView.renderer.useRGBColor = useRGBColor
                                if (useRGBColor && rgbPixelsFor3D != null) {
                                    pointCloudView.renderer.updatePointCloud(
                                        monoDisp, dispW, dispH,
                                        mi?.fx ?: 400f, mi?.fy ?: 400f,
                                        mi?.cx ?: (dispW / 2f), mi?.cy ?: (dispH / 2f),
                                        vmaxParam, rgbPixelsFor3D
                                    )
                                } else {
                                    pointCloudView.renderer.updatePointCloud(
                                        output, modelW, modelH,
                                        mi?.fx ?: 400f, mi?.fy ?: 400f,
                                        mi?.cx ?: (modelW / 2f), mi?.cy ?: (modelH / 2f),
                                        vmaxParam
                                    )
                                }
                            }
                            // else: refinement active but no result → keep previous 3D point cloud
                            if (relPose != null) {
                                textViewBaseline.text = "B: %.3fm".format(relPose.baseline)
                            }
                            if (!useRefinement) {
                                textViewMatches.text = "Mono"
                                textViewRotation.text = "R: --"
                            }
                        }
                        updateFPS()
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Depth processing failed", e)
                } finally {
                    isProcessing = false
                }
            }
            return
        }

        // Fallback path: ONNX Runtime (no longer supported without getCameraImage)
        Log.w(TAG, "ONNX fallback path not available — QNN required")
    }

    /**
     * Display [0,1] normalized depth in 2D visualizers (Split or Overlay).
     */
    private fun displayDepth2D(depth: FloatArray, width: Int, height: Int) {
        when (viewMode) {
            VIEW_SPLIT -> depthVisualizerSplit.updateDepth(depth, width, height)
            VIEW_OVERLAY, VIEW_DEPTH -> depthVisualizerOverlay.updateDepth(depth, width, height)
            VIEW_COMPARE -> depthVisualizerCompareModel.updateDepth(depth, width, height)
        }
    }

    private fun displayRGB2D(pixels: IntArray, width: Int, height: Int) {
        when (viewMode) {
            VIEW_SPLIT -> depthVisualizerSplit.updatePixels(pixels, width, height)
            VIEW_OVERLAY, VIEW_DEPTH -> depthVisualizerOverlay.updatePixels(pixels, width, height)
            VIEW_COMPARE -> depthVisualizerCompareModel.updatePixels(pixels, width, height)
        }
    }

    /**
     * Convert metric depth (meters) to [0,1] for colormap display.
     * Inverted: close (small Z) = 1.0 (warm), far (large Z) = 0.0 (cool).
     * Matches the QNN inverse depth convention.
     */
    private fun normalizeMetricDepth(depth: FloatArray): FloatArray {
        val size = depth.size
        if (normalizedBuf == null || normalizedBuf!!.size != size) {
            normalizedBuf = FloatArray(size)
        }
        val out = normalizedBuf!!

        // Use fixed range: [0, vmaxParam] (from Vmax slider, separate from pipeline max_depth)
        val vmax = vmaxParam
        for (i in 0 until size) {
            val d = depth[i]
            if (d.isFinite() && d > 0f) {
                out[i] = if (d <= vmax) d / vmax else Float.NaN  // >vmax → black
            } else {
                out[i] = Float.NaN
            }
        }
        return out
    }

    /**
     * Compute AbsRel and δ<1.25 between predicted and GT depth at different resolutions.
     * Iterates over GT (ARCore) pixels, bilinear-samples predicted depth at corresponding location.
     */
    private fun computeDepthMetricsResampled(
        pred: FloatArray, predW: Int, predH: Int,
        gt: FloatArray, gtW: Int, gtH: Int
    ): Pair<Float, Float>? {
        var absRelSum = 0.0
        var delta125Count = 0
        var validCount = 0
        val step = 2  // Subsample GT pixels
        val scaleX = (predW - 1).toFloat() / (gtW - 1).coerceAtLeast(1)
        val scaleY = (predH - 1).toFloat() / (gtH - 1).coerceAtLeast(1)
        for (gy in 0 until gtH step step) {
            for (gx in 0 until gtW step step) {
                val g = gt[gy * gtW + gx]
                if (!g.isFinite() || g <= 0.1f) continue
                // Map to pred coordinates
                val px = gx * scaleX
                val py = gy * scaleY
                val pxi = px.toInt().coerceAtMost(predW - 2)
                val pyi = py.toInt().coerceAtMost(predH - 2)
                val dx = px - pxi; val dy = py - pyi
                val d00 = pred[pyi * predW + pxi]
                val d10 = pred[pyi * predW + pxi + 1]
                val d01 = pred[(pyi + 1) * predW + pxi]
                val d11 = pred[(pyi + 1) * predW + pxi + 1]
                if (!d00.isFinite() || !d10.isFinite() || !d01.isFinite() || !d11.isFinite()) continue
                val p = (1 - dx) * (1 - dy) * d00 + dx * (1 - dy) * d10 +
                    (1 - dx) * dy * d01 + dx * dy * d11
                if (p <= 0.1f) continue
                absRelSum += Math.abs(p - g).toDouble() / g
                val ratio = if (p > g) p / g else g / p
                if (ratio < 1.25f) delta125Count++
                validCount++
            }
        }
        if (validCount < 50) return null
        return Pair((absRelSum / validCount).toFloat(), delta125Count.toFloat() / validCount)
    }

    /**
     * TAE: warp previous depth to current frame using R,t,baseline,K.
     * Backward warp: for each pixel in curr, find corresponding prev pixel.
     * Returns AbsRel between expected prev depth and actual prev depth.
     */
    private fun computeTAE(
        prevDepth: FloatArray, currDepth: FloatArray,
        w: Int, h: Int,
        R: FloatArray, t: FloatArray, baseline: Float,
        K: CameraIntrinsics
    ): Float {
        val fx = K.fx.toDouble(); val fy = K.fy.toDouble()
        val cx = K.cx.toDouble(); val cy = K.cy.toDouble()
        val ifx = 1.0 / fx; val ify = 1.0 / fy
        // R is row-major 3x3, t is unit vector scaled by baseline
        val r00 = R[0].toDouble(); val r01 = R[1].toDouble(); val r02 = R[2].toDouble()
        val r10 = R[3].toDouble(); val r11 = R[4].toDouble(); val r12 = R[5].toDouble()
        val r20 = R[6].toDouble(); val r21 = R[7].toDouble(); val r22 = R[8].toDouble()
        val tx = t[0].toDouble() * baseline
        val ty = t[1].toDouble() * baseline
        val tz = t[2].toDouble() * baseline
        // R^T for inverse transform: P_prev = R^T * (P_curr - t*baseline)
        val rt00 = r00; val rt01 = r10; val rt02 = r20
        val rt10 = r01; val rt11 = r11; val rt12 = r21
        val rt20 = r02; val rt21 = r12; val rt22 = r22

        var absRelSum = 0.0
        var count = 0
        val step = 4  // Subsample
        for (v in 0 until h step step) {
            for (u in 0 until w step step) {
                val dCurr = currDepth[v * w + u].toDouble()
                if (!dCurr.isFinite() || dCurr <= 0.1) continue
                // Unproject curr pixel to 3D
                val xc = (u - cx) * ifx * dCurr
                val yc = (v - cy) * ify * dCurr
                val zc = dCurr
                // Transform to prev frame: P_prev = R^T * (P_curr - t*baseline)
                val dx = xc - tx; val dy = yc - ty; val dz = zc - tz
                val xp = rt00 * dx + rt01 * dy + rt02 * dz
                val yp = rt10 * dx + rt11 * dy + rt12 * dz
                val zp = rt20 * dx + rt21 * dy + rt22 * dz
                if (zp <= 0.1) continue
                // Project to prev image
                val up = fx * xp / zp + cx
                val vp = fy * yp / zp + cy
                val ui = up.toInt(); val vi = vp.toInt()
                if (ui < 0 || ui >= w - 1 || vi < 0 || vi >= h - 1) continue
                // Bilinear sample prev depth
                val du = up - ui; val dv = vp - vi
                val d00 = prevDepth[vi * w + ui].toDouble()
                val d10 = prevDepth[vi * w + ui + 1].toDouble()
                val d01 = prevDepth[(vi + 1) * w + ui].toDouble()
                val d11 = prevDepth[(vi + 1) * w + ui + 1].toDouble()
                if (!d00.isFinite() || !d10.isFinite() || !d01.isFinite() || !d11.isFinite()) continue
                if (d00 <= 0.1 || d10 <= 0.1 || d01 <= 0.1 || d11 <= 0.1) continue
                val dPrevSampled = (1 - du) * (1 - dv) * d00 + du * (1 - dv) * d10 +
                    (1 - du) * dv * d01 + du * dv * d11
                // AbsRel: |expected_prev_z - sampled_prev_depth| / sampled_prev_depth
                absRelSum += Math.abs(zp - dPrevSampled) / dPrevSampled
                count++
            }
        }
        if (count < 100) return Float.NaN
        return (absRelSum / count).toFloat()
    }

    /**
     * Bilinear resize a float array from (srcW, srcH) to (dstW, dstH).
     */
    private fun resizeFloatBilinear(
        src: FloatArray, srcW: Int, srcH: Int, dstW: Int, dstH: Int
    ): FloatArray {
        val size = dstW * dstH
        if (resizedMonoBuf == null || resizedMonoBuf!!.size != size) {
            resizedMonoBuf = FloatArray(size)
        }
        val dst = resizedMonoBuf!!
        val scaleX = srcW.toFloat() / dstW
        val scaleY = srcH.toFloat() / dstH
        for (dy in 0 until dstH) {
            val sy = dy * scaleY
            val y0 = sy.toInt().coerceAtMost(srcH - 2)
            val fy = sy - y0
            for (dx in 0 until dstW) {
                val sx = dx * scaleX
                val x0 = sx.toInt().coerceAtMost(srcW - 2)
                val fx = sx - x0
                val v00 = src[y0 * srcW + x0]
                val v10 = src[y0 * srcW + x0 + 1]
                val v01 = src[(y0 + 1) * srcW + x0]
                val v11 = src[(y0 + 1) * srcW + x0 + 1]
                dst[dy * dstW + dx] = v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) +
                    v01 * (1 - fx) * fy + v11 * fx * fy
            }
        }
        return dst
    }

    private fun updateFPS() {
        val currentTime = System.currentTimeMillis()
        val elapsed = currentTime - lastFrameTime
        if (elapsed > 0) {
            val instantFps = 1000.0 / elapsed
            fpsEma = if (fpsEma <= 0.0) instantFps else fpsEma * 0.8 + instantFps * 0.2
            textViewFPS.text = "FPS: %.1f".format(fpsEma)
        }
        lastFrameTime = currentTime
    }

    /**
     * Snapshot current depth + YUV + intrinsics as a raw .ptcf blob.
     * No projection / no YUV→RGB on device — that runs offline on PC (ptcf_to_ply.py),
     * so the depth-thread save cost is just memory + file I/O (~10ms).
     */
    private fun savePLYSnapshot(
        refinedDepth: FloatArray?, monoDepth: FloatArray, modelW: Int, modelH: Int,
        yData: ByteArray, uData: ByteArray, vData: ByteArray,
        imgW: Int, imgH: Int,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int,
        rotationDegrees: Int,
        showToast: Boolean = true,
    ) {
        val ri = modelIntrinsics
        val isRotated = (rotationDegrees == 90 || rotationDegrees == 270)
        val outW = ri?.width ?: (if (isRotated) imgH else imgW)
        val outH = ri?.height ?: (if (isRotated) imgW else imgH)
        val depth = refinedDepth
            ?: if (modelW == outW && modelH == outH) monoDepth
               else resizeFloatBilinear(monoDepth, modelW, modelH, outW, outH)
        val fx = ri?.fx ?: 400f
        val fy = ri?.fy ?: 400f
        val cx = ri?.cx ?: (outW / 2f)
        val cy = ri?.cy ?: (outH / 2f)

        val dir = File(getExternalFilesDir(null), "captures")
        val outFile = File(dir, "cap_${System.currentTimeMillis()}.ptcf")
        try {
            val bytes = RawCaptureWriter.save(
                depth, outW, outH,
                yData, uData, vData,
                imgW, imgH, yRowStride, uvRowStride, uvPixelStride,
                rotationDegrees,
                fx, fy, cx, cy,
                outFile,
            )
            if (showToast) {
                runOnUiThread {
                    Toast.makeText(
                        this,
                        "저장됨: ${outFile.name} (${bytes / 1024} KB)",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "raw capture save failed", e)
            if (showToast) {
                runOnUiThread {
                    Toast.makeText(this, "저장 실패: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private var rgbBuf: IntArray? = null

    /**
     * Convert YUV420 planes to rotated ARGB IntArray (no Bitmap).
     * For use on depth thread to provide RGB pixels to 3D point cloud.
     */
    private fun yuvToARGBRotated(
        yData: ByteArray, uData: ByteArray, vData: ByteArray,
        imgW: Int, imgH: Int,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int,
        rotDeg: Int
    ): IntArray {
        val isRotated = (rotDeg == 90 || rotDeg == 270)
        val outW = if (isRotated) imgH else imgW
        val outH = if (isRotated) imgW else imgH
        val pixels = IntArray(outW * outH)

        for (oy in 0 until outH) {
            for (ox in 0 until outW) {
                val sx: Int
                val sy: Int
                when (rotDeg) {
                    90  -> { sx = oy; sy = imgH - 1 - ox }
                    270 -> { sx = imgW - 1 - oy; sy = ox }
                    180 -> { sx = imgW - 1 - ox; sy = imgH - 1 - oy }
                    else -> { sx = ox; sy = oy }
                }
                val Y = yData[sy * yRowStride + sx].toInt() and 0xFF
                val uvIdx = (sy / 2) * uvRowStride + (sx / 2) * uvPixelStride
                val U = (uData[uvIdx].toInt() and 0xFF) - 128
                val V = (vData[uvIdx].toInt() and 0xFF) - 128
                var r = Y + (1.370705f * V).toInt()
                var g = Y - (0.337633f * U).toInt() - (0.698001f * V).toInt()
                var b = Y + (1.732446f * U).toInt()
                pixels[oy * outW + ox] = (0xFF shl 24) or
                    (r.coerceIn(0, 255) shl 16) or
                    (g.coerceIn(0, 255) shl 8) or
                    b.coerceIn(0, 255)
            }
        }
        return pixels
    }

    /**
     * Convert YUV420 planes to packed ARGB IntArray.
     */
    private fun yuvToARGB(
        yData: ByteArray, uData: ByteArray, vData: ByteArray,
        width: Int, height: Int,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int
    ): IntArray {
        val size = width * height
        if (rgbBuf == null || rgbBuf!!.size != size) rgbBuf = IntArray(size)
        val out = rgbBuf!!
        for (row in 0 until height) {
            for (col in 0 until width) {
                val yVal = (yData[row * yRowStride + col].toInt() and 0xFF)
                val uvRow = row / 2
                val uvCol = col / 2
                val uVal = (uData[uvRow * uvRowStride + uvCol * uvPixelStride].toInt() and 0xFF) - 128
                val vVal = (vData[uvRow * uvRowStride + uvCol * uvPixelStride].toInt() and 0xFF) - 128
                var r = (yVal + 1.370705f * vVal).toInt()
                var g = (yVal - 0.337633f * uVal - 0.698001f * vVal).toInt()
                var b = (yVal + 1.732446f * uVal).toInt()
                r = r.coerceIn(0, 255)
                g = g.coerceIn(0, 255)
                b = b.coerceIn(0, 255)
                out[row * width + col] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        return out
    }

    // ========================================================================
    // Lifecycle
    // ========================================================================

    override fun onResume() {
        super.onResume()
        // Resume ARCore session first (must happen before GLSurfaceView resumes)
        if (::arCoreManager.isInitialized) {
            try {
                arCoreManager.resume()
            } catch (e: Exception) {
                Log.e(TAG, "ARCore resume failed", e)
            }
        }
        if (::glSurfaceView.isInitialized) {
            glSurfaceView.onResume()
        }
        if (::pointCloudView.isInitialized && pointCloudView.visibility == View.VISIBLE) {
            pointCloudView.onResume()
        }
        // Restore HTP TURBO clocks
        depthEstimatorQNN?.resumePerf()
        // Re-attach frame callback only if setupUI() completed
        if (isUIReady) {
            arCoreRenderer.onNewFrame = { frame -> processFrame(frame) }
        }
    }

    override fun onPause() {
        super.onPause()
        // Stop accepting new frames FIRST
        if (::arCoreRenderer.isInitialized) {
            arCoreRenderer.onNewFrame = null
        }
        // Stop playback on pause
        recordingPlayer?.pause()
        // Stop recordings on pause
        if (dataLogger?.isRecording == true) {
            dataLogger?.stopRecording()
            if (::btnRecordData.isInitialized) {
                btnRecordData.text = "\u25CF"
                btnRecordData.setTextColor(0xFFFF4444.toInt())
            }
        }
        // Cancel any pending depth tasks
        if (::depthHandler.isInitialized) {
            depthHandler.removeCallbacksAndMessages(null)
        }
        isProcessing = false
        // Release HTP TURBO clocks
        depthEstimatorQNN?.pausePerf()
        if (::glSurfaceView.isInitialized) {
            glSurfaceView.onPause()
        }
        if (::pointCloudView.isInitialized) {
            pointCloudView.onPause()
        }
        if (::arCoreManager.isInitialized) {
            arCoreManager.pause()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        recordingPlayer?.release()
        qnnExecutor.shutdown()
        if (::depthThread.isInitialized) {
            depthThread.quitSafely()
        }
        depthEstimatorQNN?.close()
        depthEstimator?.close()
        depthRefinementMgr?.destroy()
        if (::arCoreManager.isInitialized) {
            arCoreManager.destroy()
        }
    }

    /**
     * Convert YUV camera image to a rotated ARGB bitmap.
     * Uses the same rotation mapping as the depth model (qnn_jni_bridge.cpp).
     */
    private fun yuvToRotatedBitmap(
        yData: ByteArray, uData: ByteArray, vData: ByteArray,
        imgW: Int, imgH: Int,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int,
        rotDeg: Int
    ): Bitmap {
        val isRotated = (rotDeg == 90 || rotDeg == 270)
        val outW = if (isRotated) imgH else imgW
        val outH = if (isRotated) imgW else imgH

        if (camDispPixels == null || camDispPixels!!.size != outW * outH)
            camDispPixels = IntArray(outW * outH)
        val pixels = camDispPixels!!

        for (oy in 0 until outH) {
            for (ox in 0 until outW) {
                val sx: Int
                val sy: Int
                when (rotDeg) {
                    90  -> { sx = oy; sy = imgH - 1 - ox }
                    270 -> { sx = imgW - 1 - oy; sy = ox }
                    180 -> { sx = imgW - 1 - ox; sy = imgH - 1 - oy }
                    else -> { sx = ox; sy = oy }
                }

                val yIdx = sy * yRowStride + sx
                val uvIdx = (sy / 2) * uvRowStride + (sx / 2) * uvPixelStride

                val Y = yData[yIdx].toInt() and 0xFF
                val U = (uData[uvIdx].toInt() and 0xFF) - 128
                val V = (vData[uvIdx].toInt() and 0xFF) - 128

                var r = Y + (1.370705f * V).toInt()
                var g = Y - (0.337633f * U).toInt() - (0.698001f * V).toInt()
                var b = Y + (1.732446f * U).toInt()

                pixels[oy * outW + ox] = (0xFF shl 24) or
                    (r.coerceIn(0, 255) shl 16) or
                    (g.coerceIn(0, 255) shl 8) or
                    b.coerceIn(0, 255)
            }
        }

        var bmp = camDispBitmap
        if (bmp == null || bmp.width != outW || bmp.height != outH) {
            bmp = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
            camDispBitmap = bmp
        }
        bmp.setPixels(pixels, 0, outW, 0, 0, outW, outH)
        return bmp
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val PERMISSION_REQUEST_CODE = 100
        private const val VIEW_SPLIT = 0
        private const val VIEW_OVERLAY = 1
        private const val VIEW_DEPTH = 2
        private const val VIEW_COMPARE = 3
        private const val VIEW_3D = 4
    }
}
