# PTC-Depth — CLAUDE.md

Real-time metric depth estimation on mobile using monocular depth + multi-frame
fusion. This repo is an **Android-app-focused handoff**: the Python ML pipeline,
model weights, and phone captures were moved to `archive/` (git-ignored, out of
scope). Build/install details: [`android/BUILD.md`](android/BUILD.md). Same facts
for the Codex agent: [`AGENTS.md`](AGENTS.md).

## Architecture

### Android App (`android/`)
- **Language**: Kotlin + C++ (JNI). Package: `com.ptcdepth.android`.
- **Depth model**: Depth Anything V2 Small (518x518, float16)
- **Inference**: ONNX Runtime + QNN HTP Execution Provider → Hexagon DSP. The
  ONNX graph is compiled to a QNN context binary on first launch and cached.
  CPU fallback via ONNX Runtime if QNN init fails.
- **Pose**: ARCore (camera pose tracking; depth API for GT comparison)
- **Pipeline**: C++ pipeline (optical flow → RANSAC → triangulation → Kalman/
  Bayesian fusion)

### Key Files
- `android/app/src/main/java/com/ptcdepth/android/MainActivity.kt` — main controller, frame loop, 5 view modes
- `android/app/src/main/java/com/ptcdepth/android/ARCoreManager.kt` — ARCore session, pose extraction, relative pose
- `android/app/src/main/java/com/ptcdepth/android/DepthEstimatorQNN.kt` — ORT + QNN HTP inference; **per-SoC soc_model/htp_arch mapping**
- `android/app/src/main/java/com/ptcdepth/android/DepthEstimator.kt` — ONNX Runtime CPU fallback
- `android/app/src/main/java/com/ptcdepth/android/DepthRefinementManager.kt` — JNI bridge to the C++ pipeline
- `android/app/src/main/cpp/jni_bridge.cpp` — JNI bridge (image rotation, pose conversion, pipeline call)
- `android/app/src/main/cpp/qnn_jni_bridge.cpp` — YUV→NCHW preprocessing helpers for the ORT-QNN path
- `android/app/src/main/cpp/CMakeLists.txt` — native build; references `../../../../../cpp`
- `cpp/src/motion_field.cpp` — motion field estimation, epipolar geometry (R = R_cam^T)
- `cpp/src/{triangulation,scale_estimation,depth_warp,bayesian_fusion,ptc_depth}.cpp` — pipeline core

### C++ Pipeline (`cpp/`) — REQUIRED to build the APK
- Uses OpenCV (optical flow, image processing) and Eigen (linear algebra)
- The app compiles these sources directly via CMake (`cpp/src`, `cpp/include`).
- Also buildable standalone for desktop testing (see `cpp/README_BUILD.md`).

## Build toolchain (see android/BUILD.md for the full guide)
- compileSdk 34, minSdk 26, AGP 8.2.0, Gradle 8.2 (wrapper)
- **NDK 27.2.12479018 (r27c)** — required for 16 KB page alignment; pinned via
  `ndkVersion` in `app/build.gradle.kts`
- CMake 3.22.1
- ORT-QNN via Gradle: `com.microsoft.onnxruntime:onnxruntime-android-qnn:1.24.3`
- ARCore via Gradle: `com.google.ar:core:1.48.0` (1.48+ = 16 KB-aligned libs)
- Build: `cd android && ./gradlew :app:assembleDebug` (clean build after asset/native changes)
- Install: `adb install -r --no-streaming <apk>` (large APK; streaming can hang)

## Coordinate Conventions (CRITICAL)

### Pipeline expects "point transform" convention
```
P_curr = R * P_prev + baseline * t
```
Where R = R_cam^T (see `cpp/src/motion_field.cpp`, ~line 1403)

### ARCore → Pipeline conversion
1. **OpenGL to CV**: `C = diag(1, -1, -1)`, `R_cv = C * R_gl * C`, `t_cv = C * t_gl`
2. **Relative pose direction**: `currPose.inverse().compose(prevPose)` gives T_curr←prev (correct)
3. **Image rotation (90° CW for portrait)**: Apply `C_rot = [[0,-1,0],[1,0,0],[0,0,1]]` to R,t in jni_bridge.cpp

### Camera intrinsics
- ARCore `imageIntrinsics`: always landscape (640x480)
- After 90° rotation for portrait: swap fx↔fy, cx↔cy, w↔h

## QNN / NPU notes (CRITICAL)
- **Use the HTP backend**, not GPU. `libQnnGpu.so` does not work on Android apps
  (it dlopens `/vendor/lib64/libOpenCL.so`, blocked by linker namespace isolation).
- Manifest must keep `extractNativeLibs="true"` and
  `<uses-native-library android:name="libcdsprpc.so"/>`.
- **Per-SoC mapping** in `DepthEstimatorQNN.kt` (`Build.SOC_MODEL` → soc_model/htp_arch):
  SM8450→36/V69 (S22), SM8550→43/V73, SM8650→57/V75, SM8750→69/V79 (S25),
  SM8850→87/V81 (SM-S948N). Values from QAIRT 2.42 `QnnTypes.h` / `QnnHtpDevice.h`.
  Ship the matching `libQnnHtpV<arch>Skel.so` + `Stub.so` in `app/libs/arm64-v8a/`.
- Context-binary cache is keyed by arch: `cache/depth_anything_qnn_ctx_v<arch>.bin`.

## 16 KB page alignment (Android 15/16, e.g. SM8850)
All bundled **AArch64** `.so` must have 16 KB-aligned LOAD segments or the OS
shows an "ELF alignment check failed" warning. Hexagon skels don't count. Kept
via: NDK r27c, `-Wl,-z,max-page-size=16384` in `cpp/CMakeLists.txt`, ARCore 1.48,
and removing the unused CameraX dependency. Do not downgrade the NDK or ARCore.

## Extending: external scale / sensor input
The pipeline's metric scale ("이동거리" / travel distance) comes from ARCore VIO as
a single scalar `baseline` (metres): `ARCoreManager.computeRelativePose()` (baseline
= ‖relative translation‖) → `MainActivity.processFrame` (`relPose.baseline`) →
`DepthRefinementManager.processFrameSync(..., baseline)` → JNI → C++
`pipeline.refine(..., baseline, ...)`. Convention: `P_curr = R·P_prev + baseline·t`
(unit `t`, `baseline` = its metre scale).
- **Injection point**: `MainActivity.kt`, grep `SCALE INJECTION POINT`. Replace
  `metricBaseline` with an external sensor's distance for the same prev→curr interval.
- Still needed: a sensor data source (none exists yet — only ARCore feeds the app),
  timestamp sync to the frame pair, metres, and — only if replacing direction too —
  the CV convention + portrait `C_rot`. Scalar-only swap avoids the convention work.

## iOS Porting Notes (reference, not in this handoff)
- Core ML: `apple/coreml-depth-anything-v2-small` (HF). Float16: 49.8MB, input 518x396. ~60-80ms on A12Z.
- ARKit: `ARFrame.camera.transform` ≈ ARCore pose; `.intrinsics` ≈ `imageIntrinsics`;
  `.sceneDepth.depthMap` ≈ `acquireDepthImage16Bits()` (Float32 m vs uint16 mm);
  `.capturedImage` ≈ `acquireCameraImage()` (NV12 CVPixelBuffer).
- Reuse C++ via Objective-C++ (replace JNI); OpenCV via CocoaPods/SPM; Eigen header-only.
- UIKit for Metal/AR; point cloud OpenGL ES → Metal; depth viz custom UIView.

## Dependencies — committed for a self-contained `git clone` (see android/BUILD.md §2)
- **OpenCV 4.8.0** (trimmed to arm64-v8a) in `android/opencv-android-sdk/`
- **Eigen 3.4.0** in `android/app/libs/eigen3/`
- **QNN HTP libs** in `android/app/libs/arm64-v8a/` (from QAIRT 2.42)
- **Model files** in `android/app/src/main/assets/`: `depth_anything_v2.onnx`+`.data` (QNN),
  `depth_anything.onnx` (CPU fallback)
- **ARCore + ONNX Runtime QNN**: Gradle dependencies
- Only the Android SDK/NDK (r27c, cmake 3.22.1) must be installed by the builder.

## Archived (git-ignored, `archive/`)
Python ML training/eval/conversion (`python-ml/`), model weights (`weights/`, 6 GB),
phone captures (`out/`, 2.8 GB), conversion scripts, and unused model-format assets.
See `archive/README.md`. Do not pull these back into the app build.
