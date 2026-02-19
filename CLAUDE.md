# PR-Depth

Real-time metric depth estimation on mobile using monocular depth + multi-frame fusion.

## Architecture

### Android App (`android/`)
- **Language**: Kotlin + C++ (JNI)
- **Depth model**: Depth Anything V2 Small (518x518, float16)
- **Inference**: QNN HTP (Hexagon DSP) via `libQnnHtp.so` — ~65-80ms/frame on Snapdragon 8 Gen 1
- **Pose**: ARCore (camera pose tracking, depth API for GT comparison)
- **Pipeline**: PR-Depth C++ pipeline (optical flow → RANSAC → triangulation → Kalman fusion)

### Key Files
- `android/app/src/main/java/com/prdepth/android/MainActivity.kt` — Main controller, frame processing loop, 5 view modes
- `android/app/src/main/java/com/prdepth/android/ARCoreManager.kt` — ARCore session, pose extraction, relative pose computation
- `android/app/src/main/java/com/prdepth/android/DepthEstimatorQNN.kt` — QNN HTP inference wrapper
- `android/app/src/main/java/com/prdepth/android/DepthRefinementManager.kt` — JNI bridge to C++ PR-Depth pipeline
- `android/app/src/main/cpp/jni_bridge.cpp` — JNI native bridge (image rotation, pose conversion, pipeline call)
- `android/app/src/main/cpp/depth_refinement.cpp` — Core PR-Depth algorithm (flow, RANSAC, triangulation, fusion)
- `android/app/src/main/cpp/motion_field.cpp` — Motion field estimation, epipolar geometry

### C++ Pipeline (`cpp/`)
- Uses OpenCV (optical flow, image processing) and Eigen (linear algebra)
- Compilable for both desktop (testing) and Android (via NDK/CMake)

### Python Tools (`src/`, `tools/`, `model_conversion/`)
- Model training, evaluation, conversion scripts
- ONNX/DLC export for mobile deployment

## Coordinate Conventions (CRITICAL)

### Pipeline expects "point transform" convention
```
P_curr = R * P_prev + baseline * t
```
Where R = R_cam^T (see motion_field.cpp:1403)

### ARCore → Pipeline conversion
1. **OpenGL to CV**: `C = diag(1, -1, -1)`, `R_cv = C * R_gl * C`, `t_cv = C * t_gl`
2. **Relative pose direction**: `currPose.inverse().compose(prevPose)` gives T_curr←prev (correct)
3. **Image rotation (90° CW for portrait)**: Apply `C_rot = [[0,-1,0],[1,0,0],[0,0,1]]` to R,t in jni_bridge.cpp

### Camera intrinsics
- ARCore `imageIntrinsics`: always landscape (640x480)
- After 90° rotation for portrait: swap fx↔fy, cx↔cy, w↔h

## iOS Porting Notes

### Inference
- Core ML model: `apple/coreml-depth-anything-v2-small` on HuggingFace (Apple official)
- Float16 variant: 49.8MB, input 518x396
- Expected ~60-80ms on A12Z Neural Engine (iPad Pro 4th gen)

### ARKit equivalents
- `ARFrame.camera.transform` = ARCore `camera.pose` (same OpenGL Y-up convention)
- `ARFrame.camera.intrinsics` = `camera.imageIntrinsics` (simd_float3x3)
- `ARFrame.sceneDepth.depthMap` = `acquireDepthImage16Bits()` (but Float32 meters, not uint16 mm)
- `ARFrame.capturedImage` = `acquireCameraImage()` (CVPixelBuffer NV12, not separate YUV planes)

### C++ pipeline reuse
- Keep C++ code, use Objective-C++ bridging (replace JNI with ObjC++ wrapper)
- OpenCV for iOS available via CocoaPods/SPM
- Eigen is header-only, works as-is

### UI
- Use UIKit (not SwiftUI) for Metal/AR integration
- Point cloud: rewrite OpenGL ES → Metal
- 2D depth visualization: custom UIView with colormap

## Dependencies (not in git)
- **QNN SDK**: QAIRT 2.42 (`/home/arrl/qairt-2.42/qairt/2.42.0.251225/`)
- **OpenCV Android SDK**: Download separately, place in `android/opencv-android-sdk/`
- **Model weights**: `weights/` directory (DLC files for Android, mlpackage for iOS)
- **ARCore SDK**: Via Gradle dependency
