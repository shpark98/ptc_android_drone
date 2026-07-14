# PTC-Depth — Android

Real-time **metric depth estimation on mobile**: monocular depth (Depth Anything V2)
refined with multi-frame geometry (optical flow → RANSAC → triangulation →
Kalman/Bayesian fusion) using ARCore camera poses.

This repository is an **Android-app-focused handoff**. It contains everything
needed to build, install, and work on the Android application. The original
Python ML pipeline (training, evaluation, model conversion) and large data
(model weights, phone captures) have been moved to [`archive/`](archive/README.md),
which is git-ignored and **not** part of this handoff.

> **New here?** Read [`android/BUILD.md`](android/BUILD.md) to build and install,
> and [`AGENTS.md`](AGENTS.md) / [`CLAUDE.md`](CLAUDE.md) if you are an AI coding
> agent (Codex / Claude).

## What the app does

Live camera feed → per-frame monocular depth on the Hexagon NPU (QNN HTP) →
C++ refinement pipeline fuses depth across frames using ARCore relative poses →
on-screen depth visualization / point cloud, with optional raw capture + PLY export.

## Architecture

```
┌─────────────── Android app (Kotlin) ────────────────┐
│ MainActivity ── frame loop, 5 view modes            │
│ ARCoreManager ── camera pose (R,t), depth API       │
│ DepthEstimatorQNN ── ONNX Runtime + QNN HTP (NPU)   │
│ DepthEstimator    ── ONNX Runtime CPU (fallback)    │
│ DepthRefinementManager ── JNI bridge                │
└──────────────────────┬──────────────────────────────┘
                       │ JNI
┌──────────────────────┴──────────────────────────────┐
│ C++ pipeline (cpp/ + android/app/src/main/cpp)       │
│ motion_field · triangulation · scale_estimation ·    │
│ depth_warp · bayesian_fusion · ptc_depth             │
│ (OpenCV + Eigen)                                     │
└──────────────────────────────────────────────────────┘
```

- **Depth model**: Depth Anything V2 Small, 518×518, fp16.
- **Inference**: ONNX Runtime QNN Execution Provider → Hexagon HTP. The ONNX
  graph is compiled to a QNN context binary on first launch and cached.
- **Pose**: ARCore.
- The C++ pipeline lives in [`cpp/`](cpp/) and is compiled into the app via CMake
  (the app's `CMakeLists.txt` references `../../../../../cpp`). **`cpp/` is required
  to build the APK.**

## Repository layout

```
android/           Android application (Gradle project) — build this
  app/src/main/
    java/com/ptcdepth/android/   Kotlin sources
    cpp/                         JNI bridges + QNN estimator (C++)
    assets/                      model files (git-ignored, provided separately)
  setup_libs.sh                  downloads OpenCV + Eigen into app/libs
  install_android_sdk.sh         one-shot Android SDK bootstrap
cpp/               C++ refinement pipeline (shared library sources) — required
CLAUDE.md          project guide for the Claude coding agent
AGENTS.md          project guide for the Codex coding agent
android/BUILD.md   full build / install / troubleshooting guide
archive/           git-ignored: Python ML pipeline, weights, captures (see its README)
```

## Quick start

This repo is **self-contained**: all native dependencies (OpenCV arm64-v8a,
Eigen, QNN HTP libs) and the model files are committed, so a plain `git clone`
is all you need besides the Android SDK/NDK.

```bash
git clone git@github.com:leezy211/depth_android.git
cd depth_android/android
# One-time: Android SDK + NDK r27c + cmake;3.22.1  (see android/BUILD.md §1)
echo "sdk.dir=$HOME/Android/Sdk" > local.properties

./gradlew :app:assembleDebug
# 16 KB-page devices need --no-streaming for the large (~272 MB) APK:
adb install -r --no-streaming app/build/outputs/apk/debug/app-debug.apk
```

No `setup_libs.sh` or separate downloads required. (`android/setup_libs.sh` is
kept only for re-fetching the full multi-ABI OpenCV SDK if ever needed.)

### What's committed (so `git clone` is enough)

Everything the APK build needs is in git:

- Source: `android/` + `cpp/`
- OpenCV 4.8.0, trimmed to arm64-v8a (`android/opencv-android-sdk/`)
- Eigen 3.4.0 (`android/app/libs/eigen3/`)
- QNN HTP runtime libs V69/V73/V79/V81 (`android/app/libs/arm64-v8a/`)
- Model files: `depth_anything_v2.onnx` + `.data` (QNN) and `depth_anything.onnx`
  (CPU fallback) in `android/app/src/main/assets/`

The repo is therefore ~500 MB. Every file is under GitHub's 100 MB limit, so
**no Git LFS is needed** — a plain `git clone` pulls it all. Not included: the
unused `snpe-release.aar`, and the Python ML pipeline / weights / captures
(git-ignored `archive/`).

Install on a device (large APK — streaming can hang on some Samsung phones):

```bash
adb install -r --no-streaming app/build/outputs/apk/debug/app-debug.apk
# if adb goes unresponsive: adb kill-server && adb start-server
```

Target/verified device: **Samsung SM-S948N (Snapdragon 8 Elite Gen 5 / SM8850,
Hexagon V81, Android 16)**. Also known to run on Galaxy S25 (SM8750/V79) and
S22 (SM8450/V69).

See [`android/BUILD.md`](android/BUILD.md) for the complete guide, dependency
sourcing, and troubleshooting (16 KB alignment, QNN SoC mapping, adb quirks).
