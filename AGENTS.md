ㄴ# AGENTS.md — guide for AI coding agents (Codex)

Guidance for an AI agent working on this repository. (Claude Code reads
`CLAUDE.md`, which carries the same facts.) Read this before making changes.

## What this repo is

Android app for real-time metric depth estimation: Depth Anything V2 on the
Hexagon NPU (via ONNX Runtime + QNN HTP) refined by a C++ multi-frame geometry
pipeline using ARCore poses. This is an **Android-focused handoff**; the Python
ML pipeline and large data live in `archive/` (git-ignored, out of scope).

Full build/install/troubleshooting: [`android/BUILD.md`](android/BUILD.md).

## Where things are

| Area | Path |
|------|------|
| Frame loop, UI, 5 view modes | `android/app/src/main/java/com/ptcdepth/android/MainActivity.kt` |
| ARCore session / pose extraction | `.../ARCoreManager.kt` |
| NPU inference (QNN HTP, ORT) | `.../DepthEstimatorQNN.kt` |
| CPU fallback inference | `.../DepthEstimator.kt` |
| JNI → C++ pipeline bridge | `.../DepthRefinementManager.kt`, `android/app/src/main/cpp/jni_bridge.cpp` |
| QNN preprocessing JNI | `android/app/src/main/cpp/qnn_jni_bridge.cpp` |
| C++ pipeline (required, shared) | `cpp/src/*.cpp`, `cpp/include/ptc_depth/` |
| Native build config | `android/app/src/main/cpp/CMakeLists.txt`, `android/app/build.gradle.kts` |

The project is **PTC-Depth** (package `com.ptcdepth.android`, C++ core
`PTCDepth`). The repo directory is `pr_depth_android` and a few old code comments
say "pr_depth" — legacy only; the name is PTC-Depth.

## Build / run / verify

```bash
cd android
./gradlew :app:assembleDebug                                            # build
adb install -r --no-streaming app/build/outputs/apk/debug/app-debug.apk # install (large APK)
adb shell am start -n com.ptcdepth.android/.MainActivity                # launch
adb logcat | grep -E "DepthEstimatorQNN|ARCoreManager|FATAL"           # verify
```

- Self-contained: OpenCV (arm64-v8a), Eigen, QNN libs, and model files are all
  committed — just need the Android SDK/NDK. `cpp/` is required (CMake references it).
- Do a **clean** build (`:app:clean :app:assembleDebug`) after asset/native
  changes; incremental packaging can bloat the APK with orphaned bytes.
- A healthy QNN start logs `Detected SoC: '<model>' → soc_model=.., htp_arch=..`
  and `Successfully opened file ...libQnnHtpV<arch>Skel.so`.

## Coordinate conventions (CRITICAL — do not "fix" casually)

The pipeline uses the **point-transform** convention `P_curr = R·P_prev + baseline·t`
with `R = R_cam^T` (`motion_field.cpp`). ARCore→pipeline conversion:
1. OpenGL→CV: `C = diag(1,-1,-1)`, `R_cv = C·R_gl·C`, `t_cv = C·t_gl`.
2. Relative pose: `currPose.inverse().compose(prevPose)` = T_curr←prev.
3. Portrait 90° CW image rotation applies `C_rot = [[0,-1,0],[1,0,0],[0,0,1]]` to R,t
   in `jni_bridge.cpp`; also swap `fx↔fy, cx↔cy, w↔h` in intrinsics.

## Known gotchas (already handled — keep them that way)

- **16 KB page alignment** (Android 15/16 / SM8850): all AArch64 `.so` must be
  16 KB-aligned or the OS shows an "ELF alignment check" warning. Kept via NDK
  r27c + `-Wl,-z,max-page-size=16384` in CMake + ARCore 1.48 + no CameraX.
  Hexagon skels don't count. Don't downgrade the NDK or ARCore. (`android/BUILD.md` §6)
- **QNN SoC/arch mapping**: `DepthEstimatorQNN.kt` maps `Build.SOC_MODEL` →
  `soc_model`/`htp_arch`. New chip = add a branch + ship its skel/stub.
  (`android/BUILD.md` §5)
- **HTP manifest needs**: `extractNativeLibs="true"` +
  `<uses-native-library android:name="libcdsprpc.so"/>`. QNN **GPU** backend does
  not work on Android (linker namespace blocks `libOpenCL.so`); use **HTP**.
- **adb**: large-APK streaming install can hang → `--no-streaming`; if `adb shell`
  goes unresponsive, `adb kill-server && adb start-server`.

## Extending: external scale / sensor input

The pipeline's **metric scale** ("이동거리" / travel distance) currently comes from
ARCore VIO. It flows as a single scalar `baseline` (metres):

```
ARCoreManager.computeRelativePose()   baseline = ‖relative translation‖   (VIO)
  → MainActivity.processFrame          relPose.baseline
  → DepthRefinementManager.processFrameSync(..., baseline)
  → JNI nativeProcessFrameYUV(..., baseline)   (jni_bridge.cpp)
  → C++ pipeline.refine(gray, invDepth, baseline, ext_R, ext_t, ...)
```

Convention: `P_curr = R·P_prev + baseline·t`, where `t` is a unit direction and
`baseline` is its metre-scale magnitude (`jni_bridge.cpp` does `ext_t = t/‖t‖ *
baseline`).

**To drive scale from an external sensor instead of VIO:** the single injection
point is `MainActivity.kt` — grep `SCALE INJECTION POINT`. Replace the
`metricBaseline` value with your sensor's travel distance for the same prev→curr
frame interval. That's the only change needed for scale.

What you must still provide/handle (this is the real work — the seam itself is trivial):
- **A data source.** Today only ARCore feeds the app; add an input path for your
  sensor (BLE/serial/USB/socket/IMU). No such interface exists yet.
- **Time sync.** `baseline` must be the distance over the *same* prev→curr frame
  interval the depth pair spans (align by timestamp).
- **Units.** Metres (ARCore world units are metres).
- **Direction too?** If you also replace the translation *direction*, honor the CV
  convention (`C = diag(1,-1,-1)`) and the portrait rotation `C_rot` that
  `ARCoreManager`/`jni_bridge.cpp` apply. Replacing only the scalar magnitude
  avoids all of that.

## Conventions

- Match surrounding code style; keep changes minimal and scoped.
- Do not commit large binaries (models, QNN libs, OpenCV) — they are git-ignored
  and provisioned separately.
- Do not resurrect `archive/` content into the build.
