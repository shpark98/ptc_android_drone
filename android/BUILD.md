# Build & Install Guide — PR-Depth Android

Complete instructions to build the APK, provision dependencies, and install on a
device. Verified on **Samsung SM-S948N (SM8850, Snapdragon 8 Elite Gen 5,
Hexagon V81, Android 16)** with Ubuntu as the build host.

---

## 1. Toolchain

Install the Android SDK + these exact components:

```bash
sdkmanager --update
sdkmanager \
  "platform-tools" \
  "platforms;android-34" \
  "build-tools;34.0.0" \
  "ndk;27.2.12479018" \
  "cmake;3.22.1"
sdkmanager --licenses
```

| Tool         | Version        | Why this version |
|--------------|----------------|------------------|
| compileSdk   | 34             | set in `app/build.gradle.kts` |
| NDK          | **27.2.12479018 (r27c)** | **required** — r27+ links native `.so` with 16 KB-aligned LOAD segments by default (see §6). Older NDKs (e.g. r25) produce 4 KB libs that trip the "ELF alignment check" warning on Android 15/16. Pinned via `ndkVersion` in `app/build.gradle.kts`. |
| CMake        | 3.22.1         | set in `app/build.gradle.kts` |
| Gradle       | 8.2 (wrapper)  | downloaded automatically by `./gradlew` |
| AGP          | 8.2.0          | |

`android/install_android_sdk.sh` bootstraps the SDK, but **edit the NDK version
inside it to r27c** (the committed copy predates the 16 KB fix).

Create `android/local.properties` (git-ignored):

```properties
sdk.dir=/home/<you>/Android/Sdk
```

## 2. Native dependencies (not in git)

### 2a. OpenCV + Eigen — automated
```bash
cd android
./setup_libs.sh      # downloads OpenCV 4.8.0 Android SDK + Eigen 3.4.0
```
Note: the app's `CMakeLists.txt` expects the OpenCV SDK at
`android/opencv-android-sdk/`. `setup_libs.sh` currently unpacks into
`app/libs/opencv-android-sdk`; if the build can't find OpenCV, symlink or move it:
`ln -s app/libs/opencv-android-sdk android/opencv-android-sdk`. Eigen is expected
at `android/app/libs/eigen3/`.

### 2b. QNN / QAIRT runtime libs — manual
The app bundles QNN HTP runtime libraries in `android/app/app/libs/arm64-v8a/`
(git-ignored). They come from the **Qualcomm QAIRT 2.42 SDK**. Required files:

- `libQnnHtp.so`, `libQnnHtpPrepare.so`, `libQnnSystem.so`, `libQnnModelDlc.so`
- Per-arch stubs + skels: `libQnnHtpV{69,73,79,81}Stub.so` and
  `libQnnHtpV{69,73,79,81}Skel.so`
  - Stubs are AArch64; skels are **Hexagon** ELFs that run on the DSP.
  - V81 is required for SM8850; V79 for S25; V69 for S22.
  - Skels live under `<QAIRT>/lib/hexagon-vNN/unsigned/`, other libs under
    `<QAIRT>/lib/aarch64-android/`.
- `libc++_shared.so` comes from the NDK (r27c → 16 KB-aligned).

ONNX Runtime with the QNN EP is pulled via Gradle
(`com.microsoft.onnxruntime:onnxruntime-android-qnn:1.24.3`) — no manual step.

### 2c. Model assets — provided separately
Place these in `android/app/src/main/assets/` (git-ignored, ~190 MB total):

| File | Size | Used by |
|------|------|---------|
| `depth_anything_v2.onnx` | 1.3 MB | QNN path (graph) |
| `depth_anything_v2.data`  | 95 MB  | QNN path (external weights, referenced by the .onnx) |
| `depth_anything.onnx`     | 95 MB  | CPU fallback (`DepthEstimator`) |

`.onnx` + `.data` must sit next to each other. These are outputs of the model
conversion scripts now in `archive/scripts/` + `archive/python-ml/model_conversion/`.
Obtain them from the previous owner or regenerate from the archive.

## 3. Build

```bash
cd android
./gradlew :app:assembleDebug            # debug APK
# APK: app/build/outputs/apk/debug/app-debug.apk   (~272 MB)
```

Do a **clean** build after changing assets or native config — incremental
`packageDebug` can leave orphaned bytes in the APK, inflating its size:
```bash
./gradlew :app:clean :app:assembleDebug
```

## 4. Install

The APK is large (~272 MB). Streaming install can hang on some Samsung devices;
use the push method:

```bash
adb install -r --no-streaming app/build/outputs/apk/debug/app-debug.apk
```

First launch: grant camera permission (`adb shell pm grant com.ptcdepth.android
android.permission.CAMERA`), unlock the phone (camera can't open from background),
and ARCore compiles the QNN context binary on the first run (cached afterward at
`cache/depth_anything_qnn_ctx_v<arch>.bin`).

## 5. Runtime notes — QNN SoC/arch mapping

`DepthEstimatorQNN.kt` selects the QNN EP `soc_model` / `htp_arch` from
`Build.SOC_MODEL`. If you bring up a **new chip**, add a branch:

| SoC (Build.SOC_MODEL) | soc_model | htp_arch | device |
|-----------------------|-----------|----------|--------|
| SM8450 | 36 | 69 (V69) | Galaxy S22 (8 Gen 1) |
| SM8550 | 43 | 73 (V73) | 8 Gen 2 |
| SM8650 | 57 | 75 (V75) | 8 Gen 3 |
| SM8750 | 69 | 79 (V79) | Galaxy S25 (8 Elite) |
| SM8850 | 87 | 81 (V81) | SM-S948N (8 Elite Gen 5) |

- `soc_model` values come from `<QAIRT>/include/QNN/QnnTypes.h` (`QNN_SOC_MODEL_*`).
- `htp_arch` values come from `<QAIRT>/include/QNN/HTP/QnnHtpDevice.h`
  (`QNN_HTP_DEVICE_ARCH_*`).
- Ship the matching `libQnnHtpV<arch>Skel.so` + `Stub.so` in `app/libs/arm64-v8a/`.
- The cache filename is keyed by arch, so switching devices won't reuse a stale
  context binary.

## 6. 16 KB page alignment (Android 15/16)

Devices with 16 KB memory pages (e.g. SM8850) show an "ELF alignment check
failed" compatibility dialog if any bundled **AArch64** `.so` has 4 KB-aligned
LOAD segments. The Hexagon skel libs do **not** count (different architecture,
loaded on the DSP). This repo is already fixed:

- Our JNI libs: `add_link_options(-Wl,-z,max-page-size=16384 ...)` in
  `app/src/main/cpp/CMakeLists.txt`.
- `libc++_shared.so` + our libs: NDK r27c (16 KB by default).
- ARCore: `com.google.ar:core:1.48.0` (1.48+ ships 16 KB libs).
- CameraX was removed (unused; its lib was 4 KB).

Verify alignment (should print `0x4000` for every AArch64 lib):
```bash
RE=$ANDROID_HOME/ndk/27.2.12479018/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf
for so in app/build/intermediates/merged_native_libs/debug/out/lib/arm64-v8a/*.so; do
  [ "$($RE -h "$so" | awk -F: '/Machine/{print $2}' | xargs)" = "AArch64" ] || continue
  echo "$($RE -l "$so" | awk '/LOAD/{print $NF; exit}')  $(basename "$so")"
done
```

## 7. Troubleshooting

- **`adb shell` hangs / device shows but unresponsive** (often after an aborted
  large install): `adb kill-server && adb start-server`. Re-auth is not needed if
  "always allow" was checked.
- **Device not detected at all**: enable USB debugging; the product ID must switch
  from `6860` (MTP only) to `6864` (MTP+ADB). A charge-only cable won't work.
- **Install hangs on streaming**: use `--no-streaming` (§4).
- **QNN init fails** → the app falls back to ONNX Runtime CPU (`DepthEstimator`,
  `depth_anything.onnx`). Check logcat for `DepthEstimatorQNN` lines; a healthy
  run logs `Detected SoC: '<model>' → soc_model=.., htp_arch=..` and
  `Successfully opened file ...libQnnHtpV<arch>Skel.so`.
- **Manifest requirements**: `extractNativeLibs="true"` and
  `<uses-native-library android:name="libcdsprpc.so"/>` are required for the HTP
  backend — keep them.
