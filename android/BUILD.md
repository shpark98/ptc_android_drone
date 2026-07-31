# Build & Install Guide — PTC-Depth Android

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

## 2. Native dependencies — committed (self-contained clone)

Everything the build needs is **already in the repo** — no downloads, no
`setup_libs.sh`. A plain `git clone` gives you:

| Dependency | Location (in git) | Notes |
|------------|-------------------|-------|
| OpenCV 4.8.0 (arm64-v8a only) | `android/opencv-android-sdk/` | trimmed to arm64-v8a + JNI headers to keep the repo small (the app builds arm64-v8a only) |
| Eigen 3.4.0 (header-only) | `android/app/libs/eigen3/` | |
| QNN HTP runtime libs | `android/app/libs/arm64-v8a/` | from QAIRT 2.42 SDK; see below |
| Model files | `android/app/src/main/assets/` | `depth_anything_v2.onnx`+`.data` (QNN), `depth_anything.onnx` (CPU fallback) |
| ONNX Runtime QNN EP | Gradle | `com.microsoft.onnxruntime:onnxruntime-android-qnn:1.24.3` |

Model files: `.onnx` + `.data` must sit next to each other. The `.data` is the
external weights referenced by `depth_anything_v2.onnx`.

### Where these came from (only needed to *refresh/extend*, not to build)
- **OpenCV + Eigen**: `android/setup_libs.sh` downloads the full multi-ABI OpenCV
  4.8.0 SDK + Eigen 3.4.0. The committed OpenCV is that SDK trimmed to
  `sdk/native/{jni,libs/arm64-v8a,3rdparty/libs/arm64-v8a,staticlibs/arm64-v8a}`.
- **QNN libs** (from the Qualcomm QAIRT 2.42 SDK):
  `libQnnHtp.so`, `libQnnHtpPrepare.so`, `libQnnSystem.so`, `libQnnModelDlc.so`,
  and per-arch `libQnnHtpV{69,73,79,81}{Stub,Skel}.so`. Stubs are AArch64; skels
  are **Hexagon** ELFs (run on the DSP; `<QAIRT>/lib/hexagon-vNN/unsigned/`).
  Add a new arch's `Stub`+`Skel` here to support a new chip.
- **Model files**: outputs of the conversion scripts in `archive/scripts/` +
  `archive/python-ml/model_conversion/`.

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

## 7. AgileX Scout wheel encoder debug (direct USB-CAN)

The app contains a receive-only path matching the `gs_usb` CAN-to-USB setup in
`agilexrobotics/ugv_sdk`. Android USB host talks directly to the adapter; there
is no Linux `can0`, SocketCAN, serial-port library, or SLCAN text conversion in
this path.

1. Connect the phone in USB host/OTG mode to a `gs_usb`/candleLight-compatible
   adapter, and connect the adapter to the Scout CAN bus.
2. Power the Scout and accept the Android USB permission dialog.
3. The app configures channel 0 for classic CAN at **500 kbit/s**, receive-only.
4. Read the green three-line panel below the top HUD. It shows connection state,
   last raw RX (`last TX` remains `--`), `0x311` left/right odometry counts, and
   `0x251..0x258` motor RPM/pulse counts.

The direct driver recognizes the same common VID/PID pairs as the upstream Linux
`gs_usb` driver. `No gs_usb-style USB-CAN adapter found` means that no matching
device is attached. An adapter in SLCAN/CDC serial mode is a different USB
protocol and is intentionally not opened by this driver.

Useful log filter:

```bash
adb logcat | grep -E "GsUsbCanReceiver|MainActivity"
```

Protocol V2 byte layout (from `ugv_sdk`): all multi-byte CAN payload fields are
big-endian. `0x311` is two signed 32-bit wheel counts. Each motor high-speed
state `0x251..0x258` is signed 16-bit RPM, signed 16-bit current in 0.1 A, and a
signed 32-bit cumulative pulse count.

## 8. Troubleshooting

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
