#!/bin/bash
# Setup external libraries for PR-Depth Android

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBS_DIR="$SCRIPT_DIR/app/libs"

echo "=== PR-Depth Android Libraries Setup ==="

# Create libs directory
mkdir -p "$LIBS_DIR"
cd "$LIBS_DIR"

# ============================================================================
# 1. Download OpenCV Android SDK
# ============================================================================
echo ""
echo "[1/2] Downloading OpenCV Android SDK 4.8.0..."

if [ ! -d "opencv-android-sdk" ]; then
    OPENCV_VERSION="4.8.0"
    OPENCV_ZIP="opencv-${OPENCV_VERSION}-android-sdk.zip"
    OPENCV_URL="https://github.com/opencv/opencv/releases/download/${OPENCV_VERSION}/${OPENCV_ZIP}"

    echo "  Downloading from: $OPENCV_URL"
    curl -L -o "$OPENCV_ZIP" "$OPENCV_URL"

    echo "  Extracting..."
    unzip -q "$OPENCV_ZIP"
    mv "OpenCV-android-sdk" "opencv-android-sdk"
    rm "$OPENCV_ZIP"

    echo "  ✓ OpenCV installed to: $LIBS_DIR/opencv-android-sdk"
else
    echo "  ✓ OpenCV already installed"
fi

# ============================================================================
# 2. Download Eigen3
# ============================================================================
echo ""
echo "[2/2] Downloading Eigen3 (header-only)..."

if [ ! -d "eigen3" ]; then
    EIGEN_VERSION="3.4.0"
    EIGEN_ZIP="eigen-${EIGEN_VERSION}.zip"
    EIGEN_URL="https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/${EIGEN_ZIP}"

    echo "  Downloading from: $EIGEN_URL"
    curl -L -o "$EIGEN_ZIP" "$EIGEN_URL"

    echo "  Extracting..."
    unzip -q "$EIGEN_ZIP"
    mv "eigen-${EIGEN_VERSION}" "eigen3"
    rm "$EIGEN_ZIP"

    echo "  ✓ Eigen3 installed to: $LIBS_DIR/eigen3"
else
    echo "  ✓ Eigen3 already installed"
fi

# ============================================================================
# Done
# ============================================================================
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Libraries installed:"
echo "  - OpenCV: $LIBS_DIR/opencv-android-sdk"
echo "  - Eigen3: $LIBS_DIR/eigen3"
echo ""
echo "Next steps:"
echo "  1. Install Android SDK command-line tools"
echo "  2. Run: ./gradlew assembleDebug"
echo "  3. Install APK: adb install app/build/outputs/apk/debug/app-debug.apk"
echo ""
