#!/bin/bash
# Quick Android SDK installation script

set -e

echo "=== Installing Android SDK Command Line Tools ==="

SDK_DIR="$HOME/Android/Sdk"
CMDLINE_TOOLS_URL="https://dl.google.com/android/repository/commandlinetools-linux-9477386_latest.zip"

# Create SDK directory
mkdir -p "$SDK_DIR/cmdline-tools"

# Download command line tools
echo "Downloading Android SDK command line tools..."
cd /tmp
wget -q --show-progress "$CMDLINE_TOOLS_URL" -O cmdline-tools.zip

# Extract
echo "Extracting..."
unzip -q cmdline-tools.zip
mv cmdline-tools "$SDK_DIR/cmdline-tools/latest"
rm cmdline-tools.zip

# Set environment variables
export ANDROID_HOME="$SDK_DIR"
export PATH="$PATH:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools"

echo "export ANDROID_HOME=$SDK_DIR" >> ~/.bashrc
echo "export PATH=\$PATH:\$ANDROID_HOME/cmdline-tools/latest/bin:\$ANDROID_HOME/platform-tools" >> ~/.bashrc

# Install required SDK components
echo ""
echo "Installing SDK components (this may take a few minutes)..."
cd "$ANDROID_HOME/cmdline-tools/latest/bin"

yes | ./sdkmanager --licenses > /dev/null 2>&1
./sdkmanager "platform-tools" "platforms;android-34" "build-tools;34.0.0" "ndk;25.2.9519653" "cmake;3.22.1"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Android SDK installed to: $SDK_DIR"
echo ""
echo "Run this to set environment for current session:"
echo "  export ANDROID_HOME=$SDK_DIR"
echo "  export PATH=\$PATH:\$ANDROID_HOME/cmdline-tools/latest/bin:\$ANDROID_HOME/platform-tools"
echo ""
echo "Then try building again:"
echo "  cd /home/arrl/Desktop/algorithm/pr_depth_android/android"
echo "  ./gradlew assembleDebug"
echo ""
