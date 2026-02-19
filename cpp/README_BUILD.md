# Building PR-Depth C++ Library

## Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    libeigen3-dev \
    python3-dev \
    python3-pip

# Install pybind11
pip3 install pybind11
```

## Build Instructions

```bash
cd cpp
mkdir -p build
cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Verify
ls -lh libpr_depth_core.so
ls -lh pr_depth_cpp*.so
```

## Test the Build

```bash
# Add build directory to Python path
export PYTHONPATH=$PWD:$PYTHONPATH

# Test import
python3 -c "import pr_depth_cpp; print(pr_depth_cpp.__doc__)"

# Should print:
# PR-Depth C++ core library - minimal bindings for parity testing
```

## Run Parity Test

```bash
# 1. Extract test fixtures (from repo root)
cd ../..
python scripts/extract_test_fixtures.py \
    <path_to_kitti_img_000.png> \
    <path_to_kitti_img_001.png>

# 2. Run parity test
PYTHONPATH=cpp/build:$PYTHONPATH python tests/parity/test_flow_parity.py
```

Expected output:
```
================================================================================
PARITY TEST: OpticalFlow C++ vs Python
================================================================================
✓ Loaded test images: (375, 1242, 3)

[1/2] Computing flow with Python (DIS)...
      Python flow shape: (375, 1242, 2), dtype: float32
      Flow magnitude range: [0.000, XX.XXX] px

[2/2] Computing flow with C++ (DIS)...
      C++ flow shape: (375, 1242, 2), dtype: float32
      Flow magnitude range: [0.000, XX.XXX] px

================================================================================
PARITY METRICS
================================================================================
Mean absolute difference:   <0.01 px
...
✅ PARITY TEST PASSED
```
