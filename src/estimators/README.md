# Unified Estimator Interfaces

This module provides clean, unified interfaces for depth, flow, and pose estimation.

## Structure

```
src/estimators/
├── __init__.py           # Main exports
├── depth/
│   ├── base.py           # DepthEstimator ABC
│   ├── depth_anything.py # DepthAnything v2 (ONNX)
│   └── unidepth.py       # UniDepth (metric depth)
├── flow/
│   ├── base.py           # FlowEstimator ABC
│   └── dis.py            # DIS optical flow
└── pose/
    ├── base.py           # PoseEstimator ABC, PoseResult
    ├── pr_depth.py       # PR-Depth C++ wrapper
    └── madpose.py        # MADPose wrapper
```

## Usage

```python
from src.estimators import (
    DepthAnythingEstimator,
    DISFlowEstimator,
    PRDepthEstimator,
    MADPoseEstimator,
)

# Initialize
depth_est = DepthAnythingEstimator(encoder='vitl')
flow_est = DISFlowEstimator(preset='medium')

# For PR-Depth (uses DepthAnything inverse depth)
pr_depth = PRDepthEstimator(H=375, W=1242, fx=fx, fy=fy, cx=cx, cy=cy)

# For MADPose (should use UniDepth metric depth)
madpose = MADPoseEstimator(K=K)

# Estimate
inv_depth = depth_est.infer(img)
flow = flow_est.compute(img0, img1)

result = pr_depth.estimate(img0, img1, depth0, depth1, flow, baseline)
print(f"R: {result.R}, t: {result.t}, success: {result.success}")
```

## Notes

- **PR-Depth** uses inverse depth (DepthAnything output) and requires baseline for metric scale
- **MADPose** should use metric depth (UniDepth) and does NOT need baseline (estimates scale internally)
- Both use the same PoseResult dataclass for consistent output

## Migration from src/modules/

The old `src/modules/` contains Python implementations that have mostly been replaced by C++.
Use `src/estimators/` for new code.
