"""Method runners for evaluation framework."""

from .pr_depth import PRDepthRunner
from .madpose import MADPoseRunner

__all__ = [
    'PRDepthRunner',
    'MADPoseRunner',
]

# Optional runners (may not be available)
try:
    from .batrack import BaTrackRunner
    __all__.append('BaTrackRunner')
except ImportError:
    pass

try:
    from .monst3r import MonST3RRunner
    __all__.append('MonST3RRunner')
except ImportError:
    pass

try:
    from .video_depth_anything import VideoDepthAnythingRunner, VideoDepthAnythingPoseRunner
    __all__.extend(['VideoDepthAnythingRunner', 'VideoDepthAnythingPoseRunner'])
except ImportError:
    pass
