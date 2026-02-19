"""Rerun entity logging for depth visualization."""

import numpy as np
import cv2
import rerun as rr
from typing import List, Tuple, Optional, Dict

from .data_source import FrameData, DepthSource
from .point_cloud import depth_to_points, colorize_points


# Distinct colors for methods
METHOD_COLORS: Dict[str, Tuple[int, int, int]] = {
    "GT": (0, 200, 0),           # green
    "PR-Depth": (0, 120, 255),   # blue
    "DA-v2": (255, 100, 0),      # orange
    "VDA": (200, 0, 200),        # purple
    "UniDepth": (255, 200, 0),   # yellow
}

TRAJECTORY_COLORS: Dict[str, Tuple[int, int, int]] = {
    "gt": (0, 200, 0),
    "estimated": (0, 120, 255),
}


class RerunLogger:
    """Manages Rerun logging for depth visualization."""

    def __init__(self, app_name: str = "pr_depth_viewer"):
        self.app_name = app_name
        self._gt_positions: List[np.ndarray] = []

    def init(self, port: int = 9876, serve: bool = True, save_path: str = None):
        """Initialize Rerun recording.

        Args:
            port: gRPC server port (when serve=True). Web viewer uses port+1.
            serve: Start web server (default). False = spawn native viewer.
            save_path: If set, save .rrd file instead of serving.
        """
        rr.init(self.app_name)

        if save_path:
            rr.save(save_path)
            print(f"\nSaving to: {save_path}")
            print("Open at https://app.rerun.io (drag & drop the .rrd file)\n")
        elif serve:
            web_port = port + 1
            rr.serve_web(open_browser=False, web_port=web_port, grpc_port=port)
            print(f"\nRerun web viewer: http://localhost:{web_port}")
            print(f"  gRPC port: {port}  |  Web port: {web_port}")
            print(f"  Both ports must be forwarded for remote access.\n")
        else:
            rr.spawn()

        # Set world coordinate system (LiDAR frame: X=forward, Y=left, Z=up)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    def log_frame(
        self,
        frame: FrameData,
        color_mode: str = "rgb",
        subsample: int = 2,
        max_depth: float = 80.0,
        depth_range: Optional[Tuple[float, float]] = None,
        point_size: float = 2.0,
    ):
        """Log a single frame's data to Rerun.

        Args:
            frame: Frame data with depth sources
            color_mode: Point coloring mode
            subsample: Subsample factor for dense depth maps
            max_depth: Maximum depth for point cloud
            depth_range: Shared depth range for colormap normalization
            point_size: Point radius in pixels
        """
        rr.set_time_sequence("frame", frame.idx)

        # Log camera
        H, W = frame.image.shape[:2]
        rr.log(
            "world/camera",
            rr.Pinhole(
                resolution=[W, H],
                image_from_camera=frame.K.astype(np.float32),
            ),
        )
        rr.log(
            "world/camera",
            rr.Transform3D(
                translation=frame.T_world_cam[:3, 3],
                mat3x3=frame.T_world_cam[:3, :3],
            ),
        )

        # Log RGB image (BGR -> RGB)
        rr.log("world/camera/rgb", rr.Image(frame.image[:, :, ::-1]))

        # Compute depth range across all sources for consistent coloring
        if depth_range is None and color_mode != "rgb":
            all_depths = []
            for src in frame.depth_sources:
                valid = src.depth[src.mask]
                if len(valid) > 0:
                    all_depths.append(valid)
            if all_depths:
                combined = np.concatenate(all_depths)
                depth_range = (np.percentile(combined, 2), np.percentile(combined, 98))

        # Log each depth source as point cloud
        for src in frame.depth_sources:
            self._log_depth_source(
                frame, src, color_mode, subsample, max_depth, depth_range, point_size
            )

        # Track camera position for trajectory
        self._gt_positions.append(frame.T_world_cam[:3, 3].copy())

        # Log trajectory
        if len(self._gt_positions) >= 2:
            positions = np.array(self._gt_positions)
            rr.log(
                "world/trajectory/gt",
                rr.LineStrips3D(
                    [positions],
                    colors=[(0, 200, 0)],
                ),
            )

    def _log_depth_source(
        self,
        frame: FrameData,
        src: DepthSource,
        color_mode: str,
        subsample: int,
        max_depth: float,
        depth_range: Optional[Tuple[float, float]],
        point_size: float,
    ):
        """Log a single depth source as a point cloud."""
        depth = src.depth.copy()
        depth[~src.mask] = 0

        # For sparse GT (LiDAR), don't subsample
        sparsity = src.mask.sum() / src.mask.size
        actual_subsample = 1 if sparsity < 0.5 else subsample

        pts_world, pixel_indices = depth_to_points(
            depth, frame.K, frame.T_world_cam,
            max_depth=max_depth, subsample=actual_subsample,
        )

        if len(pts_world) == 0:
            return

        # Get depth values at the projected points
        rows, cols = pixel_indices[:, 0], pixel_indices[:, 1]
        depth_values = depth[rows, cols]

        colors = colorize_points(
            frame.image, pixel_indices,
            depth_values=depth_values,
            mode=color_mode,
            depth_range=depth_range,
        )

        # Entity path: sanitize name, include frame idx for accumulation
        entity_name = src.name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
        entity_path = f"world/{entity_name}/map/f{frame.idx:04d}"

        method_color = METHOD_COLORS.get(src.name, (180, 180, 180))

        rr.log(
            entity_path,
            rr.Points3D(
                pts_world,
                colors=colors,
                radii=np.full(len(pts_world), point_size / 1000.0),
            ),
        )

    def log_frame_seq(
        self,
        frame: FrameData,
        color_mode: str = "rgb",
        subsample: int = 2,
        max_depth: float = 80.0,
        depth_range: Optional[Tuple[float, float]] = None,
        point_size: float = 2.0,
    ):
        """Log a frame for sequential (non-accumulating) playback.

        GT (LiDAR) uses turbo colormap for visibility.
        Other methods (PR-Depth etc.) use RGB from camera.
        """
        rr.set_time_sequence("frame", frame.idx)

        # Log camera pose
        H, W = frame.image.shape[:2]
        rr.log(
            "world/camera",
            rr.Pinhole(
                resolution=[W, H],
                image_from_camera=frame.K.astype(np.float32),
            ),
        )
        rr.log(
            "world/camera",
            rr.Transform3D(
                translation=frame.T_world_cam[:3, 3],
                mat3x3=frame.T_world_cam[:3, :3],
            ),
        )

        # Log RGB image to standalone path (outside camera transform hierarchy)
        rr.log("images/rgb", rr.Image(frame.image[:, :, ::-1]))

        # Compute shared depth range for colormap
        all_depths = []
        for src in frame.depth_sources:
            valid = src.depth[src.mask]
            if len(valid) > 0:
                all_depths.append(valid)
        if all_depths:
            combined = np.concatenate(all_depths)
            depth_range = (np.percentile(combined, 2), np.percentile(combined, 98))

        # Log each depth source
        for src in frame.depth_sources:
            depth = src.depth.copy()
            depth[~src.mask] = 0

            sparsity = src.mask.sum() / src.mask.size
            is_sparse = sparsity < 0.5  # GT LiDAR is sparse
            actual_subsample = 1 if is_sparse else subsample

            pts_world, pixel_indices = depth_to_points(
                depth, frame.K, frame.T_world_cam,
                max_depth=max_depth, subsample=actual_subsample,
            )

            # Transform points to camera-local coordinates for fixed view
            if len(pts_world) > 0:
                T_cam_world = np.linalg.inv(frame.T_world_cam)
                pts_cam = (T_cam_world[:3, :3] @ pts_world.T).T + T_cam_world[:3, 3]
                pts_world = pts_cam  # Replace with camera-local coords

            if len(pts_world) == 0:
                # Log empty so previous frame's data is cleared
                entity_name = src.name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
                rr.log(f"world/{entity_name}/points", rr.Clear(recursive=False))
                continue

            rows, cols = pixel_indices[:, 0], pixel_indices[:, 1]
            depth_values = depth[rows, cols]

            # GT (sparse LiDAR) → turbo colormap, larger points
            # Other methods → RGB from camera image
            if is_sparse:
                colors = colorize_points(
                    frame.image, pixel_indices,
                    depth_values=depth_values,
                    mode="turbo",
                    depth_range=depth_range,
                )
                pt_radius = point_size * 4.0 / 1000.0
            else:
                colors = colorize_points(
                    frame.image, pixel_indices,
                    depth_values=None,
                    mode="rgb",
                    depth_range=None,
                )
                pt_radius = point_size * 5.0 / 1000.0

            entity_name = src.name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
            entity_path = f"world/{entity_name}/points"

            rr.log(
                entity_path,
                rr.Points3D(
                    pts_world,
                    colors=colors,
                    radii=np.full(len(pts_world), pt_radius),
                ),
            )

            # Log depth heatmap to standalone path
            self._log_depth_image_standalone(frame, src, max_depth=max_depth)

        # Trajectory (accumulated)
        self._gt_positions.append(frame.T_world_cam[:3, 3].copy())
        if len(self._gt_positions) >= 2:
            positions = np.array(self._gt_positions)
            rr.log(
                "world/trajectory/gt",
                rr.LineStrips3D(
                    [positions],
                    colors=[(0, 200, 0)],
                ),
            )

    def _log_depth_image_standalone(self, frame: FrameData, src: DepthSource, max_depth: float = 80.0):
        """Log depth heatmap to standalone path (not under camera transform)."""
        depth = src.depth.copy()
        depth[~src.mask] = 0

        if not src.mask.any():
            return

        d_max = min(depth[src.mask].max(), max_depth)
        norm = np.clip(depth / d_max, 0, 1)
        gray = (norm * 255).astype(np.uint8)
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        colored[~src.mask] = 0

        entity_name = src.name.lower().replace(" ", "_").replace("-", "_")
        rr.log(f"images/depth_{entity_name}", rr.Image(colored[:, :, ::-1]))

    def log_depth_image(self, frame: FrameData, src: DepthSource, max_depth: float = 80.0):
        """Log a depth map as a 2D image in the camera view."""
        depth = src.depth.copy()
        depth[~src.mask] = 0

        # Normalize to [0, 255]
        d_max = min(depth[src.mask].max(), max_depth) if src.mask.any() else max_depth
        norm = np.clip(depth / d_max, 0, 1)
        gray = (norm * 255).astype(np.uint8)
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        colored[~src.mask] = 0

        entity_name = src.name.lower().replace(" ", "_").replace("-", "_")
        rr.log(f"world/camera/depth_{entity_name}", rr.Image(colored[:, :, ::-1]))

    def reset(self):
        """Reset accumulated state."""
        self._gt_positions = []
