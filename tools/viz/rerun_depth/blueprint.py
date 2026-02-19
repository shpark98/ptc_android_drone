"""Rerun Blueprint layout definitions."""

from typing import List
import rerun.blueprint as rrb


def make_single_frame_blueprint(method_names: List[str]) -> rrb.Blueprint:
    """Create layout for single-frame depth comparison.

    Layout:
        Top row: Side-by-side 3D viewports (one per method)
        Bottom row: Camera RGB | Depth heatmap
    """
    # 3D views — one per depth source
    spatial_views = []
    for name in method_names:
        entity = name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
        spatial_views.append(
            rrb.Spatial3DView(
                name=name,
                contents=[
                    f"world/{entity}/**",
                    "world/camera/**",
                    "world/trajectory/**",
                ],
            )
        )

    # If only one method, add GT alongside
    if len(spatial_views) == 1:
        spatial_views.insert(0, rrb.Spatial3DView(
            name="GT",
            contents=[
                "world/gt/**",
                "world/camera/**",
                "world/trajectory/**",
            ],
        ))

    top_row = rrb.Horizontal(*spatial_views)

    # Bottom row: image + depth
    bottom_row = rrb.Horizontal(
        rrb.Spatial2DView(
            name="Camera RGB",
            contents=["world/camera/rgb"],
        ),
        rrb.Spatial3DView(
            name="All Methods",
            contents=["world/**"],
        ),
    )

    return rrb.Blueprint(
        rrb.Vertical(top_row, bottom_row, row_shares=[2, 1]),
    )


def make_seq_blueprint(method_names: List[str]) -> rrb.Blueprint:
    """Create layout for sequential frame-by-frame comparison.

    Layout:
        Left: Single 3D viewport with all methods overlaid
              (GT=turbo colormap, PR-Depth=RGB)
        Right: Camera RGB + depth heatmaps stacked
    """
    # Build content list for the overlaid 3D view (no trajectory in seq mode)
    contents_3d = ["world/camera"]
    for name in method_names:
        entity = name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
        contents_3d.append(f"world/{entity}/**")

    main_3d = rrb.Spatial3DView(
        name="3D Overlay",
        contents=contents_3d,
        background=[0, 0, 0, 255],  # Black background
        line_grid=rrb.LineGrid3D(visible=False),  # No grid
    )

    # Right column: Camera RGB + depth heatmaps (standalone image paths)
    right_panels = [
        rrb.Spatial2DView(
            name="Camera RGB",
            contents=["images/rgb"],
        ),
    ]
    for name in method_names:
        entity = name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
        right_panels.append(
            rrb.Spatial2DView(
                name=f"{name} Depth",
                contents=[f"images/depth_{entity}"],
            )
        )

    right_col = rrb.Vertical(*right_panels)

    return rrb.Blueprint(
        rrb.Horizontal(main_3d, right_col, column_shares=[2, 1]),
    )


def make_slam_blueprint(method_names: List[str]) -> rrb.Blueprint:
    """Create layout for SLAM-like temporal accumulation.

    Layout:
        Top: Large 3D viewport with all accumulated clouds
        Bottom: Camera RGB | Trajectory overhead
    """
    # Main 3D view with everything
    main_3d = rrb.Spatial3DView(
        name="3D Scene",
        contents=["world/**"],
    )

    # Bottom panels
    camera_view = rrb.Spatial2DView(
        name="Camera",
        contents=["world/camera/rgb"],
    )

    # Overhead trajectory view
    trajectory_view = rrb.Spatial3DView(
        name="Trajectory",
        contents=[
            "world/trajectory/**",
            "world/camera",
        ],
    )

    bottom_row = rrb.Horizontal(camera_view, trajectory_view)

    return rrb.Blueprint(
        rrb.Vertical(main_3d, bottom_row, row_shares=[3, 1]),
    )
