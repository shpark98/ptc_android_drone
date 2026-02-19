#!/usr/bin/env python3
"""
PR-Depth 중간 깊이 추정 결과 시각화 도구.

이 스크립트는 depth refinement 파이프라인의 모든 중간 깊이 추정 결과를 시각화합니다:
- prev_depth_used: 이전 프레임의 깊이 (fusion에 사용)
- z_warp_flow: optical flow로 warp된 이전 깊이
- z_warp_pose: 3D pose로 warp된 이전 깊이
- z_tri_forward: forward motion으로 triangulate한 깊이
- z_tri_backward: backward motion으로 triangulate한 깊이 (있을 경우)
- z_tri: 최종 선택된 triangulated 깊이
- z_refined: Kalman fusion 후 최종 깊이

사용법:
    # 단일 결과 시각화
    python tools/viz/visualize_depth_stages.py --result result.pkl --output viz/

    # 시퀀스 시각화 (여러 프레임)
    python tools/viz/visualize_depth_stages.py --results_dir results/ --output viz/

    # 특정 프레임만
    python tools/viz/visualize_depth_stages.py --results_dir results/ --frame 10 --output viz/
"""
import argparse
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_result(path: str) -> dict:
    """결과 파일 로드 (pickle 또는 npz)."""
    path = Path(path)
    if path.suffix == '.pkl':
        with open(path, 'rb') as f:
            return pickle.load(f)
    elif path.suffix == '.npz':
        return dict(np.load(path, allow_pickle=True))
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {path.suffix}")


def depth_to_colormap(depth: np.ndarray, vmin: float = 0.5, vmax: float = 80.0,
                      cmap: str = 'turbo', invalid_color: tuple = (0, 0, 0)) -> np.ndarray:
    """깊이 맵을 컬러 이미지로 변환."""
    # 유효하지 않은 값 마스크
    valid = np.isfinite(depth) & (depth > 0)

    # 정규화
    depth_norm = np.clip(depth, vmin, vmax)
    depth_norm = (depth_norm - vmin) / (vmax - vmin)

    # 컬러맵 적용
    cm = plt.get_cmap(cmap)
    colored = cm(depth_norm)[:, :, :3]  # RGB만

    # 유효하지 않은 픽셀은 검정색
    colored[~valid] = invalid_color

    return (colored * 255).astype(np.uint8)


def variance_to_colormap(variance: np.ndarray, vmax: float = 10.0) -> np.ndarray:
    """분산 맵을 컬러 이미지로 변환 (낮을수록 파란색, 높을수록 빨간색)."""
    valid = np.isfinite(variance)
    var_norm = np.clip(variance, 0, vmax) / vmax

    cm = plt.get_cmap('RdYlBu_r')  # 빨강-노랑-파랑 역순
    colored = cm(var_norm)[:, :, :3]
    colored[~valid] = (0, 0, 0)

    return (colored * 255).astype(np.uint8)


def create_depth_comparison_figure(result: dict, frame_idx: int = None,
                                   vmin: float = 0.5, vmax: float = 80.0,
                                   figsize: tuple = (20, 16)) -> plt.Figure:
    """모든 중간 깊이 맵을 비교하는 figure 생성."""

    # 가능한 깊이 맵 목록
    depth_keys = [
        ('prev_depth_used', 'Previous Depth (t-1)', 'turbo'),
        ('z_warp_flow', 'Warped by Flow', 'turbo'),
        ('z_warp_pose', 'Warped by Pose', 'turbo'),
        ('z_tri_forward', 'Triangulated (Forward)', 'turbo'),
        ('z_tri_backward', 'Triangulated (Backward)', 'turbo'),
        ('z_tri', 'Triangulated (Final)', 'turbo'),
        ('z_refined', 'Refined (Final)', 'turbo'),
    ]

    # 분산 맵
    variance_keys = [
        ('V_prior', 'Variance (Prior)'),
        ('V_post', 'Variance (Posterior)'),
    ]

    # 존재하는 깊이 맵만 필터링
    available_depths = [(k, t, c) for k, t, c in depth_keys if k in result and result[k] is not None and len(result[k]) > 0]
    available_variances = [(k, t) for k, t in variance_keys if k in result and result[k] is not None and len(result[k]) > 0]

    n_depths = len(available_depths)
    n_variances = len(available_variances)
    n_total = n_depths + n_variances

    if n_total == 0:
        print("시각화할 깊이 맵이 없습니다.")
        return None

    # 그리드 레이아웃 결정
    if n_total <= 4:
        nrows, ncols = 2, 2
    elif n_total <= 6:
        nrows, ncols = 2, 3
    elif n_total <= 9:
        nrows, ncols = 3, 3
    else:
        nrows, ncols = 3, 4

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    title = "PR-Depth Intermediate Depth Maps"
    if frame_idx is not None:
        title += f" (Frame {frame_idx})"
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 깊이 맵 플롯
    for i, (key, title, cmap) in enumerate(available_depths):
        ax = axes[i]
        depth = np.array(result[key])

        # NaN 통계
        valid = np.isfinite(depth) & (depth > 0)
        valid_pct = 100 * np.sum(valid) / valid.size

        if np.sum(valid) > 0:
            mean_depth = np.nanmean(depth[valid])
            median_depth = np.nanmedian(depth[valid])
            stats_text = f"Valid: {valid_pct:.1f}%, Mean: {mean_depth:.2f}m, Med: {median_depth:.2f}m"
        else:
            stats_text = "No valid pixels"

        colored = depth_to_colormap(depth, vmin, vmax, cmap)
        im = ax.imshow(colored)
        ax.set_title(f"{title}\n{stats_text}", fontsize=10)
        ax.axis('off')

    # 분산 맵 플롯
    for j, (key, title) in enumerate(available_variances):
        ax = axes[n_depths + j]
        variance = np.array(result[key])

        valid = np.isfinite(variance)
        if np.sum(valid) > 0:
            mean_var = np.nanmean(variance[valid])
            stats_text = f"Mean Var: {mean_var:.4f}"
        else:
            stats_text = "No valid pixels"

        colored = variance_to_colormap(variance, vmax=10.0)
        ax.imshow(colored)
        ax.set_title(f"{title}\n{stats_text}", fontsize=10)
        ax.axis('off')

    # 빈 axes 숨기기
    for k in range(n_depths + n_variances, len(axes)):
        axes[k].axis('off')

    # 메타데이터 표시
    metadata_text = []
    if 'depth_consistency_score' in result:
        dc_score = result['depth_consistency_score']
        metadata_text.append(f"DC Score: {dc_score:.3f}")
    if 'dc_score_forward' in result:
        metadata_text.append(f"DC Fwd: {result['dc_score_forward']:.3f}")
    if 'dc_score_backward' in result:
        metadata_text.append(f"DC Bwd: {result['dc_score_backward']:.3f}")
    if 'used_backward' in result:
        metadata_text.append(f"Used Backward: {result['used_backward']}")
    if 'num_valid_tri' in result:
        metadata_text.append(f"Valid Tri: {result['num_valid_tri']}")
    if 'baseline' in result:
        metadata_text.append(f"Baseline: {result['baseline']:.4f}m")

    if metadata_text:
        fig.text(0.02, 0.02, ' | '.join(metadata_text), fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 컬러바 추가
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='turbo', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Depth (m)', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
    return fig


def create_depth_diff_figure(result: dict, frame_idx: int = None,
                             figsize: tuple = (16, 8)) -> plt.Figure:
    """깊이 맵 간의 차이를 시각화."""

    comparisons = [
        ('z_tri', 'z_warp_flow', 'Tri vs Warp(Flow)'),
        ('z_refined', 'z_tri', 'Refined vs Tri'),
        ('z_refined', 'z_warp_flow', 'Refined vs Warp(Flow)'),
    ]

    available = []
    for k1, k2, title in comparisons:
        if (k1 in result and k2 in result and
            result[k1] is not None and result[k2] is not None and
            len(result[k1]) > 0 and len(result[k2]) > 0):
            available.append((k1, k2, title))

    if not available:
        return None

    fig, axes = plt.subplots(1, len(available), figsize=figsize)
    if len(available) == 1:
        axes = [axes]

    fig_title = "Depth Difference Maps (Relative Error)"
    if frame_idx is not None:
        fig_title += f" (Frame {frame_idx})"
    fig.suptitle(fig_title, fontsize=14, fontweight='bold')

    for ax, (k1, k2, title) in zip(axes, available):
        d1 = np.array(result[k1]).astype(np.float32)
        d2 = np.array(result[k2]).astype(np.float32)

        valid = np.isfinite(d1) & np.isfinite(d2) & (d1 > 0) & (d2 > 0)

        # Relative error: (d1 - d2) / d2
        diff = np.zeros_like(d1)
        diff[valid] = (d1[valid] - d2[valid]) / (d2[valid] + 1e-6)
        diff[~valid] = np.nan

        # 컬러맵: 음수는 파란색, 양수는 빨간색
        vmax = 0.3  # ±30% 범위
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)

        if np.sum(valid) > 0:
            mae = np.nanmean(np.abs(diff[valid]))
            rmse = np.sqrt(np.nanmean(diff[valid]**2))
            stats = f"MAE: {mae:.3f}, RMSE: {rmse:.3f}"
        else:
            stats = "No valid overlap"

        ax.set_title(f"{title}\n{stats}", fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Rel. Error')

    plt.tight_layout()
    return fig


def run_live_visualization(pipeline, image_path: str, inv_depth_path: str,
                           baseline: float, output_dir: str):
    """실시간 파이프라인 실행 및 시각화."""
    import cv2

    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

    # Inverse depth 로드
    inv_depth = np.load(inv_depth_path).astype(np.float32)

    # 파이프라인 실행
    result = pipeline.refine(img, inv_depth, baseline)

    # 시각화
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = create_depth_comparison_figure(result)
    if fig:
        fig.savefig(output_dir / 'depth_stages.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    fig_diff = create_depth_diff_figure(result)
    if fig_diff:
        fig_diff.savefig(output_dir / 'depth_diff.png', dpi=150, bbox_inches='tight')
        plt.close(fig_diff)

    print(f"시각화 저장됨: {output_dir}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description='PR-Depth 중간 깊이 추정 결과 시각화',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--result', type=str, help='단일 결과 파일 경로 (.pkl 또는 .npz)')
    parser.add_argument('--results_dir', type=str, help='결과 파일들이 있는 디렉토리')
    parser.add_argument('--frame', type=int, help='특정 프레임만 시각화 (results_dir 사용 시)')
    parser.add_argument('--output', type=str, default='viz_output', help='출력 디렉토리')
    parser.add_argument('--vmin', type=float, default=0.5, help='깊이 시각화 최소값 (m)')
    parser.add_argument('--vmax', type=float, default=80.0, help='깊이 시각화 최대값 (m)')
    parser.add_argument('--no-diff', action='store_true', help='차이 맵 시각화 비활성화')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_to_process = []

    if args.result:
        # 단일 파일 처리
        results_to_process.append((args.result, None))

    elif args.results_dir:
        # 디렉토리 내 파일 처리
        results_dir = Path(args.results_dir)
        files = sorted(list(results_dir.glob('*.pkl')) + list(results_dir.glob('*.npz')))

        for i, f in enumerate(files):
            if args.frame is not None and i != args.frame:
                continue
            results_to_process.append((str(f), i))

    else:
        parser.print_help()
        print("\n에러: --result 또는 --results_dir 중 하나를 지정해야 합니다.")
        return

    for result_path, frame_idx in results_to_process:
        print(f"처리 중: {result_path}")

        try:
            result = load_result(result_path)
        except Exception as e:
            print(f"  로드 실패: {e}")
            continue

        # 깊이 비교 figure
        fig = create_depth_comparison_figure(result, frame_idx, args.vmin, args.vmax)
        if fig:
            if frame_idx is not None:
                out_path = output_dir / f'depth_stages_{frame_idx:04d}.png'
            else:
                out_path = output_dir / 'depth_stages.png'
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  저장됨: {out_path}")

        # 차이 맵 figure
        if not args.no_diff:
            fig_diff = create_depth_diff_figure(result, frame_idx)
            if fig_diff:
                if frame_idx is not None:
                    out_path = output_dir / f'depth_diff_{frame_idx:04d}.png'
                else:
                    out_path = output_dir / 'depth_diff.png'
                fig_diff.savefig(out_path, dpi=150, bbox_inches='tight')
                plt.close(fig_diff)
                print(f"  저장됨: {out_path}")

    print(f"\n완료! 결과는 {output_dir}에 저장되었습니다.")


if __name__ == '__main__':
    main()
