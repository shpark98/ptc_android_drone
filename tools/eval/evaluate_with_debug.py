#!/usr/bin/env python3
"""
PR-Depth 평가 + 중간 깊이 결과 시각화 스크립트.

모든 프레임의 중간 깊이 맵을 저장합니다:
- prev_depth_used, z_warp_flow, z_warp_pose
- z_tri_forward, z_tri_backward, z_tri
- z_refined, V_prior, V_post

사용법:
    python tools/eval/evaluate_with_debug.py -e debug_test \
        --date 2011_10_03 --drive 0027 \
        --max-frames 100 \
        --save-every 10  # 10 프레임마다 시각화 저장
"""
import argparse
import sys
import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'cpp' / 'build'))

from configs import get_dataset_paths
from dataloader import KITTIEigenSplit
import pr_depth_cpp as cpp


def create_pipeline(H, W, fx, fy, cx, cy):
    """PR-Depth 파이프라인 생성."""
    config = cpp.DepthRefinementConfig()
    config.H = H
    config.W = W
    config.fx = fx
    config.fy = fy
    config.cx = cx
    config.cy = cy
    config.debug = False  # 중간 결과는 항상 저장됨 (수정 완료)
    config.timing = False
    return cpp.DepthRefinement(config)


def visualize_frame(result, frame_idx, output_dir, vmax=80.0):
    """단일 프레임의 중간 깊이 맵 시각화."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    depth_keys = [
        ('prev_depth_used', 'Previous Depth (t-1)'),
        ('z_warp_flow', 'Warped by Flow'),
        ('z_warp_pose', 'Warped by Pose'),
        ('z_tri_forward', 'Triangulated (Forward)'),
        ('z_tri_backward', 'Triangulated (Backward)'),
        ('z_tri', 'Triangulated (Final)'),
        ('z_refined', 'Refined (Final)'),
    ]

    # 존재하는 깊이 맵만 필터링
    available = []
    for k, title in depth_keys:
        if k in result and result[k] is not None and len(result[k]) > 0:
            available.append((k, title, result[k]))

    if not available:
        return None

    n = len(available)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    fig.suptitle(f'Frame {frame_idx} - Intermediate Depth Maps', fontsize=14, fontweight='bold')

    for i, (key, title, depth) in enumerate(available):
        ax = axes[i]
        depth = np.array(depth)

        valid = np.isfinite(depth) & (depth > 0)
        valid_pct = 100 * np.sum(valid) / valid.size

        if np.sum(valid) > 0:
            mean_d = np.nanmean(depth[valid])
            stats = f"Valid: {valid_pct:.1f}%, Mean: {mean_d:.1f}m"
        else:
            stats = "No valid pixels"

        # 컬러맵 적용
        depth_clipped = np.clip(depth, 0.5, vmax)
        depth_norm = (depth_clipped - 0.5) / (vmax - 0.5)
        depth_norm[~valid] = 0

        im = ax.imshow(depth_norm, cmap='turbo', vmin=0, vmax=1)
        ax.set_title(f"{title}\n{stats}", fontsize=9)
        ax.axis('off')

    # 빈 axes 숨기기
    for j in range(len(available), len(axes)):
        axes[j].axis('off')

    # 메타데이터
    metadata = []
    if 'depth_consistency_score' in result:
        metadata.append(f"DC: {result['depth_consistency_score']:.3f}")
    if 'used_backward' in result:
        metadata.append(f"Backward: {result['used_backward']}")
    if 'baseline' in result:
        metadata.append(f"Baseline: {result['baseline']:.4f}m")

    if metadata:
        fig.text(0.02, 0.02, ' | '.join(metadata), fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 컬러바
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = mcolors.Normalize(vmin=0.5, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='turbo', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Depth (m)', fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])

    out_path = output_dir / f'depth_stages_{frame_idx:04d}.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

    return out_path


def main():
    parser = argparse.ArgumentParser(description='PR-Depth 평가 + 중간 깊이 시각화')
    parser.add_argument('-e', '--exp-name', required=True, help='실험 이름')
    parser.add_argument('--date', default='2011_10_03', help='KITTI 날짜')
    parser.add_argument('--drive', default='0027', help='KITTI 드라이브')
    parser.add_argument('--max-frames', type=int, default=None, help='최대 프레임 수')
    parser.add_argument('--save-every', type=int, default=10, help='N 프레임마다 시각화 저장')
    parser.add_argument('--vmax', type=float, default=80.0, help='깊이 시각화 최대값')
    parser.add_argument('--save-pkl', action='store_true', help='결과를 pkl로도 저장')

    args = parser.parse_args()

    # 데이터셋 경로
    paths = get_dataset_paths('kitti')
    rgb_path = paths['rgb_path']
    depth_path = paths['depth_path']

    # 출력 디렉토리
    dataset_id = f"kitti_{args.date}_{args.drive}"
    output_dir = PROJECT_ROOT / 'results' / dataset_id / 'pr_depth' / args.exp_name / 'debug_depths'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"출력 디렉토리: {output_dir}")

    # 데이터셋 로드
    print(f"Loading KITTI {args.date}/{args.drive}...")
    dataset = KITTIEigenSplit(rgb_path, args.date, args.drive, depth_path)
    print(f"Loaded {len(dataset)} frames")

    # 파이프라인 설정
    H, W = dataset.get_image_size()
    fx, fy, cx, cy = dataset.get_intrinsics()
    pipeline = create_pipeline(H, W, fx, fy, cx, cy)

    # Depth estimator (DepthAnything)
    from src.estimators.depth import DepthAnythingEstimator
    depth_estimator = DepthAnythingEstimator(encoder='vitl')

    # 실행
    max_frames = args.max_frames or len(dataset)
    max_frames = min(max_frames, len(dataset))

    saved_count = 0
    for idx in tqdm(range(max_frames), desc="Processing"):
        data = dataset.get(idx)
        if data is None:
            continue

        img = data['image_og']

        # Inverse depth 추정
        inv_depth = depth_estimator.infer(img)

        # Baseline
        baseline = dataset.get_baseline(idx) if idx > 0 else 0.1

        # 파이프라인 실행
        result = pipeline.refine(img, inv_depth, baseline)

        # N 프레임마다 시각화 저장
        if idx > 0 and idx % args.save_every == 0:
            out_path = visualize_frame(result, idx, output_dir, args.vmax)
            if out_path:
                saved_count += 1

            # pkl로도 저장 (옵션)
            if args.save_pkl:
                pkl_path = output_dir / f'result_{idx:04d}.pkl'
                with open(pkl_path, 'wb') as f:
                    pickle.dump(dict(result), f)

    print(f"\n완료! {saved_count}개의 시각화가 {output_dir}에 저장되었습니다.")


if __name__ == '__main__':
    main()
