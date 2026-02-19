"""파이프라인 삼각측량 테스트.

실제 파이프라인을 사용하여 motion estimation + triangulation 통합 테스트를 수행합니다.
GT pose를 사용한 삼각측량과 비교하여 파이프라인의 정확도를 평가합니다.

사용법:
    python tests/triangulation/test_pipeline.py
    python tests/triangulation/test_pipeline.py --verbose
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'cpp/build'))

import argparse
import numpy as np
import pr_depth_cpp as cpp
from dataloader import KITTIEigenSplit
from src.evaluation import compute_depth_metrics


def test_pipeline_triangulation(
    date: str = "2011_10_03",
    drive: str = "0027",
    start_frame: int = 10,
    num_frames: int = 10,
    verbose: bool = False,
):
    """파이프라인 삼각측량 테스트.

    Args:
        date: KITTI 날짜
        drive: KITTI 드라이브 번호
        start_frame: 시작 프레임
        num_frames: 테스트 프레임 수
        verbose: 상세 출력
    """
    print("=" * 70)
    print("파이프라인 삼각측량 테스트")
    print("=" * 70)

    # 데이터셋 로드
    dataset = KITTIEigenSplit(
        rgb_path='/home/nas/Dataset2/KITTI/KITTI_RGB_Image',
        depth_path='/home/nas/Dataset2/KITTI/KITTI_PointCloud',
        date=date, drive=drive,
    )

    H, W = dataset.get_image_size()
    fx, fy, cx, cy = dataset.get_intrinsics()

    print(f"데이터: {date}/{drive}")
    print(f"프레임: {start_frame} ~ {start_frame + num_frames - 1}")
    print(f"이미지 크기: {H}x{W}")
    print()

    # 파이프라인 설정
    config = cpp.DepthRefinementConfig()
    config.H, config.W = H, W
    config.fx, config.fy, config.cx, config.cy = fx, fy, cx, cy
    config.use_depth_consistency = False
    config.enable_iterative_refinement = False

    pipeline = cpp.DepthRefinement(config)

    # 첫 프레임 초기화
    data0 = dataset.get(start_frame - 1)
    inv_depth0 = np.ones((H, W), dtype=np.float32) * 0.1
    pipeline.refine(data0['image_og'], inv_depth0, 0.0)

    results_pipeline = {"d125": [], "mae": []}
    results_gt = {"d125": [], "mae": []}

    print(f"{'프레임':<8} | {'Pipeline d<1.25':<15} | {'GT Pose d<1.25':<15} | {'Pipeline MAE':<12} | {'GT MAE':<10}")
    print("-" * 75)

    for i, frame_idx in enumerate(range(start_frame, start_frame + num_frames)):
        data_prev = dataset.get(frame_idx - 1)
        data = dataset.get(frame_idx)
        gt_depth = data['depth_og']
        baseline = dataset.get_baseline(frame_idx)
        R_gt, t_gt = dataset.get_relative_pose(frame_idx)

        # 파이프라인 실행
        inv_depth = np.ones((H, W), dtype=np.float32) * 0.1
        result = pipeline.refine(data['image_og'], inv_depth, baseline)

        z_tri_pipe = result['z_tri']
        mask_pipe = (gt_depth > 0) & (gt_depth < 80) & (z_tri_pipe > 0) & (z_tri_pipe < 200)

        # GT pose로 삼각측량
        flow = cpp.compute_optical_flow(data_prev['image_og'], data['image_og'])
        u0 = np.tile(np.arange(W, dtype=np.float32), (H, 1))
        v0 = np.tile(np.arange(H, dtype=np.float32).reshape(-1, 1), (1, W))
        u1 = u0 + flow[:, :, 0]
        v1 = v0 + flow[:, :, 1]

        # 정확한 수식 사용: R_tri = R^T, C1 = -R^T @ t
        R_tri = R_gt.T
        C1 = -R_gt.T @ (t_gt * baseline)

        result_gt = cpp.triangulate_depth(
            u0.flatten().astype(np.float32),
            v0.flatten().astype(np.float32),
            u1.flatten().astype(np.float32),
            v1.flatten().astype(np.float32),
            R_tri, C1, fx, fy, cx, cy, H, W
        )
        z_tri_gt = result_gt['z1_tri']
        mask_gt = (gt_depth > 0) & (gt_depth < 80) & (z_tri_gt > 0) & (z_tri_gt < 200)

        # 메트릭 계산
        m_pipe = compute_depth_metrics(z_tri_pipe, gt_depth) if mask_pipe.sum() > 100 else None
        m_gt = compute_depth_metrics(z_tri_gt, gt_depth) if mask_gt.sum() > 100 else None

        d125_pipe = m_pipe['d125'] if m_pipe else 0.0
        mae_pipe = m_pipe['MAE'] if m_pipe else 0.0
        d125_gt = m_gt['d125'] if m_gt else 0.0
        mae_gt = m_gt['MAE'] if m_gt else 0.0

        results_pipeline["d125"].append(d125_pipe)
        results_pipeline["mae"].append(mae_pipe)
        results_gt["d125"].append(d125_gt)
        results_gt["mae"].append(mae_gt)

        print(f"F{frame_idx:02d}      | {d125_pipe:>13.1f}% | {d125_gt:>13.1f}% | {mae_pipe:>10.2f}m | {mae_gt:>8.2f}m")

    print("-" * 75)
    avg_pipe_d125 = np.mean(results_pipeline["d125"])
    avg_gt_d125 = np.mean(results_gt["d125"])
    avg_pipe_mae = np.mean(results_pipeline["mae"])
    avg_gt_mae = np.mean(results_gt["mae"])

    print(f"{'평균':<8} | {avg_pipe_d125:>13.1f}% | {avg_gt_d125:>13.1f}% | {avg_pipe_mae:>10.2f}m | {avg_gt_mae:>8.2f}m")
    print()

    # 결론
    print("=" * 70)
    print("결과 요약")
    print("=" * 70)
    print(f"GT Pose 삼각측량 평균 d<1.25: {avg_gt_d125:.1f}%")
    print(f"Pipeline 삼각측량 평균 d<1.25: {avg_pipe_d125:.1f}%")
    print()

    if avg_gt_d125 >= 80:
        print("✓ GT Pose 삼각측량 정확도 80% 이상 달성!")
    else:
        print("✗ GT Pose 삼각측량 정확도 80% 미달 - 수식 확인 필요")

    if avg_pipe_d125 >= 70:
        print("✓ Pipeline 삼각측량 정확도 양호")
    else:
        print("⚠ Pipeline 삼각측량 정확도 저조 - motion estimation 확인 필요")


def main():
    parser = argparse.ArgumentParser(description="파이프라인 삼각측량 테스트")
    parser.add_argument("--date", default="2011_10_03", help="KITTI 날짜")
    parser.add_argument("--drive", default="0027", help="KITTI 드라이브")
    parser.add_argument("--start-frame", type=int, default=10, help="시작 프레임")
    parser.add_argument("--num-frames", type=int, default=10, help="프레임 수")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 출력")

    args = parser.parse_args()
    test_pipeline_triangulation(
        args.date, args.drive, args.start_frame, args.num_frames, args.verbose
    )


if __name__ == "__main__":
    main()
