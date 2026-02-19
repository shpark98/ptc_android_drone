"""삼각측량 테스트 코드.

저장된 테스트 데이터를 사용하여 삼각측량 정확도를 평가합니다.
GT pose와 GT optical flow를 사용하여 삼각측량의 순수 성능을 측정합니다.

사용법:
    python tests/triangulation/test_triangulation.py
    python tests/triangulation/test_triangulation.py --formula all  # 모든 수식 테스트
    python tests/triangulation/test_triangulation.py --verbose       # 상세 출력
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'cpp/build'))

import argparse
import json
import numpy as np
import cv2
from pathlib import Path

import pr_depth_cpp as cpp
from src.evaluation import compute_depth_metrics


def load_test_data(data_dir: str = "tests/data/kitti_sample"):
    """테스트 데이터 로드.

    Args:
        data_dir: 테스트 데이터 디렉토리

    Returns:
        dict: 메타데이터와 데이터 경로 정보
    """
    data_path = Path(data_dir)

    # 메타데이터 로드
    with open(data_path / "metadata.json", 'r') as f:
        metadata = json.load(f)

    return metadata, data_path


def load_frame_data(data_path: Path, idx: int):
    """개별 프레임 데이터 로드.

    Args:
        data_path: 데이터 디렉토리 경로
        idx: 프레임 인덱스

    Returns:
        dict: 프레임 데이터 (rgb, depth_gt, pose)
    """
    # RGB 이미지
    rgb_path = data_path / "rgb" / f"frame_{idx:03d}.png"
    rgb = cv2.imread(str(rgb_path))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # GT depth
    depth_path = data_path / "depth_gt" / f"frame_{idx:03d}.npy"
    depth_gt = np.load(str(depth_path))

    # Pose 정보
    pose_path = data_path / "pose" / f"frame_{idx:03d}.json"
    with open(str(pose_path), 'r') as f:
        pose = json.load(f)

    return {
        "rgb": rgb,
        "depth_gt": depth_gt,
        "baseline": pose["baseline"],
        "R": np.array(pose["R"]),
        "t": np.array(pose["t"]),
    }


def load_flow(data_path: Path, from_idx: int, to_idx: int):
    """Optical flow 로드.

    Args:
        data_path: 데이터 디렉토리 경로
        from_idx: 시작 프레임 인덱스
        to_idx: 종료 프레임 인덱스

    Returns:
        np.ndarray: Optical flow (H, W, 2)
    """
    flow_path = data_path / "flow" / f"flow_{from_idx:03d}_to_{to_idx:03d}.npy"
    return np.load(str(flow_path))


def triangulate_with_formula(
    flow: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    baseline: float,
    fx: float, fy: float, cx: float, cy: float,
    H: int, W: int,
    formula: str = "correct"
):
    """주어진 수식으로 삼각측량 수행.

    Args:
        flow: Optical flow (H, W, 2)
        R: 회전 행렬 (motion field convention: p_curr = R @ p_prev + t)
        t: 단위 변환 벡터
        baseline: 기준선 길이
        fx, fy, cx, cy: 카메라 내부 파라미터
        H, W: 이미지 크기
        formula: 삼각측량 수식
            - "correct": R^T, -R^T @ t (정확한 수식, 91%+ 정확도)
            - "current": R, -R^T @ t (현재 버그가 있는 수식, 71% 정확도)
            - "t_only": R, t (잘못된 수식)
            - "neg_t": R, -t (잘못된 수식)

    Returns:
        dict: 삼각측량 결과 (z1_tri, num_valid)
    """
    # 픽셀 좌표 생성
    u0 = np.tile(np.arange(W, dtype=np.float32), (H, 1))
    v0 = np.tile(np.arange(H, dtype=np.float32).reshape(-1, 1), (1, W))
    u1 = u0 + flow[:, :, 0]
    v1 = v0 + flow[:, :, 1]

    # 수식에 따른 R, C1 계산
    if formula == "correct":
        # 정확한 수식: R_tri = R, C1 = -R^T @ t (삼각측량이 R 직접 사용)
        R_tri = R
        C1 = -R.T @ (t * baseline)
    elif formula == "current":
        # 이전 (버그) 수식: R_tri = R^T, C1 = -R^T @ t
        R_tri = R.T
        C1 = -R.T @ (t * baseline)
    elif formula == "t_only":
        R_tri = R
        C1 = t * baseline
    elif formula == "neg_t":
        R_tri = R
        C1 = -(t * baseline)
    else:
        raise ValueError(f"Unknown formula: {formula}")

    # C++ 삼각측량 호출
    result = cpp.triangulate_depth(
        u0.flatten().astype(np.float32),
        v0.flatten().astype(np.float32),
        u1.flatten().astype(np.float32),
        v1.flatten().astype(np.float32),
        R_tri, C1, fx, fy, cx, cy, H, W
    )

    return result


def test_triangulation(
    data_dir: str = "tests/data/kitti_sample",
    formula: str = "correct",
    verbose: bool = False,
):
    """삼각측량 테스트 실행.

    Args:
        data_dir: 테스트 데이터 디렉토리
        formula: 테스트할 삼각측량 수식
        verbose: 상세 출력 여부

    Returns:
        dict: 테스트 결과
    """
    metadata, data_path = load_test_data(data_dir)
    num_frames = metadata["num_frames"]
    H, W = metadata["H"], metadata["W"]
    fx, fy = metadata["fx"], metadata["fy"]
    cx, cy = metadata["cx"], metadata["cy"]

    print(f"=" * 70)
    print(f"삼각측량 테스트 - 수식: {formula}")
    print(f"=" * 70)
    print(f"데이터: {data_dir}")
    print(f"프레임 수: {num_frames}")
    print(f"이미지 크기: {H}x{W}")
    print(f"카메라 파라미터: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    print()

    results = {
        "formula": formula,
        "frames": [],
        "d105": [],
        "d115": [],
        "d125": [],
        "mae": [],
        "rmse": [],
    }

    for i in range(1, num_frames):
        # 데이터 로드
        prev_data = load_frame_data(data_path, i - 1)
        curr_data = load_frame_data(data_path, i)
        flow = load_flow(data_path, i - 1, i)

        # 삼각측량 수행
        tri_result = triangulate_with_formula(
            flow,
            curr_data["R"],
            curr_data["t"],
            curr_data["baseline"],
            fx, fy, cx, cy, H, W,
            formula=formula
        )
        z_tri = tri_result["z1_tri"]

        # 평가
        gt_depth = curr_data["depth_gt"]
        mask = (gt_depth > 0) & (gt_depth < 80) & (z_tri > 0) & (z_tri < 200)

        if mask.sum() > 100:
            metrics = compute_depth_metrics(z_tri, gt_depth)
            if metrics is not None:
                results["frames"].append(i)
                results["d105"].append(metrics.get("d105", 0))
                results["d115"].append(metrics.get("d115", 0))
                results["d125"].append(metrics["d125"])
                results["mae"].append(metrics["MAE"])
                results["rmse"].append(metrics["RMSE"])

                if verbose:
                    print(f"Frame {i:2d}: d<1.25={metrics['d125']:.1f}%, "
                          f"MAE={metrics['MAE']:.2f}m, RMSE={metrics['RMSE']:.2f}m, "
                          f"valid={mask.sum()}")

    # 요약 출력
    print()
    print("-" * 70)
    print(f"{'프레임':<10}", end="")
    for frame in results["frames"]:
        print(f"F{frame:02d}", end="  ")
    print("평균")
    print("-" * 70)

    print(f"{'d<1.25(%)':<10}", end="")
    for d125 in results["d125"]:
        print(f"{d125:3.0f}", end="  ")
    print(f"{np.mean(results['d125']):.1f}")

    print(f"{'MAE(m)':<10}", end="")
    for mae in results["mae"]:
        print(f"{mae:3.1f}", end="  ")
    print(f"{np.mean(results['mae']):.2f}")

    print("-" * 70)
    print()
    print(f"최종 결과:")
    print(f"  d<1.05: {np.mean(results.get('d105', [0])):.1f}%")
    print(f"  d<1.15: {np.mean(results.get('d115', [0])):.1f}%")
    print(f"  d<1.25: {np.mean(results['d125']):.1f}%")
    print(f"  MAE: {np.mean(results['mae']):.2f}m")
    print(f"  RMSE: {np.mean(results['rmse']):.2f}m")

    return results


def test_all_formulas(data_dir: str = "tests/data/kitti_sample"):
    """모든 수식 테스트 및 비교.

    Args:
        data_dir: 테스트 데이터 디렉토리
    """
    formulas = ["correct", "current", "t_only", "neg_t"]

    print("=" * 80)
    print("모든 삼각측량 수식 비교 테스트")
    print("=" * 80)
    print()
    print("수식 설명:")
    print("  - correct: R_tri=R^T, C1=-R^T@t (정확한 수식)")
    print("  - current: R_tri=R, C1=-R^T@t   (현재 버그 있는 수식)")
    print("  - t_only:  R_tri=R, C1=t        (단순 t)")
    print("  - neg_t:   R_tri=R, C1=-t       (부호 반전 t)")
    print()

    all_results = {}
    for formula in formulas:
        print(f"\n>>> {formula} 테스트 중...")
        all_results[formula] = test_triangulation(data_dir, formula)
        print()

    # 비교 테이블 출력
    print()
    print("=" * 80)
    print("수식별 비교 결과")
    print("=" * 80)
    print(f"{'수식':<15} | {'d<1.05':<8} | {'d<1.15':<8} | {'d<1.25':<8} | {'MAE':<8} | {'RMSE':<8}")
    print("-" * 80)
    for formula in formulas:
        r = all_results[formula]
        print(f"{formula:<15} | "
              f"{np.mean(r.get('d105', [0])):>6.1f}% | "
              f"{np.mean(r.get('d115', [0])):>6.1f}% | "
              f"{np.mean(r['d125']):>6.1f}% | "
              f"{np.mean(r['mae']):>6.2f}m | "
              f"{np.mean(r['rmse']):>6.2f}m")
    print("-" * 80)

    # 권장 사항
    best_formula = max(formulas, key=lambda f: np.mean(all_results[f]['d125']))
    print()
    print(f"권장 수식: {best_formula} (d<1.25 = {np.mean(all_results[best_formula]['d125']):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="삼각측량 테스트")
    parser.add_argument("--data-dir", default="tests/data/kitti_sample",
                        help="테스트 데이터 디렉토리")
    parser.add_argument("--formula", default="correct",
                        choices=["correct", "current", "t_only", "neg_t", "all"],
                        help="테스트할 삼각측량 수식")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="상세 출력")

    args = parser.parse_args()

    if args.formula == "all":
        test_all_formulas(args.data_dir)
    else:
        test_triangulation(args.data_dir, args.formula, args.verbose)


if __name__ == "__main__":
    main()
