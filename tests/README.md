# PR-Depth 테스트

이 폴더에는 PR-Depth 파이프라인의 각 기능을 테스트하기 위한 코드와 데이터가 포함되어 있습니다.

## 폴더 구조

```
tests/
├── data/                     # 테스트 데이터
│   ├── kitti_sample/         # KITTI 샘플 데이터 (10 프레임)
│   │   ├── rgb/              # RGB 이미지 (PNG)
│   │   ├── depth_gt/         # GT depth (NPY)
│   │   ├── flow/             # Optical flow (NPY)
│   │   ├── pose/             # GT pose (JSON)
│   │   └── metadata.json     # 메타데이터
│   └── generate_test_data.py # 테스트 데이터 생성 스크립트
│
├── triangulation/            # 삼각측량 테스트
│   ├── test_triangulation.py # 삼각측량 수식 테스트
│   └── test_pipeline.py      # 파이프라인 통합 테스트
│
└── README.md                 # 이 파일
```

## C++ 빌드 방법

### 요구사항

- CMake 3.15+
- C++17 지원 컴파일러 (GCC 7+, Clang 5+)
- OpenCV 4.x
- Eigen 3.3+
- pybind11
- OpenMP (선택사항, 성능 향상)

### 빌드 명령어

```bash
# 프로젝트 루트에서
cd cpp
mkdir -p build && cd build

# 릴리즈 빌드 (권장)
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 디버그 빌드
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

### 빌드 확인

```bash
# Python에서 모듈 임포트 확인
python -c "import sys; sys.path.insert(0, 'cpp/build'); import pr_depth_cpp; print('OK')"
```

## 테스트 데이터 생성

```bash
# pr_depth conda 환경 활성화
conda activate pr_depth

# 테스트 데이터 생성 (KITTI 데이터셋 필요)
python tests/data/generate_test_data.py
```

생성되는 데이터:
- RGB 이미지 10개 (frame_000.png ~ frame_009.png)
- GT depth 10개 (frame_000.npy ~ frame_009.npy)
- Optical flow 9개 (flow_000_to_001.npy ~ flow_008_to_009.npy)
- Pose 정보 10개 (frame_000.json ~ frame_009.json)

## 테스트 실행

### 삼각측량 수식 테스트

```bash
# 정확한 수식 테스트 (권장)
python tests/triangulation/test_triangulation.py --formula correct

# 모든 수식 비교 테스트
python tests/triangulation/test_triangulation.py --formula all

# 상세 출력
python tests/triangulation/test_triangulation.py --formula correct --verbose
```

#### 수식 설명

| 수식 | R_tri | C1 | 설명 |
|------|-------|-----|------|
| **correct** | R^T | -R^T @ t | 정확한 수식 (92%+ 정확도) |
| current | R | -R^T @ t | 버그 있는 수식 (71% 정확도) |
| t_only | R | t | 잘못된 수식 |
| neg_t | R | -t | 잘못된 수식 |

#### 기대 결과

```
수식별 비교 결과
================================================================================
수식              | d<1.05   | d<1.15   | d<1.25   | MAE      | RMSE
--------------------------------------------------------------------------------
correct         |   38.5% |   82.8% |   92.2% |   2.19m |   8.94m  <- 권장
current         |   23.7% |   56.5% |   73.7% |   4.41m |  13.74m
t_only          |    3.2% |    8.5% |   12.9% |  39.22m |  55.63m
neg_t           |   22.0% |   55.1% |   72.9% |   4.48m |  13.86m
--------------------------------------------------------------------------------
```

### 파이프라인 통합 테스트

```bash
# 기본 테스트
python tests/triangulation/test_pipeline.py

# 다른 시퀀스 테스트
python tests/triangulation/test_pipeline.py --date 2011_09_26 --drive 0001

# 더 많은 프레임 테스트
python tests/triangulation/test_pipeline.py --num-frames 20
```

## 핵심 발견 사항

### 삼각측량 R,t 변환 규칙

Motion field와 삼각측량은 서로 다른 좌표계 규약을 사용합니다:

**Motion field 출력:**
```
p_curr = R @ p_prev + t  (R은 prev → curr 변환)
```

**삼각측량 기대:**
```
r1_in_frame0 = R_tri^T @ r1_in_frame1  (R_tri^T는 frame1 → frame0 변환)
```

**변환 규칙:**
```cpp
// Motion field R을 삼각측량 R로 변환
R_for_tri = R_motion.transpose();  // R^T

// 카메라 위치 계산
C1 = -R_for_tri * (t_motion * baseline);  // -R^T @ t
```

### 정확도 목표

- GT pose 삼각측량: **δ<1.25 > 80%**
- Pipeline 삼각측량: **δ<1.25 > 70%** (motion estimation 품질에 따라 다름)

## 문제 해결

### 모듈 import 오류

```python
ModuleNotFoundError: No module named 'pr_depth_cpp'
```

해결:
```bash
# Python 경로에 빌드 디렉토리 추가
export PYTHONPATH=/path/to/pr_depth/cpp/build:$PYTHONPATH

# 또는 코드에서
import sys
sys.path.insert(0, '/path/to/pr_depth/cpp/build')
```

### 데이터 경로 오류

```
FileNotFoundError: [Errno 2] No such file or directory: 'tests/data/kitti_sample/metadata.json'
```

해결:
1. 절대 경로 사용
2. 또는 프로젝트 루트에서 실행

### Motion estimation 실패

```
Motion estimation failed: Not enough valid points
```

원인:
- 입력 inv_depth가 상수 (예: 0.1)인 경우
- 유효한 depth prior가 필요함

해결:
- DepthAnything 등의 네트워크로 생성한 depth prior 사용
- GT depth를 prior로 사용 (테스트용)

## 관련 파일

- `cpp/src/triangulation.cpp`: 삼각측량 구현
- `cpp/src/depth_refinement.cpp`: 파이프라인 메인 로직
- `cpp/src/motion_field.cpp`: Motion field 추정
- `dataloader/dataset/base.py`: GT pose 로딩
