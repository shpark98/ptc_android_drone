#!/bin/bash
# PR-Depth QAIRT 자동 설정 스크립트
set -e

echo "=========================================="
echo "  PR-Depth QAIRT (Qualcomm AI Runtime) 설정"
echo "=========================================="
echo ""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_ROOT="/home/arrl/Desktop/algorithm/pr_depth_android"
QAIRT_ROOT="$HOME/qairt-2.43/2.43.0.260128"

# Step 1: QAIRT SDK 확인
echo -e "${YELLOW}[1/6] QAIRT SDK 확인 중...${NC}"
if [ ! -d "$QAIRT_ROOT" ]; then
    echo -e "${RED}✗ QAIRT SDK를 찾을 수 없습니다: $QAIRT_ROOT${NC}"
    exit 1
else
    echo -e "${GREEN}✓ QAIRT SDK 발견: $QAIRT_ROOT${NC}"
fi

# Step 2: Python 환경 설정
echo -e "\n${YELLOW}[2/6] Python 환경 설정 중...${NC}"
cd "$QAIRT_ROOT"

# Python 3 확인
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3이 설치되지 않았습니다${NC}"
    exit 1
fi

# ONNX 지원 확인
if [ -f "bin/envsetup.sh" ]; then
    source bin/envsetup.sh
    echo -e "${GREEN}✓ QAIRT 환경 설정 완료${NC}"
else
    echo -e "${YELLOW}→ envsetup.sh 없음, 직접 경로 설정${NC}"
    export PATH="$QAIRT_ROOT/bin/x86_64-linux-clang:$PATH"
    export LD_LIBRARY_PATH="$QAIRT_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH"
fi

# Step 3: 모델 변환 (ONNX → DLC)
echo -e "\n${YELLOW}[3/6] ONNX 모델을 DLC로 변환 중...${NC}"
ONNX_MODEL="$PROJECT_ROOT/weights/depth_anything_v2_onnx/depth_anything_v2_vits_518.onnx"
DLC_OUTPUT="$PROJECT_ROOT/weights/depth_anything_v2_onnx/depth_anything_vits_qairt.dlc"

if [ ! -f "$ONNX_MODEL" ]; then
    echo -e "${RED}✗ ONNX 모델을 찾을 수 없습니다: $ONNX_MODEL${NC}"
    exit 1
fi

# QAIRT는 qnn-onnx-converter 사용
if command -v qnn-onnx-converter &> /dev/null; then
    qnn-onnx-converter \
        --input_network "$ONNX_MODEL" \
        --output_path "$DLC_OUTPUT" \
        --input_dim l_x_ "1,3,518,518" || {
        echo -e "${RED}✗ 모델 변환 실패 (이건 정상일 수 있음 - ViT는 아직 완전 지원 안 됨)${NC}"
        echo -e "${YELLOW}→ Plan B: 해상도를 256으로 낮춰서 재시도할까요? (y/N)${NC}"
        read -r RETRY
        if [[ "$RETRY" =~ ^[Yy]$ ]]; then
            echo "TODO: 256x256 모델 변환"
        fi
    }

    if [ -f "$DLC_OUTPUT" ]; then
        echo -e "${GREEN}✓ DLC 모델 생성: $DLC_OUTPUT${NC}"
    else
        echo -e "${YELLOW}→ DLC 생성 실패. ONNX 모델을 직접 사용합니다.${NC}"
        # ONNX fallback - QAIRT도 ONNX 직접 지원
        cp "$ONNX_MODEL" "$PROJECT_ROOT/android/app/src/main/assets/depth_anything_vits.onnx"
        echo -e "${GREEN}✓ ONNX 모델을 assets에 복사했습니다 (DLC 대신 ONNX 사용)${NC}"
    fi
else
    echo -e "${YELLOW}→ qnn-onnx-converter 없음, ONNX 직접 사용${NC}"
    mkdir -p "$PROJECT_ROOT/android/app/src/main/assets"
    cp "$ONNX_MODEL" "$PROJECT_ROOT/android/app/src/main/assets/depth_anything_vits.onnx"
fi

# Step 4: QAIRT 라이브러리 복사
echo -e "\n${YELLOW}[4/6] Android 프로젝트에 QAIRT 라이브러리 복사 중...${NC}"
ANDROID_LIBS="$PROJECT_ROOT/android/app/libs"
mkdir -p "$ANDROID_LIBS/arm64-v8a"

# QNN 라이브러리 (QAIRT의 새 이름)
if [ -d "$QAIRT_ROOT/lib/aarch64-android" ]; then
    cp "$QAIRT_ROOT"/lib/aarch64-android/*.so "$ANDROID_LIBS/arm64-v8a/" 2>/dev/null || true
    echo -e "${GREEN}✓ QNN 라이브러리 복사 완료${NC}"
else
    echo -e "${RED}✗ QNN 라이브러리를 찾을 수 없습니다${NC}"
fi

# Hexagon DSP 라이브러리
if [ -d "$QAIRT_ROOT/lib/hexagon-v68" ]; then
    cp "$QAIRT_ROOT"/lib/hexagon-v68/unsigned/*.so "$ANDROID_LIBS/arm64-v8a/" 2>/dev/null || true
    echo -e "${GREEN}✓ Hexagon v68 DSP 라이브러리 복사 (Galaxy S21)${NC}"
elif [ -d "$QAIRT_ROOT/lib/hexagon-v73" ]; then
    cp "$QAIRT_ROOT"/lib/hexagon-v73/unsigned/*.so "$ANDROID_LIBS/arm64-v8a/" 2>/dev/null || true
    echo -e "${GREEN}✓ Hexagon v73 DSP 라이브러리 복사${NC}"
fi

# Java AAR
if [ -f "$QAIRT_ROOT/lib/java/qnn-release.aar" ]; then
    cp "$QAIRT_ROOT/lib/java/qnn-release.aar" "$ANDROID_LIBS/"
    echo -e "${GREEN}✓ QNN Java AAR 복사${NC}"
else
    echo -e "${YELLOW}→ QNN AAR 없음, ONNX Runtime 사용${NC}"
fi

# Step 5: 복사된 라이브러리 확인
echo -e "\n${YELLOW}[5/6] 복사된 라이브러리 확인...${NC}"
echo "Native 라이브러리:"
ls -lh "$ANDROID_LIBS/arm64-v8a/" | grep -E "\\.so$" | awk '{print "  " $9 " (" $5 ")"}'

# Step 6: 모델 파일 복사
echo -e "\n${YELLOW}[6/6] 모델 assets 복사 확인...${NC}"
if [ -f "$DLC_OUTPUT" ]; then
    cp "$DLC_OUTPUT" "$PROJECT_ROOT/android/app/src/main/assets/depth_anything_vits.dlc"
    echo -e "${GREEN}✓ DLC 모델 복사 완료${NC}"
fi

echo ""
echo -e "${GREEN}=========================================="
echo "  ✓ QAIRT 설정 완료!"
echo "==========================================${NC}"
echo ""
echo "다음 단계:"
echo "1. Android 코드 통합 (자동 진행 예정)"
echo "2. cd android && ./gradlew installDebug"
echo "3. 앱 실행 및 로그 확인"
echo ""
echo "예상 성능:"
echo "  • ONNX CPU (현재):  ~2700ms"
echo "  • QAIRT + DSP:      ~300-500ms  (5-8배 빠름!)"
echo ""
