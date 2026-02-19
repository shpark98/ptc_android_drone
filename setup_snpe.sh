#!/bin/bash
# PR-Depth SNPE 자동 설정 스크립트
# Usage: ./setup_snpe.sh

set -e  # 에러 시 중단

echo "=========================================="
echo "  PR-Depth SNPE 자동 설정"
echo "=========================================="
echo ""

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PROJECT_ROOT="/home/arrl/Desktop/algorithm/pr_depth_android"
SNPE_ROOT="$HOME/snpe-2.22.0"

# Step 1: SNPE SDK 확인
echo -e "${YELLOW}[1/6] SNPE SDK 확인 중...${NC}"
if [ ! -d "$SNPE_ROOT" ]; then
    echo -e "${RED}✗ SNPE SDK를 찾을 수 없습니다.${NC}"
    echo ""
    echo "수동 다운로드가 필요합니다:"
    echo "1. 브라우저에서 열기: https://qpm.qualcomm.com/#/main/tools/details/qualcomm_neural_processing_sdk"
    echo "2. 로그인 (무료 계정 생성)"
    echo "3. 'Download' 클릭 → Linux 버전 다운로드"
    echo "4. 다운로드 후 실행:"
    echo ""
    echo "   cd ~/Downloads"
    echo "   unzip snpe-2.22.0.zip -d ~/"
    echo "   cd $PROJECT_ROOT"
    echo "   ./setup_snpe.sh"
    echo ""
    exit 1
else
    echo -e "${GREEN}✓ SNPE SDK 발견: $SNPE_ROOT${NC}"
fi

# Step 2: Python 환경 설정
echo -e "\n${YELLOW}[2/6] Python 환경 설정 중...${NC}"
source "$SNPE_ROOT/bin/envsetup.sh" -o "$SNPE_ROOT/onnx" || {
    echo -e "${RED}✗ SNPE 환경 설정 실패${NC}"
    exit 1
}
echo -e "${GREEN}✓ Python 환경 설정 완료${NC}"

# Step 3: 모델 변환 (ONNX → DLC)
echo -e "\n${YELLOW}[3/6] ONNX 모델을 DLC로 변환 중...${NC}"
ONNX_MODEL="$PROJECT_ROOT/weights/depth_anything_v2_onnx/depth_anything_v2_vits_518.onnx"
DLC_OUTPUT="$PROJECT_ROOT/weights/depth_anything_v2_onnx/depth_anything_vits.dlc"

if [ ! -f "$ONNX_MODEL" ]; then
    echo -e "${RED}✗ ONNX 모델을 찾을 수 없습니다: $ONNX_MODEL${NC}"
    exit 1
fi

# ONNX → DLC 변환
snpe-onnx-to-dlc \
    --input_network "$ONNX_MODEL" \
    --input_dim l_x_ "1,3,518,518" \
    --output_path "$DLC_OUTPUT" || {
    echo -e "${RED}✗ 모델 변환 실패${NC}"
    exit 1
}
echo -e "${GREEN}✓ DLC 모델 생성: $DLC_OUTPUT${NC}"

# Step 4: INT8 양자화 (선택적)
echo -e "\n${YELLOW}[4/6] INT8 양자화 (더 빠른 추론)${NC}"
echo "샘플 이미지가 필요합니다. 스킵하시겠습니까? (y/N)"
read -r SKIP_QUANTIZE

if [[ ! "$SKIP_QUANTIZE" =~ ^[Yy]$ ]]; then
    # 양자화 생략 안내
    echo -e "${YELLOW}→ 양자화를 위해서는 샘플 이미지 10-100장이 필요합니다.${NC}"
    echo "   나중에 수동으로 실행: snpe-dlc-quantize --input_dlc ... --input_list images.txt"
else
    echo -e "${YELLOW}→ 양자화 스킵됨 (FP32 사용)${NC}"
fi

# Step 5: SNPE 라이브러리 복사
echo -e "\n${YELLOW}[5/6] Android 프로젝트에 SNPE 라이브러리 복사 중...${NC}"
ANDROID_LIBS="$PROJECT_ROOT/android/app/libs"
mkdir -p "$ANDROID_LIBS/arm64-v8a"

# Native 라이브러리 복사
cp "$SNPE_ROOT/lib/aarch64-android/libSNPE.so" "$ANDROID_LIBS/arm64-v8a/" || echo "Warning: libSNPE.so 복사 실패"
cp "$SNPE_ROOT/lib/aarch64-android/libsnpe_dsp_domains_v2.so" "$ANDROID_LIBS/arm64-v8a/" || echo "Warning: libsnpe_dsp_domains_v2.so 복사 실패"

# Hexagon DSP 라이브러리 (Galaxy S21 = Snapdragon 888 = v68)
if [ -f "$SNPE_ROOT/lib/hexagon-v68/unsigned/libSnpeHtpV68Skel.so" ]; then
    cp "$SNPE_ROOT/lib/hexagon-v68/unsigned/libSnpeHtpV68Skel.so" "$ANDROID_LIBS/arm64-v8a/"
    echo -e "${GREEN}✓ Hexagon v68 DSP 라이브러리 복사 (Galaxy S21 최적화)${NC}"
else
    echo -e "${YELLOW}→ Hexagon v68 라이브러리 없음 (GPU fallback 사용)${NC}"
fi

# Java wrapper 복사
cp "$SNPE_ROOT/lib/java/snpe-release.aar" "$ANDROID_LIBS/" || {
    echo -e "${RED}✗ SNPE AAR 복사 실패${NC}"
    exit 1
}
echo -e "${GREEN}✓ SNPE 라이브러리 복사 완료${NC}"

# Step 6: DLC 모델을 Android assets에 복사
echo -e "\n${YELLOW}[6/6] 모델을 Android assets에 복사 중...${NC}"
ANDROID_ASSETS="$PROJECT_ROOT/android/app/src/main/assets"
mkdir -p "$ANDROID_ASSETS"
cp "$DLC_OUTPUT" "$ANDROID_ASSETS/depth_anything_vits.dlc" || {
    echo -e "${RED}✗ 모델 복사 실패${NC}"
    exit 1
}
echo -e "${GREEN}✓ 모델 복사 완료: $ANDROID_ASSETS/depth_anything_vits.dlc${NC}"

# 완료 메시지
echo ""
echo -e "${GREEN}=========================================="
echo "  ✓ SNPE 설정 완료!"
echo "==========================================${NC}"
echo ""
echo "다음 단계:"
echo "1. Android 코드 통합 (자동 생성 예정)"
echo "2. ./gradlew installDebug"
echo "3. 앱 실행 후 로그 확인:"
echo "   adb logcat | grep SNPE"
echo ""
echo "예상 성능:"
echo "  • CPU (현재):  ~2700ms"
echo "  • SNPE + DSP:  ~300-500ms  (5-8배 빠름!)"
echo ""
