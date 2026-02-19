# PR-Depth Android 앱 빌드 가이드

안드로이드 핸드폰에서 실시간 depth refinement를 수행하는 앱입니다.

## 🎯 주요 기능

- ✅ **실시간 depth refinement**: 핸드폰 카메라로 실시간 depth 정제
- ✅ **ARCore 통합**: R, t 포즈 자동 추출
- ✅ **GT pose 온/오프**: 앱에서 토글하여 ARCore 포즈 사용 여부 선택
- ✅ **실시간 시각화**: Depth map 컬러맵 표시
- ✅ **성능 통계**: FPS, baseline, rotation 등 실시간 표시

---

## 📋 사전 요구사항

### 1. 시스템 요구사항
- Linux (Ubuntu 20.04+ 권장)
- 8GB+ RAM
- 10GB+ 디스크 공간

### 2. 안드로이드 핸드폰
- **ARCore 지원 기기** ([리스트 확인](https://developers.google.com/ar/devices))
- Android 8.0 (API 26) 이상
- USB 디버깅 활성화

---

## 🚀 빠른 시작 (5단계)

### 1단계: Android SDK 설치

```bash
cd ~
wget https://dl.google.com/android/repository/commandlinetools-linux-9477386_latest.zip
unzip commandlinetools-linux-9477386_latest.zip
mkdir -p ~/Android/Sdk/cmdline-tools
mv cmdline-tools ~/Android/Sdk/cmdline-tools/latest

# 환경 변수 설정
export ANDROID_HOME=~/Android/Sdk
export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools

# .bashrc에 추가
echo 'export ANDROID_HOME=~/Android/Sdk' >> ~/.bashrc
echo 'export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools' >> ~/.bashrc

# SDK 컴포넌트 설치
sdkmanager --update
sdkmanager "platform-tools" "platforms;android-34" "build-tools;34.0.0" "ndk;25.2.9519653" "cmake;3.22.1"

# 라이선스 동의
sdkmanager --licenses
```

### 2단계: 외부 라이브러리 설치

```bash
cd /home/arrl/Desktop/algorithm/pr_depth_android/android
./setup_libs.sh
```

이 스크립트는 자동으로 다운로드합니다:
- OpenCV Android SDK 4.8.0
- Eigen3 3.4.0

### 3단계: Gradle Wrapper 설치

```bash
cd /home/arrl/Desktop/algorithm/pr_depth_android/android

# Gradle wrapper 다운로드 (없는 경우)
if [ ! -f "gradlew" ]; then
    wget https://services.gradle.org/distributions/gradle-8.2-bin.zip
    unzip gradle-8.2-bin.zip
    gradle-8.2/bin/gradle wrapper
    rm -rf gradle-8.2 gradle-8.2-bin.zip
fi

# 실행 권한 부여
chmod +x gradlew
```

### 4단계: 빌드

```bash
# Debug APK 빌드
./gradlew assembleDebug

# 빌드 성공하면:
# app/build/outputs/apk/debug/app-debug.apk 생성됨
```

**예상 빌드 시간**: 첫 빌드 5-10분 (의존성 다운로드 포함)

### 5단계: 핸드폰에 설치

```bash
# 핸드폰 USB 연결 확인
adb devices

# 출력 예시:
# List of devices attached
# ABC123456    device

# APK 설치
adb install -r app/build/outputs/apk/debug/app-debug.apk

# 앱 실행
adb shell am start -n com.prdepth.android/.MainActivity
```

---

## 📱 앱 사용법

### UI 설명

```
┌─────────────────────────────┐
│                             │
│    Depth Visualization      │
│      (Colormap)             │
│                             │
│                             │
├─────────────────────────────┤
│ [✓] Use ARCore Pose (GT)    │
│ FPS: 25.3                   │
│ Baseline: 0.124m            │
│ Rotation: 2.3°              │
│ Matches: 1523 / Valid: 1245 │
└─────────────────────────────┘
```

### GT Pose 토글

- **OFF (기본)**: 파이프라인이 R, t를 직접 추정
- **ON**: ARCore에서 받은 R, t를 GT로 사용
  - 회전 각도 > 3도일 때만 GT 사용 (fallback 모드)

---

## 🐛 디버깅

### 로그 확인

```bash
# PR-Depth 관련 로그만 필터링
adb logcat | grep "PR-Depth"

# 특정 태그만
adb logcat | grep "DepthRefineManager"
adb logcat | grep "ARCoreManager"
```

### 일반적인 문제 해결

**1. "ARCore not available"**
```bash
# ARCore APK 수동 설치
adb install -r arcore.apk

# 또는 Play Store에서 "Google Play Services for AR" 설치
```

**2. "Camera permission denied"**
- 앱 설정 → 권한 → 카메라 활성화

**3. "libpr_depth_jni.so not found"**
```bash
# NDK/CMake가 제대로 설치되었는지 확인
sdkmanager --list | grep ndk
sdkmanager --list | grep cmake

# 재빌드
./gradlew clean
./gradlew assembleDebug
```

**4. 빌드 실패 (OpenCV not found)**
```bash
# setup_libs.sh 다시 실행
cd /home/arrl/Desktop/algorithm/pr_depth_android/android
rm -rf app/libs
./setup_libs.sh
```

---

## 📊 성능 최적화

현재 설정 (실시간 모드):
- RANSAC iterations: 50
- Forward-backward consistency: OFF
- Iterative refinement: OFF

**목표 성능**:
- 480p: 20-30 FPS
- 720p: 15-20 FPS

**메모리 사용량**: ~500MB (OpenCV 포함)

---

## 🔧 고급 설정

### C++ 설정 변경

[jni_bridge.cpp:143](android/app/src/main/cpp/jni_bridge.cpp#L143)에서 수정:

```cpp
// 실시간 모드 비활성화 (더 높은 정확도)
config.set_realtime_mode(false);

// Forward-backward consistency 활성화
config.skip_fb_consistency = false;

// Iterative refinement 활성화
config.enable_iterative_refinement = true;
```

### 해상도 변경

MainActivity.kt에서 카메라 해상도 설정 가능 (TODO: 아직 미구현)

---

## 📂 프로젝트 구조

```
android/
├── app/
│   ├── src/main/
│   │   ├── java/com/prdepth/android/
│   │   │   ├── MainActivity.kt              # 메인 UI
│   │   │   ├── DepthRefinementManager.kt    # C++ 파이프라인 래퍼
│   │   │   ├── ARCoreManager.kt             # ARCore 포즈 추출
│   │   │   ├── DepthVisualizerView.kt       # Depth 시각화
│   │   │   └── DepthResult.kt               # 결과 데이터
│   │   ├── cpp/
│   │   │   ├── jni_bridge.cpp               # JNI 브릿지
│   │   │   └── CMakeLists.txt               # Native 빌드 설정
│   │   ├── res/layout/
│   │   │   └── activity_main.xml            # UI 레이아웃
│   │   └── AndroidManifest.xml
│   ├── libs/
│   │   ├── opencv-android-sdk/              # OpenCV (setup_libs.sh로 설치)
│   │   └── eigen3/                          # Eigen3 (setup_libs.sh로 설치)
│   └── build.gradle.kts
├── build.gradle.kts
├── settings.gradle.kts
└── setup_libs.sh                            # 라이브러리 설치 스크립트
```

---

## 🚧 TODO (향후 개선사항)

- [ ] **Depth Anything 통합**: 모노큘러 depth 추정 모델 추가
- [ ] **데이터 녹화 기능**: 프레임, depth, pose 저장
- [ ] **설정 UI**: 해상도, 성능 모드 선택
- [ ] **카메라 프리뷰**: ARCore 카메라 이미지 표시
- [ ] **GPU 가속**: OpenCL 또는 Vulkan 활용

---

## 📖 참고 자료

- [ARCore 개발 가이드](https://developers.google.com/ar/develop)
- [Android NDK 가이드](https://developer.android.com/ndk/guides)
- [OpenCV Android](https://opencv.org/android/)
- [Eigen3 문서](https://eigen.tuxfamily.org/)

---

## 🆘 문제 발생 시

1. **GitHub Issues**: https://github.com/your-repo/pr_depth_android/issues
2. **로그 수집**:
```bash
adb logcat > pr_depth_log.txt
```
3. **시스템 정보**:
```bash
adb shell getprop ro.build.version.release  # Android 버전
adb shell getprop ro.product.model          # 기기 모델
```

---

## 📄 라이선스

PR-Depth 프로젝트 라이선스를 따름.
