cmake_minimum_required(VERSION 3.18.1)
project(pr_depth_android)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Android-specific flags
if(ANDROID)
    message(STATUS "Building for Android ABI: ${ANDROID_ABI}")

    # Find Android log library
    find_library(log-lib log)
endif()

# ============================================================
# QNN Depth Estimator (GPU backend, no OpenCV/Eigen dependency)
# ============================================================
add_library(qnn_depth_jni SHARED
    qnn_depth_estimator.cpp
    qnn_jni_bridge.cpp
)

target_include_directories(qnn_depth_jni PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/qnn_include
)

target_link_libraries(qnn_depth_jni
    ${log-lib}
    android
)

# Always optimize native code (even debug) — critical for YUV preprocessing loop
target_compile_options(qnn_depth_jni PRIVATE -O3 -ffast-math)

message(STATUS "QNN Depth Estimator build configured")

# ============================================================
# PR-Depth core + JNI (OpenCV + Eigen)
# ============================================================
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../opencv-android-sdk/sdk/native/jni")
find_package(OpenCV REQUIRED COMPONENTS core imgproc video)
message(STATUS "OpenCV found: ${OpenCV_VERSION} libs=${OpenCV_LIBS}")

set(EIGEN3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/eigen3")
set(PR_DEPTH_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../cpp/include")
set(PR_DEPTH_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../cpp/src")

# PR-Depth core library (optical flow, motion estimation, triangulation, fusion)
add_library(pr_depth_core STATIC
    ${PR_DEPTH_SRC_DIR}/optical_flow.cpp
    ${PR_DEPTH_SRC_DIR}/motion_field.cpp
    ${PR_DEPTH_SRC_DIR}/triangulation.cpp
    ${PR_DEPTH_SRC_DIR}/depth_fusion.cpp
    ${PR_DEPTH_SRC_DIR}/depth_refinement.cpp
)

target_include_directories(pr_depth_core PUBLIC
    ${PR_DEPTH_INCLUDE_DIR}
    ${EIGEN3_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(pr_depth_core
    ${OpenCV_LIBS}
    ${log-lib}
)

target_compile_options(pr_depth_core PRIVATE -O3 -ffast-math)

# PR-Depth JNI bridge
add_library(pr_depth_jni SHARED
    jni_bridge.cpp
)

target_include_directories(pr_depth_jni PRIVATE
    ${PR_DEPTH_INCLUDE_DIR}
    ${EIGEN3_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(pr_depth_jni
    pr_depth_core
    ${OpenCV_LIBS}
    ${log-lib}
    android
)

target_compile_options(pr_depth_jni PRIVATE -O3 -ffast-math)

message(STATUS "PR-Depth build configured with OpenCV ${OpenCV_VERSION}")
