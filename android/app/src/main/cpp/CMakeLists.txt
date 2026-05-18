cmake_minimum_required(VERSION 3.18.1)
project(ptc_depth_android)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(ANDROID)
    message(STATUS "Building for Android ABI: ${ANDROID_ABI}")
    find_library(log-lib log)
endif()

# ============================================================
# qnn_depth_jni — preprocessing helpers for the ORT-QNN code path
# (inference lives in ONNX Runtime / Java).
# ============================================================
add_library(qnn_depth_jni SHARED
    qnn_jni_bridge.cpp
)

target_include_directories(qnn_depth_jni PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(qnn_depth_jni
    ${log-lib}
    android
)

target_compile_options(qnn_depth_jni PRIVATE -O3 -ffast-math)

message(STATUS "qnn_depth_jni build configured (preprocessing helpers only)")

# ============================================================
# PTC-Depth core + JNI (OpenCV + Eigen)
# ============================================================
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../opencv-android-sdk/sdk/native/jni")
find_package(OpenCV REQUIRED COMPONENTS core imgproc video)
message(STATUS "OpenCV found: ${OpenCV_VERSION} libs=${OpenCV_LIBS}")

set(EIGEN3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/eigen3")
set(PTC_DEPTH_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../cpp/include")
set(PTC_DEPTH_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../cpp/src")

# PTC-Depth core library
add_library(ptc_depth_core STATIC
    ${PTC_DEPTH_SRC_DIR}/motion_field.cpp
    ${PTC_DEPTH_SRC_DIR}/triangulation.cpp
    ${PTC_DEPTH_SRC_DIR}/scale_estimation.cpp
    ${PTC_DEPTH_SRC_DIR}/depth_warp.cpp
    ${PTC_DEPTH_SRC_DIR}/bayesian_fusion.cpp
    ${PTC_DEPTH_SRC_DIR}/ptc_depth.cpp
)

target_include_directories(ptc_depth_core PUBLIC
    ${PTC_DEPTH_INCLUDE_DIR}
    ${EIGEN3_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ptc_depth_core
    ${OpenCV_LIBS}
    ${log-lib}
)

# OpenMP is available in NDK r21+ via libomp. Adding -fopenmp enables #pragma omp
# directives in the PTC-Depth core (motion field RANSAC, fusion, scale solver).
target_compile_options(ptc_depth_core PRIVATE -O3 -ffast-math -fopenmp -static-openmp)
target_link_options(ptc_depth_core PUBLIC -fopenmp -static-openmp)

# PTC-Depth JNI bridge
add_library(ptc_depth_jni SHARED
    jni_bridge.cpp
)

target_include_directories(ptc_depth_jni PRIVATE
    ${PTC_DEPTH_INCLUDE_DIR}
    ${EIGEN3_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ptc_depth_jni
    ptc_depth_core
    ${OpenCV_LIBS}
    ${log-lib}
    android
)

target_compile_options(ptc_depth_jni PRIVATE -O3 -ffast-math)

message(STATUS "PTC-Depth build configured with OpenCV ${OpenCV_VERSION}")
