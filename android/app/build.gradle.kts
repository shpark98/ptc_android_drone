plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.ptcdepth.android"
    compileSdk = 34

    // NDK r27+ links native .so with 16 KB-aligned LOAD segments by default,
    // required for Android 15+/16 devices with 16 KB memory pages (SM8850).
    ndkVersion = "27.2.12479018"

    defaultConfig {
        applicationId = "com.ptcdepth.android"
        minSdk = 26  // ARCore minimum requirement
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters += listOf("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DANDROID_PLATFORM=android-26"
                )
                cppFlags += listOf("-std=c++17", "-O3", "-ffast-math")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        viewBinding = true
    }

    sourceSets {
        getByName("main") {
            jniLibs.srcDirs("libs")
        }
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }

    // Don't compress the ONNX graph or its external weights so they can be
    // memory-mapped from the APK / extracted to cache without re-inflating.
    androidResources {
        noCompress += listOf("onnx", "data", "bin", "dlc")
        // Exclude experimental model formats the app never loads. The active
        // pipeline uses depth_anything_v2.onnx/.data (QNN) with
        // depth_anything.onnx as the CPU fallback; the rest are leftovers that
        // bloat the APK by ~527 MB and make USB installs slow/flaky.
        ignoreAssetsPatterns += listOf(
            "*.tflite",
            "*.dlc",
            "depth_anything_vits.onnx",
            "depth_anything_v2_s25.bin",
        )
    }
}

dependencies {
    // ARCore — 1.48+ ships 16 KB page-aligned native libraries
    implementation("com.google.ar:core:1.48.0")

    // CameraX removed: the app drives the camera through ARCore (GL), never
    // through androidx.camera. Its libimage_processing_util_jni.so was 4 KB
    // aligned and only added to the 16 KB compatibility warning.

    // ONNX Runtime with QNN Execution Provider — matches AI Hub's published
    // 22.8 ms benchmark setup. Uses our shipped libQnnHtp.so + Hexagon V79
    // skel from app/libs/arm64-v8a/.
    // Use the same ORT version AI Hub reports for the 22.8 ms benchmark.
    implementation("com.microsoft.onnxruntime:onnxruntime-android-qnn:1.24.3")

    // Kotlin & AndroidX
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")
}
