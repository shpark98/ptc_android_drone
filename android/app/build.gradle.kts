plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.ptcdepth.android"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    // NDK r27+ links native .so with 16 KB-aligned LOAD segments by default,
    // required for Android 15+/16 devices with 16 KB memory pages (SM8850).
    ndkVersion = "27.2.12479018"

    defaultConfig {
        applicationId = "com.ptcdepth.android"
        // FLIR Atlas Android SDK 2.22.0 requires Android 13 (API 33)+.
        minSdk = 33
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters += listOf("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                arguments += listOf(
                    // FLIR Atlas 2.22 ships a newer 16 KB-aligned libc++_shared.
                    // Link our isolated JNI modules statically so Gradle packages
                    // the FLIR runtime required by libatlas_native.so.
                    "-DANDROID_STL=c++_static",
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

    androidResources {
        // Keep model files uncompressed so runtime extraction/mapping does not
        // need to inflate them first.
        noCompress += listOf("onnx", "data", "bin", "dlc")
        // Exclude only unused experimental model formats.
        ignoreAssetsPatterns += listOf(
            "*.tflite",
            "*.dlc",
            "depth_anything_vits.onnx",
            "depth_anything_v2_s25.bin",
        )
    }
}

dependencies {
    // FLIR Atlas Android SDK 2.22.0 (provisioned locally; AARs are git-ignored).
    implementation(files("libs/thermalsdk-release.aar"))
    implementation(files("libs/androidsdk-release.aar"))

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

    testImplementation("junit:junit:4.13.2")
}
