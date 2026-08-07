pluginManagement {
    repositories {
        google()
        mavenCentral()
        // usb-serial-for-android is published through JitPack.
        maven { url = uri("https://jitpack.io") }
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://jitpack.io") }
    }
}

rootProject.name = "PTCDepthAndroid"
include(":app")
