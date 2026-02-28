plugins {
	id("com.android.application") version "9.0.1"
}

layout.buildDirectory.set(file(".build"))

kotlin { jvmToolchain(11) }

android {
	namespace = "com.yjoer.samples"
	compileSdk = 36

	defaultConfig {
		applicationId = "com.yjoer.samples"
		minSdk = 24
		targetSdk = 36
		versionCode = 1
		versionName = "1.0"
	}

	buildTypes {
		release {
			isMinifyEnabled = true
			isShrinkResources = true
			proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
		}
	}

	sourceSets {
		named("main") {
			kotlin.directories += "src"
			res.directories += "res"
			manifest.srcFile("src/AndroidManifest.xml")
		}
	}
}

dependencies {
	implementation("androidx.appcompat:appcompat:1.7.1")
	implementation("com.google.android.material:material:1.14.0-alpha09")
}
