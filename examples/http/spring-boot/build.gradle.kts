import org.gradle.plugins.ide.eclipse.model.Classpath
import org.gradle.plugins.ide.eclipse.model.SourceFolder

plugins {
  java
  eclipse
  id("org.springframework.boot") version "4.0.1"
  id("io.spring.dependency-management") version "1.1.7"
}

repositories { mavenCentral() }

eclipse {
  classpath {
    defaultOutputDir = file(".build/eclipse/default")
    file.whenMerged {
      val cp = this as Classpath
      cp.entries.forEach { entry ->
        if (entry is SourceFolder && entry.output != null) {
          entry.output = entry.output.replace("bin", ".build/eclipse")
        }
      }
    }
  }
}

dependencies {
  implementation("org.springframework.boot:spring-boot-starter-webmvc")
}

layout.buildDirectory.set(file(".build"))

sourceSets {
  main {
    java.srcDirs("src")
    resources.srcDirs("res")
  }
}
