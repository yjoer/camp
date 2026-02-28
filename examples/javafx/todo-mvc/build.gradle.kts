import org.gradle.plugins.ide.eclipse.model.Classpath
import org.gradle.plugins.ide.eclipse.model.SourceFolder

plugins {
	application
	eclipse
	id("org.openjfx.javafxplugin") version "0.1.0"
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

javafx {
	version = "23"
	modules("javafx.controls")
}

layout.buildDirectory.set(file(".build"))

sourceSets { main { java.srcDirs("src") } }

application { mainClass = "Launcher" }
