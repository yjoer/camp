// oxlint-disable import/no-default-export
export default function commitlint_loader(source: string) {
	// use the bundled formatter instead of resolving it at runtime
	return source.replace(
		'    if (modulePath) {\n        return dynamicImport(modulePath);\n    }',
		'    if (moduleName === "@commitlint/format") {\n        return require("@commitlint/format").default;\n    }\n    if (modulePath) {\n        return dynamicImport(modulePath);\n    }',
	);
}
