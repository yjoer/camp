// oxlint-disable import/no-default-export
export default function commitlint_loader(source: string) {
	// use the bundled formatter instead of resolving it at runtime
	const patched = source.replace(
		'    if (modulePath) {\n        return dynamicImport(modulePath);\n    }',
		'    if (moduleName === "@commitlint/format") {\n        return require("@commitlint/format").default;\n    }\n    if (modulePath) {\n        return dynamicImport(modulePath);\n    }',
	);

	if (patched === source) throw new Error('commitlint loader: formatter pattern not found');
	return patched;
}
