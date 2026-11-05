// oxlint-disable import/no-default-export
export default function stylistic_loader(source: string) {
	if (!/from\s*['"].*rolldown-runtime/.test(source)) return source;

	const names = new Map<string, string>();
	let imports = '';
	const patched = source.replaceAll(/__require\("([^"]+)"\)/g, (_match: string, specifier: string) => {
		const name = names.get(specifier);
		if (name) return name;

		const new_name = `__bundled_require_${names.size}`;
		imports += `import * as ${new_name} from '${specifier}';\n`;
		names.set(specifier, new_name);
		return new_name;
	});

	if (patched === source) throw new Error('stylistic loader: __require pattern not found');
	return imports + patched;
}
