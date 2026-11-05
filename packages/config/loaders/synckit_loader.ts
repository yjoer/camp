// oxlint-disable import/no-default-export
export default function synckit_loader(source: string) {
	const patched = source.replace("path.resolve(_dirname, '../register.cjs')", "path.resolve(_dirname, 'register.cjs')");

	if (patched === source) throw new Error('synckit loader: register pattern not found');
	return patched;
}
