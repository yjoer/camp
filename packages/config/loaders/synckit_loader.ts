// oxlint-disable import/no-default-export
export default function synckit_loader(source: string) {
	return source.replace("path.resolve(_dirname, '../register.cjs')", "path.resolve(_dirname, 'register.cjs')");
}
