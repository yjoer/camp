declare module '@stylexjs/postcss-plugin' {
	import type { ConfigPlugin } from 'postcss-load-config';

	interface StylexOptions {
		include: string[];
		useCSSLayers?: boolean;
		babelConfig?: object;
	}

	export default function stylex(options: StylexOptions): ConfigPlugin;
}
