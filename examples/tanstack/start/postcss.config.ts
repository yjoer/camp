// oxlint-disable import/no-default-export
import stylex from '@stylexjs/postcss-plugin';
import autoprefixer from 'autoprefixer';

// @ts-expect-error ext
import { babelConfig } from './vite.config.ts';

import type { Config } from 'postcss-load-config';

const config = {
	plugins: [
		stylex({
			include: ['_components/**/*.{ts,tsx}', 'src/**/*.{ts,tsx}', 'ui/**/*.{ts,tsx}'],
			useCSSLayers: true,
			babelConfig,
		}),
		autoprefixer(),
	],
} satisfies Config;

export default config;
