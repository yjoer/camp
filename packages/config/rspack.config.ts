import type { Configuration } from '@rspack/core';

import { rspack } from '@rspack/core';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const project_path = import.meta.dirname;

// eslint-plugin-better-tailwindcss resolves these workers from
// import.meta.dirname, so emit them next to the config.
const tailwind_worker_path = fileURLToPath(new URL('../tailwindcss', import.meta.resolve('eslint-plugin-better-tailwindcss')));

const config: Configuration = {
	entry: {
		commitlint: './commitlint.ts',
		eslint: './eslint.ts',
		prettier: './prettier.ts',
		stylelint: './stylelint.ts',
		'tailwind.async.worker.v3': path.join(tailwind_worker_path, 'tailwind.async.worker.v3.js'),
		'tailwind.async.worker.v4': path.join(tailwind_worker_path, 'tailwind.async.worker.v4.js'),
	},
	mode: 'production',
	devtool: 'nosources-source-map',
	module: {
		parser: {
			javascript: {
				importDynamic: false, // leave eslint-plugin-better-tailwindcss runtime imports to node
				importMeta: {
					url: false,
				},
			},
		},
		rules: [
			{
				test: /\.ts$/,
				exclude: /[\\/]node_modules[\\/]/,
				loader: 'builtin:swc-loader',
				options: {
					detectSyntax: 'auto',
				},
			},
			{
				test: /synckit[\\/]lib[\\/]helpers\.js$/,
				loader: path.join(project_path, 'loaders', 'synckit_loader.ts'),
			},
		],
	},
	resolve: {
		alias: {
			'@eslint/css-tree$': require.resolve('@eslint/css-tree'), // force cjs so rspack can bundle its json data
			json5$: require.resolve('json5'), // json5's esm build has no named parse export
		},
		extensions: ['.js', '.ts', '.json'],
	},
	plugins: [
		new rspack.IgnorePlugin({ resourceRegExp: /^@babel\/preset-typescript(\/|$)/ }),
		new rspack.IgnorePlugin({ resourceRegExp: /^enhanced-resolve\/lib\/createInnerCallback$/ }),
	],
	externals: [
		/^eslint($|\/)/,
		/^postcss($|\/)/,
		/^prettier($|\/)/,
		/^stylelint($|\/)/,
		/^tailwindcss($|\/)/,
		/^typescript($|\/)/,
		/^unrs-resolver($|\/)/,
	],
	// keep esm imports static and preserve cjs require calls, which matters for
	// extensionless and json imports, e.g. typescript/lib/tsserverlibrary and
	// eslint/package.json
	externalsType: 'modern-module',
	target: 'node',
	output: {
		path: path.join(project_path, '.build'),
		filename: '[name].js',
		chunkFilename: '[name].js',
		clean: false,
		module: true,
		library: { type: 'module' },
	},
	optimization: {
		minimize: false,
		splitChunks: {
			chunks: 'all',
			cacheGroups: {
				default: false,
				defaultVendors: false,
				plugins: {
					test: /[\\/]node_modules[\\/]/,
					name: (_module, chunks) => `${chunks[0]?.name ?? 'shared'}_plugins`,
					chunks: chunk => !chunk.name?.startsWith('tailwind.async.worker'),
					enforce: true,
				},
			},
		},
	},
};

const commitlint_config: Configuration = {
	entry: {
		'bin/commitlint': './bin/commitlint.ts',
	},
	mode: 'production',
	module: {
		parser: {
			javascript: {
				createRequire: true,
				dynamicImportMode: 'eager',
				importDynamic: false,
				importMeta: {
					url: false,
				},
			},
		},
		rules: [
			{
				test: /@commitlint[\\/]cli[\\/]lib[\\/]cli\.js$/,
				loader: path.join(project_path, 'loaders', 'commitlint_loader.ts'),
			},
		],
	},
	plugins: [
		new rspack.BannerPlugin({ banner: '#!/usr/bin/env node', raw: true }),
	],
	externals: [
		/^typescript($|\/)/,
	],
	externalsType: 'modern-module',
	target: 'node',
	output: {
		path: path.join(project_path, '.build'),
		filename: '[name].js',
		clean: false,
		module: true,
	},
	optimization: {
		minimize: false,
	},
};

const synckit_config: Configuration = {
	entry: {
		register: fileURLToPath(new URL('../register.cjs', import.meta.resolve('synckit'))),
	},
	mode: 'production',
	target: 'node',
	output: {
		path: path.join(project_path, '.build'),
		filename: '[name].cjs',
		clean: false,
	},
	optimization: {
		minimize: false,
	},
};

export default [config, commitlint_config, synckit_config];
