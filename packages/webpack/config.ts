import type { Configuration } from 'webpack';

import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import webpack from 'webpack';
import { merge } from 'webpack-merge';
import nodeExternals from 'webpack-node-externals';

import { AssetRelocatorCachePlugin, OptionalModulesPlugin, RunScriptPlugin } from './plugins.js';

const require = createRequire(import.meta.url);

interface Options {
	entry: Configuration['entry'];
	mode: 'development' | 'production';
	project_path: string;
	config_path: string;
	transpile_packages?: (string | RegExp)[];
	minimize?: boolean;
}

export const get_server_config = ({
	entry = [],
	mode,
	project_path,
	config_path,
	transpile_packages = [],
	minimize = false,
}: Options): Configuration => {
	// Convert strings into regular expressions for exact matches.
	for (let i = 0; i < transpile_packages.length; i++) {
		if (typeof transpile_packages[i] === 'string') {
			transpile_packages[i] = new RegExp(transpile_packages[i]!);
		}
	}

	const common_config = {
		entry,
		mode,
		module: {
			rules: [
				{
					test: /\.(js|ts|node)$/,
					parser: { amd: false },
					use: {
						loader: require.resolve('@vercel/webpack-asset-relocator-loader'),
						options: {
							outputAssetBase: 'assets',
						},
					},
				},
				{
					test: /\.ts$/,
					use: {
						loader: require.resolve('ts-loader'),
						options: {
							transpileOnly: true,
						},
					},
					exclude: [
						{
							and: [path.resolve(project_path, 'node_modules')],
							not: transpile_packages,
						},
					],
				},
			],
		},
		resolve: {
			extensions: ['.js', '.ts'],
		},
		plugins: [
			new webpack.ProgressPlugin(),
			new AssetRelocatorCachePlugin(),
			new OptionalModulesPlugin(),
		],
		target: 'node',
	} satisfies Configuration;

	const dev_config = {
		entry: ['webpack/hot/poll?100'],
		output: {
			path: path.join(project_path, 'node_modules', '.camp', 'build'),
			filename: '[name].js',
			clean: true,
		},
		plugins: [
			new webpack.HotModuleReplacementPlugin(),
			new RunScriptPlugin(),
		],
		cache: {
			buildDependencies: {
				config: [fileURLToPath(import.meta.url), config_path],
			},
			type: 'filesystem',
		},
		externals: [
			nodeExternals({
				allowlist: ['webpack/hot/poll?100', ...transpile_packages],
			}),
		],
	} satisfies Configuration;

	const prod_config = {
		output: {
			path: path.join(project_path, '.camp', 'build'),
			filename: '[name].js',
			clean: true,
			module: true,
		},
		optimization: {
			minimize,
		},
	} satisfies Configuration;

	if (mode === 'development') return merge<Configuration>(common_config, dev_config);
	else return merge<Configuration>(common_config, prod_config);
};
