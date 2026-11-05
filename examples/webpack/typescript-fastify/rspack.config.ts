import { get_server_config } from '@camp-org/webpack/rspack_config';
import { rspack } from '@rspack/core';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { merge } from 'webpack-merge';

const mode = process.env.NODE_ENV === 'production' ? 'production' : 'development';

const ignored_packages: Record<string, boolean> = {
	'aws-sdk': true,
	'mock-aws-s3': true,
	nock: true,
};

const custom_config = {
	plugins: [
		new rspack.IgnorePlugin({
			checkResource: resource => !!ignored_packages[resource],
		}),
	],
};

const config = get_server_config({
	entry: ['./server.ts'],
	mode,
	project_path: path.dirname(fileURLToPath(import.meta.url)),
	config_path: fileURLToPath(import.meta.url),
});

export default merge(config, custom_config);
