/* eslint-disable import-x/no-extraneous-dependencies */
import { eslint_config, get_import_resolver } from '@camp-org/config/eslint';
import { defineConfig } from 'eslint/config';

process.chdir(import.meta.dirname);

export default defineConfig([
	...eslint_config,
	...get_import_resolver(['examples/tanstack/start', 'examples/tanstack/start-solid', 'examples/tanstack/start-trpc']),
	{
		files: ['stories/**/*.tsx'],
		settings: {
			'better-tailwindcss': {
				entryPoint: '.storybook/global.css',
				cwd: './stories',
			},
		},
	},
]);
