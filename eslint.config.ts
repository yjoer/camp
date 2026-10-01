/* eslint-disable import-x/no-extraneous-dependencies */
// oxlint-disable import/no-default-export
import { defineImportResolver, eslint_config } from '@camp-org/config/eslint';
import { defineConfig } from 'eslint/config';

process.chdir(import.meta.dirname);

export default defineConfig([
	...eslint_config,
	...defineImportResolver(['examples/tanstack/start', 'examples/tanstack/start-solid', 'examples/tanstack/start-trpc']),
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
