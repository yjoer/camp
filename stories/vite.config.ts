// oxlint-disable import/no-default-export
import { storybookTest } from '@storybook/addon-vitest/vitest-plugin';
import tailwindcss from '@tailwindcss/vite';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite-plus';
import { playwright } from 'vite-plus/test/browser-playwright';

const dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
	plugins: [
		tailwindcss(),
	],
	test: {
		projects: [{
			plugins: [
				storybookTest({
					configDir: path.join(dirname, '.storybook'),
				}),
			],
			test: {
				name: 'storybook',
				browser: {
					enabled: true,
					provider: playwright(),
					headless: true,
					instances: [{ browser: 'chromium' }],
				},
			},
		}],
	},
});
