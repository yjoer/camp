// oxlint-disable import/no-default-export
import { oxlint_config } from '@xcamp/config/oxlint.ts';
import { defineConfig } from 'vite-plus';

export default defineConfig({
	lint: {
		extends: [oxlint_config],
		options: {
			typeAware: true,
			typeCheck: true,
		},
	},
});
