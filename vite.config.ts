// oxlint-disable import/no-default-export
import { defineConfig } from 'vite-plus';

import { oxlint_config } from '@xcamp/config/oxlint.ts';

export default defineConfig({
	lint: {
		extends: [oxlint_config],
		options: {
			typeAware: true,
			typeCheck: true,
		},
	},
});
