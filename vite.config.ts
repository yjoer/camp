import { oxlint_config } from '@camp-org/config/oxlint';
import { defineConfig } from 'vite-plus';

export default defineConfig({
	lint: {
		extends: [oxlint_config],
		options: {
			typeAware: true,
			typeCheck: true,
		},
	},
	test: {
		projects: [
			'examples/http/_bench',
		],
	},
});
