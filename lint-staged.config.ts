// oxlint-disable import/no-default-export
import { lint_staged_config } from '@xcamp/config/lint_staged.ts';

export default {
	...lint_staged_config,
	'*.rs': [
		() => 'cargo clippy -- --deny warnings',
		() => 'cargo +nightly fmt --check',
	],
};
