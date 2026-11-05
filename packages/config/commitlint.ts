import type { UserConfig } from '@commitlint/types';

import conventional_config from '@commitlint/config-conventional';
import create_preset from 'conventional-changelog-conventionalcommits';

const conventional_preset = create_preset() as { parser: unknown };

export const commitlint_config: UserConfig = {
	parserPreset: {
		name: 'conventional-changelog-conventionalcommits',
		parserOpts: conventional_preset.parser,
	},
	rules: {
		...conventional_config.rules,
		'header-max-length': [2, 'always', 100],
		'subject-case': [2, 'always', 'lower-case'],
		// build, ci, docs, test -> chore
		// style -> refactor
		'type-enum': [2, 'always', ['chore', 'feat', 'fix', 'perf', 'refactor', 'revert']],
	},
};
