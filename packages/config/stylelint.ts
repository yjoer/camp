import type { Config } from 'stylelint';

import scss_syntax from 'postcss-scss';
import styled_syntax from 'postcss-styled-syntax';
import recess_config from 'stylelint-config-recess-order';
import recommended_config from 'stylelint-config-recommended';
import standard_config from 'stylelint-config-standard';
import order_plugin from 'stylelint-order';

export const stylelint_config = {
	plugins: [
		...order_plugin, //
	],
	overrides: [{
		files: '**/*.{jsx,tsx}',
		customSyntax: styled_syntax,
		rules: {
			'function-no-unknown': [true, { ignoreFunctions: ['${', '/theme.+/'] }],
			'value-keyword-case': undefined,
			'value-no-vendor-prefix': [true, { ignoreValues: ['box'] }],
		},
	}, {
		files: '**/*.scss',
		customSyntax: scss_syntax,
		rules: {
			'at-rule-no-unknown': [true, { ignoreAtRules: ['tailwind', 'use'] }],
			'function-no-unknown': [true, { ignoreFunctions: ['darken'] }],
			'selector-pseudo-class-no-unknown': [true, { ignorePseudoClasses: ['global'] }],
		},
	}],
	rules: {
		...recommended_config.rules,
		...standard_config.rules,
		...recess_config.rules,
		'at-rule-no-unknown': [true, { ignoreAtRules: ['theme', 'source', 'utility', 'variant'] }],
		'function-no-unknown': [true, { ignoreFunctions: ['theme'] }],
		'import-notation': 'string',
		'selector-class-pattern': undefined,
		'selector-id-pattern': undefined,
	},
} satisfies Config;
