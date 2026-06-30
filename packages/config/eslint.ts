import type { ConfigWithExtends, ConfigWithExtendsArray, Plugin } from '@eslint/config-helpers';

import stylex_plugin from '@stylexjs/eslint-plugin';
import stylistic_plugin from '@stylistic/eslint-plugin';
import query_plugin from '@tanstack/eslint-plugin-query';
import router_plugin from '@tanstack/eslint-plugin-router';
import { defineConfig } from 'eslint/config';
import gitignore from 'eslint-config-flat-gitignore';
import { createTypeScriptImportResolver } from 'eslint-import-resolver-typescript';
import tailwind_plugin from 'eslint-plugin-better-tailwindcss';
import compat_plugin from 'eslint-plugin-compat';
import * as imp from 'eslint-plugin-import-x';
import perfectionist_plugin from 'eslint-plugin-perfectionist';
import { Alphabet } from 'eslint-plugin-perfectionist/alphabet';
import react_plugin from 'eslint-plugin-react';
import react_hooks_plugin from 'eslint-plugin-react-hooks';
import react_you_might_not_need_an_effect_plugin from 'eslint-plugin-react-you-might-not-need-an-effect';
import unicorn_plugin from 'eslint-plugin-unicorn';
import globals from 'globals';
import ts from 'typescript-eslint';

function import_x() {
	return {
		name: 'import/recommended',
		plugins: { 'import-x': imp.importX },
		rules: {
			'import-x/no-extraneous-dependencies': ['error', {
				devDependencies: [
					'**/babel.config.*s',
					'**/commitlint.config.*s',
					'**/eslint.config.*s',
					'**/lint-staged.config.*s',
					'**/postcss.config.*s',
					'**/prettier.config.*s',
					'**/rspack.config.*s',
					'**/stylelint.config.*s',
					'**/vite.config.*s',
					'**/vitest.config.*s',
					'**/webpack.config.*s',
				],
			}],
			'import-x/no-unresolved': ['error', { ignore: [String.raw`^@\/build`] }], // x
		},
		settings: {
			'import-x/resolver-next': [createTypeScriptImportResolver()],
		},
	} satisfies ConfigWithExtends;
}

function typescript() {
	return {
		name: 'typescript',
		files: ['**/*.{ts,cts,mts,tsx}'],
		extends: [imp.flatConfigs.typescript],
		languageOptions: {
			parser: ts.parser,
		},
	} satisfies ConfigWithExtends;
}

function react() {
	return {
		name: 'react/recommended',
		files: ['**/*.{jsx,tsx}'],
		extends: [react_plugin.configs.flat['jsx-runtime']!],
		languageOptions: {
			globals: { ...globals.browser, ...globals.serviceworker },
		},
		rules: {
			'react/no-deprecated': 'error',
			//
			'react/jsx-no-leaked-render': 'error',
			'react/require-default-props': ['error', { forbidDefaultForRequired: true, classes: 'ignore', functions: 'ignore' }],
		},
	} satisfies ConfigWithExtends;
}

function react_hooks() {
	return {
		name: 'react-hooks/recommended',
		files: ['**/*.{jsx,tsx}'],
		extends: [react_hooks_plugin.configs.flat['recommended-latest']],
	} satisfies ConfigWithExtends;
}

function react_you_might_not_need_an_effect() {
	return {
		name: 'react/you-might-not-need-an-effect/recommended',
		files: ['**/*.{jsx,tsx}'],
		extends: [react_you_might_not_need_an_effect_plugin.configs.recommended],
	} satisfies ConfigWithExtends;
}

function tanstack_query() {
	return {
		name: '@tanstack/query',
		plugins: { '@tanstack/query': query_plugin as unknown as Plugin },
		rules: {
			'@tanstack/query/exhaustive-deps': 'error',
			'@tanstack/query/infinite-query-property-order': 'error',
			'@tanstack/query/mutation-property-order': 'error',
			'@tanstack/query/no-rest-destructuring': 'error',
			'@tanstack/query/no-unstable-deps': 'error',
			'@tanstack/query/no-void-query-fn': 'error',
			'@tanstack/query/prefer-query-options': 'error',
			'@tanstack/query/stable-query-client': 'error',
		},
	} satisfies ConfigWithExtends;
}

function tanstack_router() {
	return {
		name: '@tanstack/router',
		plugins: { '@tanstack/router': router_plugin as unknown as Plugin },
		rules: {
			'@tanstack/router/create-route-property-order': 'error',
			'@tanstack/router/route-param-names': 'error',
		},
	} satisfies ConfigWithExtends;
}

function stylex() {
	return {
		name: 'stylex/recommended',
		files: ['**/*.{jsx,tsx}'],
		plugins: { '@stylexjs': stylex_plugin as unknown as Plugin },
		rules: {
			'@stylexjs/enforce-extension': 'error',
			'@stylexjs/no-conflicting-props': 'error',
			'@stylexjs/no-legacy-contextual-styles': 'error',
			'@stylexjs/no-nonstandard-styles': 'error',
			'@stylexjs/no-unused': 'error',
			'@stylexjs/sort-keys': ['warn', {
				validImports: ['stylex', '@stylexjs/stylex', { from: 'react-strict-dom', as: 'css' }],
				order: 'recess',
			}],
			'@stylexjs/valid-shorthands': 'warn',
			'@stylexjs/valid-styles': 'error',
		},
	} satisfies ConfigWithExtends;
}

function tailwind() {
	return {
		name: 'better-tailwindcss/recommended',
		files: ['**/*.{jsx,tsx}'],
		plugins: { 'better-tailwindcss': tailwind_plugin },
		rules: {
			'better-tailwindcss/enforce-consistent-line-wrapping': ['warn', { printWidth: 100, preferSingleLine: true, indent: 'tab' }],
			'better-tailwindcss/enforce-consistent-class-order': 'warn',
			'better-tailwindcss/enforce-consistent-variant-order': 'warn',
			'better-tailwindcss/enforce-consistent-variable-syntax': 'warn',
			'better-tailwindcss/enforce-consistent-important-position': 'warn',
			'better-tailwindcss/enforce-canonical-classes': 'warn',
			'better-tailwindcss/no-duplicate-classes': 'warn',
			'better-tailwindcss/no-deprecated-classes': 'warn',
			'better-tailwindcss/no-unnecessary-whitespace': 'warn',
			'better-tailwindcss/no-unknown-classes': 'error',
			'better-tailwindcss/no-conflicting-classes': 'error',
		},
	} satisfies ConfigWithExtends;
}

function unicorn() {
	return {
		name: 'unicorn/recommended',
		plugins: { unicorn: unicorn_plugin },
		rules: {
			'unicorn/expiring-todo-comments': 'error',
			'unicorn/import-style': 'error',
			'unicorn/isolated-functions': 'error',
			'unicorn/no-for-loop': 'off', // x
			'unicorn/no-unnecessary-polyfills': 'error',
			'unicorn/prefer-export-from': 'error',
			'unicorn/prefer-simple-condition-first': 'error',
			'unicorn/prefer-single-call': 'error',
			'unicorn/prefer-switch': 'error',
			'unicorn/prevent-abbreviations': 'off',
			'unicorn/switch-case-break-position': 'error',
			'unicorn/template-indent': 'error',
		},
	} satisfies ConfigWithExtends;
}

function stylistic() {
	const config = stylistic_plugin.configs.customize({
		indent: 'tab',
		quotes: 'single',
		semi: true,
		jsx: true,
		braceStyle: '1tbs',
	});

	return [
		{
			name: 'stylistic',
			plugins: { '@stylistic': stylistic_plugin },
			rules: {
				...config.rules,
				'@stylistic/indent': ['error', 'tab', {
					ignoredNodes: ['TSUnionType', 'TSIntersectionType'],
					SwitchCase: 1,
					MemberExpression: 0,
					offsetTernaryExpressions: true,
				}],
				'@stylistic/jsx-one-expression-per-line': ['error', { allow: 'non-jsx' }],
				'@stylistic/jsx-closing-bracket-location': ['error', { nonEmpty: 'after-props', selfClosing: 'tag-aligned' }],
				'@stylistic/jsx-wrap-multilines': ['error', {
					declaration: 'parens-new-line',
					assignment: 'parens-new-line',
					return: 'parens-new-line',
					arrow: 'parens-new-line',
					condition: 'ignore',
					logical: 'ignore',
					prop: 'ignore',
					propertyValue: 'ignore',
				}],
				'@stylistic/multiline-ternary': ['error', 'never'],
				'@stylistic/no-mixed-spaces-and-tabs': ['error', 'smart-tabs'],
				'@stylistic/no-tabs': ['error', { allowIndentationTabs: true }],
				'@stylistic/operator-linebreak': ['error', 'after', { overrides: { '?': 'before', ':': 'before', '|': 'before' } }],
				'@stylistic/quotes': ['error', 'single', { allowTemplateLiterals: 'always', avoidEscape: true }],
			},
		},
		{
			name: 'stylistic/react',
			files: ['**/*.{jsx,tsx}'],
			rules: {
				'@stylistic/quote-props': ['error', 'as-needed'],
			},
		},
	] satisfies ConfigWithExtendsArray;
}

function perfectionist() {
	return {
		name: 'perfectionist',
		plugins: { perfectionist: perfectionist_plugin },
		rules: {
			'perfectionist/sort-array-includes': 'warn',
			'perfectionist/sort-enums': 'warn',
			'perfectionist/sort-heritage-clauses': 'warn',
			'perfectionist/sort-imports': ['error', {
				type: 'custom',
				alphabet: Alphabet.generateRecommendedAlphabet()
				.sortByNaturalSort()
				.placeCharacterBefore({ characterBefore: '/', characterAfter: '-' })
				.placeCharacterBefore({ characterBefore: '.', characterAfter: '/' })
				.getCharacters(),
			}],
			'perfectionist/sort-interfaces': ['warn', {
				type: 'unsorted',
				groups: ['property', 'method'],
			}],
			'perfectionist/sort-intersection-types': 'warn',
			'perfectionist/sort-jsx-props': ['warn', {
				groups: ['key', 'ref', 'name', 'content', 'unknown', 'callback'],
				customGroups: [
					{ groupName: 'key', elementNamePattern: '^key$' },
					{ groupName: 'ref', elementNamePattern: '^ref$' },
					{ groupName: 'name', elementNamePattern: '^name$' },
					{ groupName: 'content', elementNamePattern: '^content$' },
					{ groupName: 'callback', elementNamePattern: ['^on_.+', '^on[A-Z].*'] },
				],
			}],
			'perfectionist/sort-maps': 'warn',
			'perfectionist/sort-named-exports': 'warn',
			'perfectionist/sort-named-imports': 'warn',
			'perfectionist/sort-object-types': ['warn', {
				type: 'unsorted',
				groups: ['property', 'method'],
			}],
			'perfectionist/sort-sets': 'warn',
			'perfectionist/sort-switch-case': 'warn',
			'perfectionist/sort-union-types': ['warn', {
				type: 'unsorted',
				groups: ['unknown', 'named', 'nullish'],
			}],
		},
		settings: {
			perfectionist: {
				type: 'natural',
				ignoreCase: false,
			},
		},
	} satisfies ConfigWithExtends;
}

function compat() {
	return {
		name: 'compat/recommended',
		extends: [compat_plugin.configs['flat/recommended']],
	};
}

export const eslint_config = defineConfig([
	gitignore(),
	import_x(),
	typescript(),
	react(),
	react_hooks(),
	react_you_might_not_need_an_effect(),
	tanstack_query(),
	tanstack_router(),
	stylex(),
	tailwind(),
	unicorn(),
	stylistic(),
	perfectionist(),
	compat(),
	{
		files: ['**/*.{js,cjs,mjs,ts,cts,mts}'],
		languageOptions: {
			globals: { ...globals.node },
		},
	},
	{
		files: ['**/router.tsx'],
		linterOptions: {
			reportUnusedDisableDirectives: false,
		},
	},
]);

export function defineImportResolver(pkgs: string[], options = {}) {
	return pkgs.map(pkg => ({
		files: [`./${pkg}/**/*`],
		settings: {
			'import-x/resolver-next': [
				createTypeScriptImportResolver({
					...options,
					project: pkg,
				}),
			],
		},
	}));
}
