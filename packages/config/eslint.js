import path from 'node:path';

import { includeIgnoreFile } from '@eslint/compat';
import js from '@eslint/js';
import { defineConfig } from 'eslint/config';
import imp from 'eslint-plugin-import';
import jsxA11y from 'eslint-plugin-jsx-a11y';
import prettier from 'eslint-plugin-prettier/recommended';
import react from 'eslint-plugin-react';
import * as reactHooks from 'eslint-plugin-react-hooks';
import unicorn from 'eslint-plugin-unicorn';
import globals from 'globals';
import ts from 'typescript-eslint';

const gitignorePath = path.resolve(process.cwd(), '.gitignore');

const getImportGroups = () => [
  'builtin',
  'external',
  'internal',
  'parent',
  'sibling',
  'index',
  'object',
  'type',
];

export default defineConfig([
  includeIgnoreFile(gitignorePath),
  js.configs.recommended,
  ts.configs.recommended,
  {
    name: 'import/recommended',
    extends: [imp.flatConfigs.recommended],
    rules: {
      'import/default': 'off',
      'import/extensions': [
        'error',
        { js: 'ignorePackages', jsx: 'never', ts: 'ignorePackages', tsx: 'never' },
      ],
      'import/no-extraneous-dependencies': [
        'error',
        {
          devDependencies: [
            '**/commitlint.config.*s',
            '**/eslint.config.*s',
            '**/lint-staged.config.*s',
            '**/prettier.config.*s',
            '**/stylelint.config.*s',
            '**/vite.config.*s',
            '**/vitest.config.*s',
            '**/webpack.config.*s',
          ],
        },
      ],
      'import/no-named-as-default': 'off',
      'import/no-named-as-default-member': 'off',
      'import/order': [
        'error',
        {
          'alphabetize': { order: 'asc' },
          'groups': getImportGroups(),
          'newlines-between': 'always',
        },
      ],
    },
    settings: {
      'import/resolver': {
        typescript: {
          project: [
            'server/tsconfig.json', //
            'web/tsconfig.json',
          ],
        },
      },
    },
  },
  {
    name: 'typescript',
    files: ['**/*.{ts,tsx}'],
    extends: [imp.flatConfigs.typescript],
    rules: {
      '@typescript-eslint/member-ordering': 'error',
      '@typescript-eslint/no-explicit-any': 'off',
    },
  },
  {
    name: 'react/recommended',
    files: ['**/*.{js,jsx,ts,tsx}'],
    extends: [react.configs.flat.recommended, react.configs.flat['jsx-runtime']],
    languageOptions: {
      globals: { ...globals.browser, ...globals.serviceworker },
    },
    rules: {
      'react/jsx-filename-extension': ['error', { extensions: ['.jsx', '.tsx'] }],
      'react/jsx-props-no-spreading': 'off',
      'react/require-default-props': ['error', { functions: 'defaultArguments' }],
    },
  },
  {
    name: 'react-hooks/recommended',
    extends: [reactHooks.configs['recommended-latest']],
  },
  {
    name: 'jsx-a11y/recommended',
    extends: [jsxA11y.flatConfigs.recommended],
  },
  {
    name: 'unicorn/recommended',
    plugins: { unicorn },
    rules: {
      ...unicorn.configs.recommended.rules,
      'unicorn/consistent-function-scoping': 'off',
      'unicorn/filename-case': 'off',
      'unicorn/no-array-for-each': 'off',
      'unicorn/no-for-loop': 'off',
      'unicorn/no-null': 'off',
      'unicorn/no-useless-switch-case': 'off',
      'unicorn/no-useless-undefined': 'off',
      'unicorn/prefer-at': 'off',
      'unicorn/prefer-string-replace-all': 'off',
      'unicorn/prevent-abbreviations': 'off',
    },
  },
  {
    name: 'prettier/recommended',
    extends: [prettier],
  },
  {
    files: ['**/*.{js,cjs,mjs,ts,cts,mts}'],
    languageOptions: {
      globals: { ...globals.node },
    },
  },
  {
    files: ['**.{cjs,cts}'],
    rules: {
      '@typescript-eslint/no-require-imports': 'off',
    },
  },
  {
    rules: {
      'arrow-body-style': 'off',
      'class-methods-use-this': 'off',
      'no-await-in-loop': 'off',
      'no-console': 'warn',
      'no-continue': 'off',
      'no-empty-function': ['error', { allow: ['arrowFunctions', 'constructors'] }],
      'no-param-reassign': 'off',
      'no-plusplus': 'off',
      'no-restricted-exports': ['error', { restrictedNamedExports: ['then'] }],
      'no-use-before-define': 'off',
      'no-useless-constructor': 'off',
    },
  },
]);
