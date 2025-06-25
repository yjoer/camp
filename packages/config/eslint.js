import path from 'node:path';

import compat from '@cmpx/eslint-plugin-compat';
import { includeIgnoreFile } from '@eslint/compat';
import { defineConfig } from 'eslint/config';
import prettier from 'eslint-config-prettier/flat';
import { createTypeScriptImportResolver } from 'eslint-import-resolver-typescript';
import * as imp from 'eslint-plugin-import-x';
import jsxA11y from 'eslint-plugin-jsx-a11y';
import perfectionist from 'eslint-plugin-perfectionist';
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
  {
    // https://github.com/typescript-eslint/typescript-eslint/blob/v8.34.1/packages/eslint-plugin/src/configs/eslint-recommended-raw.ts
    rules: {
      'constructor-super': 'off',
      'getter-return': 'off', // nursery
      'no-dupe-args': 'off', // x
      'no-misleading-character-class': 'error',
      'no-octal': 'error', // x
      'no-undef': 'off', // nursery
      'no-unreachable': 'off', // nursery
    },
  },
  {
    name: 'import/recommended',
    plugins: { 'import-x': imp.importX },
    rules: {
      'import-x/export': 'error', // nursery
      'import-x/extensions': [
        'error',
        { js: 'ignorePackages', jsx: 'never', ts: 'ignorePackages', tsx: 'never' },
      ],
      'import-x/no-extraneous-dependencies': [
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
      'import-x/no-unresolved': 'error', // x
      'import-x/order': [
        'error',
        {
          'alphabetize': { order: 'asc' },
          'groups': getImportGroups(),
          'newlines-between': 'always',
        },
      ],
    },
    settings: {
      'import-x/resolver-next': [createTypeScriptImportResolver()],
    },
  },
  {
    name: 'typescript',
    files: ['**/*.{ts,tsx,cts,mts}'],
    plugins: { '@typescript-eslint': ts.plugin },
    extends: [imp.flatConfigs.typescript],
    languageOptions: {
      parser: ts.parser,
    },
    rules: {
      '@typescript-eslint/member-ordering': 'error',
    },
  },
  {
    name: 'react/recommended',
    files: ['**/*.{js,jsx,ts,tsx}'],
    extends: [react.configs.flat['jsx-runtime']],
    languageOptions: {
      globals: { ...globals.browser, ...globals.serviceworker },
    },
    rules: {
      'react/display-name': 'error',
      'react/jsx-uses-vars': 'error',
      'react/no-deprecated': 'error',
      'react/prop-types': 'error',
      //
      'react/jsx-no-leaked-render': 'error',
      'react/require-default-props': ['error', { forbidDefaultForRequired: true, classes: 'ignore', functions: 'ignore' }], // prettier-ignore
    },
  },
  {
    name: 'react-hooks/recommended',
    extends: [reactHooks.configs['recommended-latest']],
  },
  {
    name: 'jsx-a11y/recommended',
    plugins: { 'jsx-a11y': jsxA11y },
    rules: {
      'jsx-a11y/aria-proptypes': 'error',
      'jsx-a11y/interactive-supports-focus': [
        'error',
        {
          tabbable: ['button', 'checkbox', 'link', 'searchbox', 'spinbutton', 'switch', 'textbox'],
        },
      ],
      'jsx-a11y/no-interactive-element-to-noninteractive-role': [
        'error',
        {
          tr: ['none', 'presentation'],
          canvas: ['img'],
        },
      ],
      'jsx-a11y/no-noninteractive-element-interactions': [
        'error',
        {
          handlers: [
            'onClick',
            'onError',
            'onLoad',
            'onMouseDown',
            'onMouseUp',
            'onKeyPress',
            'onKeyDown',
            'onKeyUp',
          ],
          alert: ['onKeyUp', 'onKeyDown', 'onKeyPress'],
          body: ['onError', 'onLoad'],
          dialog: ['onKeyUp', 'onKeyDown', 'onKeyPress'],
          iframe: ['onError', 'onLoad'],
          img: ['onError', 'onLoad'],
        },
      ],
      'jsx-a11y/no-noninteractive-element-to-interactive-role': [
        'error',
        {
          ul: ['listbox', 'menu', 'menubar', 'radiogroup', 'tablist', 'tree', 'treegrid'],
          ol: ['listbox', 'menu', 'menubar', 'radiogroup', 'tablist', 'tree', 'treegrid'],
          li: ['menuitem', 'menuitemradio', 'menuitemcheckbox', 'option', 'row', 'tab', 'treeitem'],
          table: ['grid'],
          td: ['gridcell'],
          fieldset: ['radiogroup', 'presentation'],
        },
      ],
      'jsx-a11y/no-static-element-interactions': [
        'error',
        {
          allowExpressionValues: true,
          handlers: ['onClick', 'onMouseDown', 'onMouseUp', 'onKeyPress', 'onKeyDown', 'onKeyUp'],
        },
      ],
    },
  },
  {
    name: 'unicorn/recommended',
    plugins: { unicorn },
    rules: {
      'unicorn/expiring-todo-comments': 'error',
      'unicorn/import-style': 'error',
      'unicorn/no-array-callback-reference': 'error',
      'unicorn/no-named-default': 'error',
      'unicorn/no-unnecessary-array-splice-count': 'error',
      'unicorn/no-unnecessary-polyfills': 'error',
      'unicorn/prefer-at': 'error',
      'unicorn/prefer-default-parameters': 'error',
      'unicorn/prefer-export-from': 'error',
      'unicorn/prefer-keyboard-event-key': 'error',
      'unicorn/prefer-module': 'error',
      'unicorn/prefer-single-call': 'error',
      'unicorn/prefer-switch': 'error',
      'unicorn/prefer-ternary': 'error',
      'unicorn/prefer-top-level-await': 'error',
      'unicorn/relative-url-style': 'error',
      'unicorn/template-indent': 'error',
      //
      'unicorn/no-for-loop': 'off', // x
      'unicorn/prevent-abbreviations': 'off',
    },
  },
  {
    name: 'prettier/config',
    extends: [prettier],
  },
  {
    name: 'perfectionist',
    plugins: { perfectionist },
    rules: {
      'perfectionist/sort-array-includes': 'warn',
      'perfectionist/sort-enums': 'warn',
      'perfectionist/sort-heritage-clauses': 'warn',
      'perfectionist/sort-interfaces': 'warn',
      'perfectionist/sort-intersection-types': 'warn',
      'perfectionist/sort-jsx-props': [
        'warn',
        {
          groups: ['key', 'ref', 'unknown'],
          customGroups: [
            {
              groupName: 'key',
              elementNamePattern: 'key',
            },
            {
              groupName: 'ref',
              elementNamePattern: 'ref',
            },
          ],
        },
      ],
      'perfectionist/sort-maps': 'warn',
      'perfectionist/sort-named-exports': 'warn',
      'perfectionist/sort-named-imports': 'warn',
      'perfectionist/sort-object-types': 'warn',
      'perfectionist/sort-sets': 'warn',
      'perfectionist/sort-switch-case': 'warn',
      'perfectionist/sort-union-types': 'warn',
    },
    settings: {
      perfectionist: {
        type: 'natural',
        ignoreCase: false,
      },
    },
  },
  {
    name: 'compat/recommended',
    extends: [compat.configs['flat/recommended']],
  },
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

export function defineImportResolver(pkgs, options = {}) {
  return pkgs.map((pkg) => ({
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
