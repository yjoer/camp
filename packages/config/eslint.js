import stylex_plugin from '@stylexjs/eslint-plugin';
import { defineConfig } from 'eslint/config';
import gitignore from 'eslint-config-flat-gitignore';
import prettier_config from 'eslint-config-prettier/flat';
import { createTypeScriptImportResolver } from 'eslint-import-resolver-typescript';
import compat_plugin from 'eslint-plugin-compat';
import * as imp from 'eslint-plugin-import-x';
import jsx_a11y_plugin from 'eslint-plugin-jsx-a11y';
import perfectionist_plugin from 'eslint-plugin-perfectionist';
import react_plugin from 'eslint-plugin-react';
import react_hooks_plugin from 'eslint-plugin-react-hooks';
import unicorn_plugin from 'eslint-plugin-unicorn';
import globals from 'globals';
import ts from 'typescript-eslint';

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

function eslint() {
  return {
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
  };
}

function import_x() {
  return {
    name: 'import/recommended',
    plugins: { 'import-x': imp.importX },
    rules: {
      'import-x/export': 'error', // nursery
      // 'import-x/extensions': [
      //   'error',
      //   { js: 'ignorePackages', jsx: 'never', ts: 'ignorePackages', tsx: 'never' },
      // ],
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
          alphabetize: { order: 'asc' },
          groups: getImportGroups(),
          'newlines-between': 'always',
        },
      ],
    },
    settings: {
      'import-x/resolver-next': [createTypeScriptImportResolver()],
    },
  };
}

function typescript() {
  return {
    name: 'typescript',
    files: ['**/*.{ts,cts,mts,tsx}'],
    plugins: { '@typescript-eslint': ts.plugin },
    extends: [imp.flatConfigs.typescript],
    languageOptions: {
      parser: ts.parser,
    },
    rules: {
      '@typescript-eslint/member-ordering': 'error',
    },
  };
}

function react() {
  return {
    name: 'react/recommended',
    files: ['**/*.{js,jsx,ts,tsx}'],
    extends: [react_plugin.configs.flat['jsx-runtime']],
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
  };
}

function react_hooks() {
  return {
    name: 'react-hooks/recommended',
    files: ['**/*.{js,jsx,ts,tsx}'],
    extends: [react_hooks_plugin.configs.flat['recommended-latest']],
  };
}

function stylex() {
  return {
    name: 'stylex/recommended',
    files: ['**/*.{js,jsx,ts,tsx}'],
    plugins: { '@stylexjs': stylex_plugin },
    rules: {
      '@stylexjs/no-unused': 'error',
      '@stylexjs/sort-keys': [
        'warn',
        {
          validImports: ['stylex', '@stylexjs/stylex', { from: 'react-strict-dom', as: 'css' }],
          order: 'recess-order',
        },
      ],
      '@stylexjs/valid-shorthands': 'warn',
      '@stylexjs/valid-styles': 'error',
    },
  };
}

function jsx_a11y() {
  return {
    name: 'jsx-a11y/recommended',
    plugins: { 'jsx-a11y': jsx_a11y_plugin },
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
  };
}

function unicorn() {
  return {
    name: 'unicorn/recommended',
    plugins: { unicorn: unicorn_plugin },
    rules: {
      'unicorn/expiring-todo-comments': 'error',
      'unicorn/import-style': 'error',
      'unicorn/no-named-default': 'error',
      'unicorn/no-unnecessary-polyfills': 'error',
      'unicorn/prefer-default-parameters': 'error',
      'unicorn/prefer-export-from': 'error',
      'unicorn/prefer-keyboard-event-key': 'error',
      'unicorn/prefer-module': 'error',
      'unicorn/prefer-single-call': 'error',
      'unicorn/prefer-switch': 'error',
      'unicorn/prefer-ternary': 'error',
      'unicorn/relative-url-style': 'error',
      'unicorn/template-indent': 'error',
      //
      'unicorn/no-for-loop': 'off', // x
      'unicorn/prevent-abbreviations': 'off',
    },
  };
}

function prettier() {
  return {
    name: 'prettier/config',
    extends: [prettier_config],
  };
}

function perfectionist() {
  return {
    name: 'perfectionist',
    plugins: { perfectionist: perfectionist_plugin },
    rules: {
      'perfectionist/sort-array-includes': 'warn',
      'perfectionist/sort-enums': 'warn',
      'perfectionist/sort-heritage-clauses': 'warn',
      'perfectionist/sort-interfaces': [
        'warn',
        {
          groups: ['property', 'method'],
        },
      ],
      'perfectionist/sort-intersection-types': 'warn',
      'perfectionist/sort-jsx-props': [
        'warn',
        {
          groups: ['key', 'ref', 'name', 'content', 'unknown'],
          customGroups: [
            {
              groupName: 'key',
              elementNamePattern: 'key',
            },
            {
              groupName: 'ref',
              elementNamePattern: 'ref',
            },
            {
              groupName: 'name',
              elementNamePattern: 'name',
            },
            {
              groupName: 'content',
              elementNamePattern: 'content',
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
  };
}

function compat() {
  return {
    name: 'compat/recommended',
    extends: [compat_plugin.configs['flat/recommended']],
  };
}

export default defineConfig([
  gitignore(),
  eslint(),
  import_x(),
  typescript(),
  react(),
  react_hooks(),
  stylex(),
  jsx_a11y(),
  unicorn(),
  prettier(),
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
