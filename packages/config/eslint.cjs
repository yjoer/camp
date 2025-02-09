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

module.exports = {
  env: {
    browser: true,
    node: true,
  },
  extends: [
    'airbnb',
    'plugin:import/typescript',
    'plugin:react/jsx-runtime',
    'plugin:unicorn/recommended',
    'prettier',
  ],
  overrides: [
    {
      files: ['*.ts', '*.tsx'],
      extends: ['plugin:@typescript-eslint/recommended'],
      parser: '@typescript-eslint/parser',
      plugins: ['@typescript-eslint'],
      rules: {
        '@typescript-eslint/member-ordering': 'error',
        '@typescript-eslint/no-explicit-any': 'off',
      },
    },
  ],
  rules: {
    'arrow-body-style': 'off',
    'class-methods-use-this': 'off',
    'no-await-in-loop': 'off',
    'no-continue': 'off',
    'no-empty-function': ['error', { allow: ['arrowFunctions', 'constructors'] }],
    'no-param-reassign': 'off',
    'no-plusplus': 'off',
    'no-restricted-exports': ['error', { restrictedNamedExports: ['then'] }],
    'no-use-before-define': 'off',
    'no-useless-constructor': 'off',
    'import/extensions': [
      'error',
      'ignorePackages',
      { js: 'never', jsx: 'never', ts: 'never', tsx: 'never' },
    ],
    'import/order': [
      'error',
      {
        'alphabetize': { order: 'asc' },
        'groups': getImportGroups(),
        'newlines-between': 'always',
      },
    ],
    'import/prefer-default-export': 'off',
    'react/jsx-filename-extension': ['error', { extensions: ['.jsx', '.tsx'] }],
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
  settings: {
    'import/resolver': {
      typescript: {
        project: [
          'core/tsconfig.json', //
          'web/tsconfig.json',
        ],
      },
    },
  },
};
