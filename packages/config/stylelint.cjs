/** @type {import('stylelint').Config} */
module.exports = {
  extends: [
    'stylelint-config-standard',
    'stylelint-config-recess-order', //
  ],
  overrides: [
    {
      files: '**/*.{jsx,tsx}',
      customSyntax: 'postcss-styled-syntax',
      rules: {
        'function-no-unknown': [true, { ignoreFunctions: ['${', '/theme.+/'] }],
        'value-keyword-case': undefined,
        'value-no-vendor-prefix': [true, { ignoreValues: ['box'] }],
      },
    },
    {
      files: '**/*.scss',
      customSyntax: 'postcss-scss',
      rules: {
        'at-rule-no-unknown': [true, { ignoreAtRules: ['tailwind', 'use'] }],
        'function-no-unknown': [true, { ignoreFunctions: ['darken'] }],
        'selector-pseudo-class-no-unknown': [true, { ignorePseudoClasses: ['global'] }],
      },
    },
  ],
  rules: {
    'function-no-unknown': [true, { ignoreFunctions: ['theme'] }],
    'import-notation': 'string',
    'selector-class-pattern': undefined,
    'selector-id-pattern': undefined,
  },
};
