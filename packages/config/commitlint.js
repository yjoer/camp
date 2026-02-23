/** @type {import('@commitlint/types').UserConfig} */
export const commitlint_config = {
  extends: [
    '@commitlint/config-conventional', //
  ],
  rules: {
    'header-max-length': [2, 'always', 100],
    'subject-case': [2, 'always', 'lower-case'],
  },
};
