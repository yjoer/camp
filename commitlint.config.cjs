module.exports = {
  extends: [
    '@commitlint/config-conventional', //
  ],
  rules: {
    'header-max-length': [2, 'always', 100],
    'subject-case': [2, 'always', 'lower-case'],
  },
};
