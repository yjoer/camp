const config = require('@camp/config/lint-staged.cjs');

module.exports = {
  ...config,
  '*.rs': [
    () => 'cargo clippy -- --deny warnings', //
    () => 'cargo fmt --check',
  ],
};
