import config from '@camp/config/lint-staged.js';

export default {
  ...config,
  '*.rs': [
    () => 'cargo clippy -- --deny warnings', //
    () => 'cargo fmt --check',
  ],
};
