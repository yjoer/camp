/* eslint-disable import-x/no-extraneous-dependencies */
// oxlint-disable import/no-default-export
import { defineConfig } from 'eslint/config';

import { defineImportResolver, eslint_config } from '@xcamp/config/eslint.js';

process.chdir(import.meta.dirname);

export default defineConfig([
  ...eslint_config,
  ...defineImportResolver(['examples/tanstack/start', 'examples/tanstack/start-solid', 'examples/tanstack/start-trpc']),
]);
