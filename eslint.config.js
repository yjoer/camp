/* eslint-disable import-x/no-extraneous-dependencies */
import config, { defineImportResolver } from '@camp/config/eslint.js';
import { defineConfig } from 'eslint/config';

export default defineConfig([
  ...config,
  ...defineImportResolver(['examples/tanstack/start', 'examples/tanstack/start-trpc']),
]);
