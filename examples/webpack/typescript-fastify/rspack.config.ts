/* eslint-disable import-x/no-extraneous-dependencies */
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { getServerConfig } from '@camp/webpack/rspack-config.ts';

const mode = process.env.NODE_ENV === 'production' ? 'production' : 'development';

const config = getServerConfig({
  entry: ['./server.ts'],
  mode,
  projectPath: path.dirname(fileURLToPath(import.meta.url)),
  configPath: fileURLToPath(import.meta.url),
});

export default config;
