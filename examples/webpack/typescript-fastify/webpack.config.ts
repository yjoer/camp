import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { getServerConfig } from '@camp/webpack/config.ts';

import type { Configuration } from 'webpack';

const config = getServerConfig({
  entry: ['./server.ts'],
  projectPath: path.dirname(fileURLToPath(import.meta.url)),
  configPath: fileURLToPath(import.meta.url),
}) satisfies Configuration;

export default config;
