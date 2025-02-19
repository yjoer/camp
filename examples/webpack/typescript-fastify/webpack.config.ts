/* eslint-disable import/no-extraneous-dependencies */
import { getServerConfig } from '@camp/webpack/config.ts';

import type { Configuration } from 'webpack';

const config = getServerConfig({
  entry: ['./server.ts'],
  projectPath: import.meta.dirname,
  configPath: import.meta.filename,
}) satisfies Configuration;

export default config;
