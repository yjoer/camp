/* eslint-disable import/no-extraneous-dependencies */
// @ts-expect-error jiti
import pkg from '@camp/webpack/config.ts';

import type { getServerConfig as GSC } from '@camp/webpack/config.ts';
import type { Configuration } from 'webpack';

const { getServerConfig } = pkg as { getServerConfig: typeof GSC };

const config = getServerConfig({
  entry: ['./server.ts'],
  projectPath: import.meta.dirname,
  configPath: import.meta.filename,
}) satisfies Configuration;

export default config;
