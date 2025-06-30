/* eslint-disable import-x/no-extraneous-dependencies */
import * as pkg from '@camp/webpack/config.ts';

import type { getServerConfig as GSC } from '@camp/webpack/config.ts';

// @ts-expect-error jiti
// oxlint-disable-next-line namespace
const { getServerConfig } = pkg.default as { getServerConfig: typeof GSC };

const mode = process.env.NODE_ENV === 'production' ? 'production' : 'development';

const config = getServerConfig({
  entry: ['./server.ts'],
  mode,
  projectPath: import.meta.dirname,
  configPath: import.meta.filename,
});

export default config;
