import * as pkg from '@camp/webpack/config.ts';

import type { getServerConfig as GSC } from '@camp/webpack/config.ts';
import type { Configuration } from 'webpack';

// @ts-expect-error jiti
// oxlint-disable-next-line namespace
const { getServerConfig } = pkg.default as { getServerConfig: typeof GSC };

const config = getServerConfig({
  entry: ['./server.ts'],
  projectPath: import.meta.dirname,
  configPath: import.meta.filename,
}) satisfies Configuration;

export default config;
