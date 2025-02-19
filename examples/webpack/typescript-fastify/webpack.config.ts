/* eslint-disable import/extensions */
/* eslint-disable import/no-extraneous-dependencies */
import { getServerConfig } from '@camp/webpack/config.ts';

import type { Configuration } from 'webpack';

const config = getServerConfig({
  projectPath: import.meta.dirname,
  configPath: import.meta.filename,
}) satisfies Configuration;

export default config;
