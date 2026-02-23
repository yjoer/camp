// oxlint-disable import/no-default-export
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { rspack } from '@rspack/core';
import { getServerConfig } from '@xcamp/webpack/rspack-config.ts';
import { merge } from 'webpack-merge';

const mode = process.env.NODE_ENV === 'production' ? 'production' : 'development';

const ignored_packages: Record<string, boolean> = {
  'aws-sdk': true,
  'mock-aws-s3': true,
  'nock': true,
};

const custom_config = {
  plugins: [
    new rspack.IgnorePlugin({
      checkResource: resource => !!ignored_packages[resource],
    }),
  ],
};

const config = getServerConfig({
  entry: ['./server.ts'],
  mode,
  projectPath: path.dirname(fileURLToPath(import.meta.url)),
  configPath: fileURLToPath(import.meta.url),
});

export default merge(config, custom_config);
