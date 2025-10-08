/* eslint-disable import-x/no-extraneous-dependencies */
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { getServerConfig } from '@camp/webpack/rspack-config.ts';
import { rspack } from '@rspack/core';
import { merge } from 'webpack-merge';

const mode = process.env.NODE_ENV === 'production' ? 'production' : 'development';

const ignoredPackages = {
  'aws-sdk': true,
  'mock-aws-s3': true,
  nock: true,
};

const customConfig = {
  plugins: [
    new rspack.IgnorePlugin({
      checkResource: (resource) => !!ignoredPackages[resource],
    }),
  ],
};

const config = getServerConfig({
  entry: ['./server.ts'],
  mode,
  projectPath: path.dirname(fileURLToPath(import.meta.url)),
  configPath: fileURLToPath(import.meta.url),
});

export default merge(config, customConfig);
