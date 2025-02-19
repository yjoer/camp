import { createRequire } from 'node:module';
import path from 'node:path';

import webpack from 'webpack';
import nodeExternals from 'webpack-node-externals';

import { AssetRelocatorCachePlugin, RunScriptPlugin } from './plugins.ts';

import type { Configuration } from 'webpack';

const require = createRequire(import.meta.url);

interface Options {
  entry?: string[];
  projectPath: string;
  configPath: string;
  transpilePackages?: (string | RegExp)[];
}

export const getServerConfig = ({
  entry = [],
  projectPath,
  configPath,
  transpilePackages = [],
}: Options) => {
  // Convert strings into regular expressions for exact matches.
  for (let i = 0; i < transpilePackages.length; i++) {
    if (typeof transpilePackages[i] === 'string') {
      transpilePackages[i] = new RegExp(transpilePackages[i]);
    }
  }

  const config = {
    entry: ['webpack/hot/poll?100', ...entry],
    mode: 'development',
    output: {
      path: path.join(projectPath, 'node_modules', '.camp', 'build'),
      filename: 'server.js',
      clean: true,
    },
    module: {
      rules: [
        {
          test: /\.(js|ts|node)$/,
          parser: { amd: false },
          use: {
            loader: require.resolve('@vercel/webpack-asset-relocator-loader'),
          },
        },
        {
          test: /\.ts$/,
          use: {
            loader: require.resolve('ts-loader'),
            options: {
              transpileOnly: true,
            },
          },
          exclude: [
            {
              and: [/node_modules/],
              not: transpilePackages,
            },
          ],
        },
      ],
    },
    resolve: {
      extensions: ['.js', '.ts'],
    },
    plugins: [
      new webpack.HotModuleReplacementPlugin(),
      new AssetRelocatorCachePlugin(),
      new RunScriptPlugin(),
    ],
    cache: {
      buildDependencies: {
        config: [import.meta.filename, configPath],
      },
      type: 'filesystem',
    },
    target: 'node',
    externals: [
      nodeExternals({
        allowlist: ['webpack/hot/poll?100', ...transpilePackages],
      }),
    ],
  } satisfies Configuration;

  return config;
};
