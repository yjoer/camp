import { createRequire } from 'node:module';
import path from 'node:path';
// import { fileURLToPath } from 'node:url';

import { rspack } from '@rspack/core';
import { merge } from 'webpack-merge';
// import nodeExternals from 'webpack-node-externals';

import { AssetRelocatorCachePlugin, RunScriptPlugin } from './plugins.ts';

import type { Configuration } from '@rspack/core';

const require = createRequire(import.meta.url);

interface Options {
  configPath: string;
  entry: Configuration['entry'];
  minimize?: boolean;
  mode: 'development' | 'production';
  projectPath: string;
  transpilePackages?: (string | RegExp)[];
}

export const getServerConfig = ({
  entry = [],
  mode,
  projectPath,
  // configPath,
  transpilePackages = [],
  minimize = false,
}: Options): Configuration => {
  // Convert strings into regular expressions for exact matches.
  for (let i = 0; i < transpilePackages.length; i++) {
    if (typeof transpilePackages[i] === 'string') {
      transpilePackages[i] = new RegExp(transpilePackages[i]);
    }
  }

  const commonConfig: Configuration = {
    entry,
    mode,
    module: {
      rules: [
        {
          test: /\.(js|ts|node)$/,
          parser: { amd: false },
          use: {
            loader: require.resolve('@vercel/webpack-asset-relocator-loader'),
            options: {
              outputAssetBase: 'assets',
            },
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
              and: [path.resolve(projectPath, 'node_modules')],
              not: transpilePackages,
            },
          ],
        },
      ],
    },
    resolve: {
      extensions: ['.js', '.ts'],
      tsConfig: path.resolve(projectPath, 'tsconfig.json'),
    },
    plugins: [
      new rspack.ProgressPlugin(), //
      new AssetRelocatorCachePlugin(),
    ],
    target: 'node',
  };

  const developmentConfig: Configuration = {
    entry: ['@rspack/core/hot/poll?100'],
    output: {
      path: path.join(projectPath, 'node_modules', '.camp', 'build'),
      filename: '[name].js',
      clean: true,
    },
    plugins: [
      new rspack.HotModuleReplacementPlugin(), //
      new RunScriptPlugin(),
    ],
    // cache: {
    //   buildDependencies: {
    //     config: [fileURLToPath(import.meta.url), configPath],
    //   },
    //   type: 'filesystem',
    // },
    externals: [
      // nodeExternals({
      //   allowlist: ['@rspack/core/hot/poll?100', ...transpilePackages],
      // }),
    ],
  };

  const productionConfig: Configuration = {
    output: {
      path: path.join(projectPath, '.camp', 'build'),
      filename: '[name].js',
      clean: true,
    },
    optimization: {
      minimize,
    },
    experiments: {
      outputModule: true,
    },
  };

  if (mode === 'development') {
    return merge(commonConfig, developmentConfig);
  } else if (mode === 'production') {
    return merge(commonConfig, productionConfig);
  }

  return {};
};
