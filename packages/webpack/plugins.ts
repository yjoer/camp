import cp from 'node:child_process';
import path from 'node:path';

import relocateLoader from '@vercel/webpack-asset-relocator-loader';

import type { Compiler } from 'webpack';

export class AssetRelocatorCachePlugin {
  apply(compiler: Compiler) {
    compiler.hooks.compilation.tap('AssetRelocatorCachePlugin', (compilation) => {
      relocateLoader.initAssetCache(compilation);
    });
  }
}

export class RunScriptPlugin {
  subprocess?: cp.ChildProcess;

  apply(compiler: Compiler) {
    compiler.hooks.afterEmit.tap('RunScriptPlugin', (compilation) => {
      if (this.subprocess?.connected) return;

      // find the asset of the first entry file.
      const entry = compilation.entrypoints.keys().next().value;
      const assets = compilation.getAssets();
      const asset = assets.find((asset) => asset.name === `${entry}.js`);
      if (!asset) return;

      // const { filename } = compiler.options.output;
      const { path: outputPath } = compiler.options.output;
      if (!outputPath) return;

      const entrypoint = path.join(outputPath, asset.name);
      this.subprocess = cp.fork(entrypoint);
    });
  }
}
