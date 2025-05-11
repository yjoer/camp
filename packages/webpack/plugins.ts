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
    compiler.hooks.afterEmit.tap('RunScriptPlugin', () => {
      if (this.subprocess?.connected) return;

      const { path: outputPath, filename } = compiler.options.output;
      if (!outputPath) return;

      const entrypoint = path.join(outputPath, filename as string);
      this.subprocess = cp.fork(entrypoint);
    });
  }
}
