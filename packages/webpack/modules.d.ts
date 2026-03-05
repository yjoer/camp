declare module '@vercel/webpack-asset-relocator-loader' {
  import type { Compilation } from 'webpack';

  function initAssetCache(compilation: Compilation, outputAssetBase: string): void;
}
