// oxlint-disable no-console
// oxlint-disable no-null
// oxlint-disable no-this-alias
// oxlint-disable no-this-assignment
import cp from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// @ts-expect-error no-types-available
import relocateLoader from '@vercel/webpack-asset-relocator-loader';

import type { Compiler, Resolver } from 'webpack';

export class AssetRelocatorCachePlugin {
  apply(compiler: Compiler) {
    const outputAssetBase = 'assets';

    compiler.hooks.compilation.tap('AssetRelocatorCachePlugin', (compilation) => {
      relocateLoader.initAssetCache(compilation, outputAssetBase);
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
      const asset = assets.find(asset => asset.name === `${entry}.js`);
      if (!asset) return;

      // const { filename } = compiler.options.output;
      const { path: outputPath } = compiler.options.output;
      if (!outputPath) return;

      const entrypoint = path.join(outputPath, asset.name);
      this.subprocess = cp.fork(entrypoint);
    });
  }
}

export class OptionalModulesPlugin {
  apply(compiler: Compiler) {
    const logger = compiler.getInfrastructureLogger('OptionalModulesPlugin');

    compiler.resolverFactory.hooks.resolver
    .for('normal')
    .tap('OptionalModulesPlugin', (resolver) => {
      const plugin = new OptionalModulesResolverPlugin(logger);
      plugin.apply(resolver);
    });
  }
}

export class OptionalModulesResolverPlugin {
  logger: ReturnType<Compiler['getInfrastructureLogger']>;

  constructor(logger: ReturnType<Compiler['getInfrastructureLogger']>) {
    this.logger = logger;
  }

  apply(resolver: Resolver) {
    const self = this;
    const dirname = path.dirname(fileURLToPath(import.meta.url));
    const resolve = resolver.resolve;

    resolver.resolve = function (context, fp, request, resolveContext, callback) {
      const boundResolve: typeof resolve = resolve.bind(this);

      boundResolve(context, fp, request, resolveContext, (err, innerPath, result) => {
        if (result) return callback(null, innerPath, result);
        if (err && !err.message.startsWith("Can't resolve")) return callback(err);

        const issuer = (context as any)?.issuer;
        const fromTS = issuer?.endsWith('.ts') || issuer?.endsWith('.tsx');

        if (request.endsWith('.js') && fromTS) {
          return boundResolve(
            context,
            fp,
            request.slice(0, -3),
            resolveContext,
            (err, innerPath, result) => {
              if (result) return callback(null, innerPath, result);
              if (err && !err.message.startsWith("Can't resolve")) return callback(err);

              const ctx = { path: request };
              callback(null, path.join(dirname, `__missing.js?${request}`), ctx);
              self.logger.warn(`${request} is missing, using __missing.js instead.`);
            },
          );
        }

        const ctx = { path: request };
        callback(null, path.join(dirname, `__missing.js?${request}`), ctx);
        self.logger.warn(`${request} is missing, using __missing.js instead.`);
      });
    };
  }
}
