// oxlint-disable no-console
import { createReadStream } from 'node:fs';
import fs from 'node:fs/promises';
import { createServer } from 'node:http';
import path from 'node:path';

import type * as EntryServer from './src/entry-server';
import type { IncomingMessage, ServerResponse } from 'node:http';
import type { ViteDevServer } from 'vite';

const is_development = process.env.NODE_ENV !== 'production';
const server = createServer();

let vite: ViteDevServer;
if (is_development) {
  // eslint-disable-next-line import-x/no-extraneous-dependencies
  const { createServer: create_vite_server } = await import('vite');

  vite = await create_vite_server({
    server: { middlewareMode: { server } },
    appType: 'custom',
  });
}

const p = path.resolve(import.meta.dirname, '.output', 'client', 'index.html');
let template = is_development ? '' : await fs.readFile(p, 'utf8');

server.on('request', function (req, res) {
  void handler(req, res);
});

async function handler(req: IncomingMessage, res: ServerResponse) {
  try {
    const url = req.url ?? '';

    let render: typeof EntryServer.render;
    if (is_development) {
      await new Promise(resolve => vite.middlewares(req, res, resolve));

      template = await fs.readFile(path.resolve(import.meta.dirname, 'index.html'), 'utf8');
      template = await vite.transformIndexHtml(url, template);
      ({ render } = await vite.ssrLoadModule('/src/entry-server.tsx') as typeof EntryServer);
    } else {
      if (url.startsWith('/assets/')) return static_files(req, res);

      // @ts-expect-error build-time generated
      // eslint-disable-next-line import-x/no-unresolved
      ({ render } = await import('./.output/server/entry-server.js') as typeof EntryServer);
    }

    let html = render();
    html = template.replace('<!--app-html-->', () => html);

    res.statusCode = 200;
    res.setHeader('Content-Type', 'text/html');
    res.end(html);
  } catch (error) {
    if (!(error instanceof Error)) return;
    vite?.ssrFixStacktrace(error);
    res.statusCode = 500;
    res.end(error.stack);
  }
}

async function static_files(req: IncomingMessage, res: ServerResponse) {
  const url = req.url ?? '/';
  const asset_path = path.resolve(import.meta.dirname, '.output', 'client', url.slice(1));

  try {
    const stat = await fs.stat(asset_path);
    if (!stat.isFile()) {
      res.statusCode = 404;
      res.end('not found');
      return;
    }
  } catch {
    res.statusCode = 404;
    res.end('not found');
    return;
  }

  res.statusCode = 200;
  if (url.endsWith('.js')) res.setHeader('Content-Type', 'text/javascript');
  else if (url.endsWith('.css')) res.setHeader('Content-Type', 'text/css');

  const stream = createReadStream(asset_path);
  stream.pipe(res);
}

server.listen({ port: 3000 }, function () {
  console.log('server listening on port 3000');
});
