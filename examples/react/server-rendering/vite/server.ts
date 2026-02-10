/* eslint-disable import-x/no-unresolved */
/* eslint-disable import-x/no-extraneous-dependencies */
// oxlint-disable no-console
import { createReadStream } from 'node:fs';
import fs from 'node:fs/promises';
import { createServer } from 'node:http';
import path from 'node:path';

import type { IncomingMessage, ServerResponse } from 'node:http';
import type { ViteDevServer } from 'vite';

const isDevelopment = process.env.NODE_ENV !== 'production';

const server = createServer();

let vite: ViteDevServer;
if (isDevelopment) {
  const { createServer: createViteServer } = await import('vite');
  vite = await createViteServer({
    server: { middlewareMode: { server } },
    appType: 'custom',
  });
}

const p = path.resolve(import.meta.dirname, '.output', 'client', 'index.html');
let template = isDevelopment ? '' : await fs.readFile(p, 'utf8');

server.on('request', async function (req, res) {
  try {
    const url = req.url ?? '';

    let render;
    if (isDevelopment) {
      await new Promise(resolve => vite.middlewares(req, res, resolve));

      template = await fs.readFile(path.resolve(import.meta.dirname, 'index.html'), 'utf8');
      template = await vite.transformIndexHtml(url, template);
      ({ render } = await vite.ssrLoadModule('/src/entry-server.tsx'));
    } else {
      if (url.startsWith('/assets/')) return staticFiles(req, res);

      // @ts-expect-error build-time generated
      ({ render } = await import('./.output/server/entry-server.js'));
    }

    let html = await render(url);
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
});

async function staticFiles(req: IncomingMessage, res: ServerResponse) {
  const url = req.url ?? '/';
  const assetPath = path.resolve(import.meta.dirname, '.output', 'client', url.slice(1));

  try {
    const stat = await fs.stat(assetPath);
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

  const stream = createReadStream(assetPath);
  stream.pipe(res);
}

server.listen({ port: 3000 }, function () {
  console.log('server listening on port 3000');
});
