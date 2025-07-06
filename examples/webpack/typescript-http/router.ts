import { about } from './handlers/about.ts';
import { home } from './handlers/home.ts';
import { posts } from './handlers/posts.ts';

import type { IncomingMessage, ServerResponse } from 'node:http';

export async function router(req: IncomingMessage, res: ServerResponse) {
  if (req.url === '/') {
    await home(req, res);
  } else if (req.url.startsWith('/posts')) {
    await posts(req, res);
  } else if (req.url.startsWith('/about')) {
    await about(req, res);
  } else {
    res.end();
  }
}

const enabled = true;
if (enabled && import.meta.webpackHot) {
  import.meta.webpackHot.accept('./handlers/home.ts');
  import.meta.webpackHot.accept('./handlers/posts.ts');
  import.meta.webpackHot.accept('./handlers/about.ts');
}
