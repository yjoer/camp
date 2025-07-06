import { about } from './handlers/about.ts';
import { home } from './handlers/home.ts';
import { posts } from './handlers/posts.ts';

import type { IncomingMessage, ServerResponse } from 'node:http';

const handlers = {
  home,
  posts,
  about,
};

export async function router(req: IncomingMessage, res: ServerResponse) {
  if (req.url === '/') {
    await handlers.home(req, res);
  } else if (req.url.startsWith('/posts')) {
    await handlers.posts(req, res);
  } else if (req.url.startsWith('/about')) {
    await handlers.about(req, res);
  } else {
    res.end();
  }
}

const enabled = true;
if (enabled && import.meta.webpackHot) {
  import.meta.webpackHot.accept('./handlers/home.ts', async () => {
    const mod = await import('./handlers/home.ts');
    handlers.home = mod.home;
  });

  import.meta.webpackHot.accept('./handlers/posts.ts', async () => {
    const mod = await import('./handlers/posts.ts');
    handlers.posts = mod.posts;
  });

  import.meta.webpackHot.accept('./handlers/about.ts', async () => {
    const mod = await import('./handlers/about.ts');
    handlers.about = mod.about;
  });
}
