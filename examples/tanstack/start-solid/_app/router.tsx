/* eslint-disable import-x/extensions */
/* eslint-disable import-x/no-unresolved */
import { createRouter as createTanStackRouter } from '@tanstack/solid-router';

import { routeTree } from './routeTree.gen';

export function getRouter() {
  return createTanStackRouter({
    routeTree,
    defaultPreload: 'intent',
    defaultPreloadStaleTime: 0,
    scrollRestoration: true,
  });
}
