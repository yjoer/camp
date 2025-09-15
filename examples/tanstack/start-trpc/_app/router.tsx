/* eslint-disable import-x/extensions */
/* eslint-disable import-x/no-unresolved */
import { QueryClient } from '@tanstack/react-query';
import { createRouter as createTanStackRouter } from '@tanstack/react-router';
import { setupRouterSsrQueryIntegration } from '@tanstack/react-router-ssr-query';
import { createIsomorphicFn } from '@tanstack/react-start';
import { getHeader } from '@tanstack/react-start/server';
import { createTRPCClient, httpBatchLink } from '@trpc/client';

import { TRPCProvider } from '@/lib/TRPCProvider';

import { routeTree } from './routeTree.gen';

import type { AppRouter } from '../server';

export function createRouter() {
  const queryClient = new QueryClient();

  const trpcClient = createTRPCClient<AppRouter>({
    links: [
      httpBatchLink({
        url: 'http://localhost:3001/trpc',
        fetch(url, options) {
          return fetch(url, {
            ...options,
            headers: {
              ...options?.headers,
              ...getCookie(),
            },
            credentials: 'include',
          });
        },
      }),
    ],
  });

  const router = createTanStackRouter({
    routeTree,
    defaultPreload: 'intent',
    defaultPreloadStaleTime: 0,
    scrollRestoration: true,
    Wrap: ({ children }: { children: React.ReactNode }) => {
      return (
        <TRPCProvider queryClient={queryClient} trpcClient={trpcClient}>
          {children}
        </TRPCProvider>
      );
    },
  });

  setupRouterSsrQueryIntegration({ router, queryClient });

  return router;
}

const getCookie = createIsomorphicFn().server(() => ({ cookie: getHeader('cookie') }));

declare module '@tanstack/react-router' {
  interface Register {
    router: ReturnType<typeof createRouter>;
  }
}
