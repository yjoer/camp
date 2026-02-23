/* eslint-disable import-x/extensions */
/* eslint-disable import-x/no-unresolved */
import { QueryClient } from '@tanstack/react-query';
import { createRouter } from '@tanstack/react-router';
import { setupRouterSsrQueryIntegration } from '@tanstack/react-router-ssr-query';
import { createIsomorphicFn } from '@tanstack/react-start';
import { getRequestHeader } from '@tanstack/react-start/server';
import { createTRPCClient, httpBatchLink } from '@trpc/client';

import { TRPCProvider } from '@/lib/trpc-provider';

import { routeTree } from './routeTree.gen';

import type { AppRouter } from '../server';

export function getRouter() {
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

  const router = createRouter({
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

const getCookie = createIsomorphicFn().server(() => ({ cookie: getRequestHeader('cookie') }));
