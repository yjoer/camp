/// <reference types="vite/client" />
import { createRootRoute, HeadContent, Outlet, Scripts } from '@tanstack/solid-router';
import * as Solid from 'solid-js';
import { HydrationScript } from 'solid-js/web';

import styles from './root.css?url';

export const Route = createRootRoute({
  head: () => ({
    meta: [
      { charSet: 'utf8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1' },
    ],
    links: [
      { rel: 'stylesheet', href: styles }, //
    ],
  }),
  component: RootComponent,
  notFoundComponent: NotFound,
});

function NotFound() {
  return <div>Not Found</div>;
}

function RootComponent() {
  return (
    <html lang="en">
      <head>
        <HydrationScript />
      </head>
      <body>
        <HeadContent />
        <Solid.Suspense>
          <Outlet />
        </Solid.Suspense>
        <Scripts />
      </body>
    </html>
  );
}
