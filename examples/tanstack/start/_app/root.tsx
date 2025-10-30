/// <reference types="vite/client" />
import { createRootRoute, HeadContent, Outlet, Scripts } from '@tanstack/react-router';

import ReduxProvider from '@/lib/ReduxProvider';

import styles from './root.css?url';
import stylex from './stylex.css?url';

export const Route = createRootRoute({
  head: () => ({
    meta: [
      { charSet: 'utf8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1' },
    ],
    links: [
      { rel: 'stylesheet', href: styles }, //
      { rel: 'stylesheet', href: stylex },
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
        <HeadContent />
      </head>
      <body>
        <ReduxProvider>
          <Outlet />
        </ReduxProvider>
        <Scripts />
      </body>
    </html>
  );
}
