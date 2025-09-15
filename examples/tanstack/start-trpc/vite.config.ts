import path from 'node:path';

import tailwindcss from '@tailwindcss/vite';
import { tanstackStart } from '@tanstack/react-start/plugin/vite';
import { rootRoute, route } from '@tanstack/virtual-file-routes';
import { defineConfig } from 'vite';

const routes = rootRoute('_app/root.tsx', [
  route('/', 'src/LandingPage.tsx'), //
]);

export default defineConfig({
  resolve: {
    alias: {
      '@/lib': path.resolve(import.meta.dirname, 'lib'),
    },
  },
  server: {
    port: 3000,
  },
  plugins: [
    tanstackStart({
      tsr: {
        virtualRouteConfig: routes,
        verboseFileRoutes: false,
        routesDirectory: '.',
        srcDirectory: '_app',
      },
    }),
    tailwindcss(),
  ],
});
