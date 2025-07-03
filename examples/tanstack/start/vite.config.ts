import path from 'node:path';

import tailwindcss from '@tailwindcss/vite';
import { tanstackStart } from '@tanstack/react-start/plugin/vite';
import { rootRoute, route } from '@tanstack/virtual-file-routes';
import { defineConfig } from 'vite';

const routes = rootRoute('_app/root.tsx', [
  route('/', 'src/Gallery.tsx'),
  route('/stale-closures', 'src/StaleClosures.tsx'),
  route('/transition-use-state', 'src/TransitionUseState.tsx'),
  route('/transition-use-context-selector', 'src/TransitionUseContextSelector.tsx'),
  route('/transition-use-search', 'src/TransitionUseSearch.tsx'),
  route('/transition-redux', 'src/TransitionRedux.tsx'),
  route('/transition-zustand', 'src/TransitionZustand.tsx'),
]);

export default defineConfig({
  resolve: {
    alias: {
      '@/lib': path.resolve(import.meta.dirname, 'lib'),
      '@/state': path.resolve(import.meta.dirname, 'state'),
    },
  },
  server: {
    port: 3000,
  },
  plugins: [
    tanstackStart({
      target: process.env.TSS_TARGET,
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
