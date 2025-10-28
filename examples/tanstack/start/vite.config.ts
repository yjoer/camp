import path from 'node:path';

import tailwindcss from '@tailwindcss/vite';
import { nitroV2Plugin } from '@tanstack/nitro-v2-vite-plugin';
import { tanstackStart } from '@tanstack/react-start/plugin/vite';
import { rootRoute, route } from '@tanstack/virtual-file-routes';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

const routes = rootRoute('root.tsx', [
  route('/', '../src/Gallery.tsx'),
  route('/stale-closures', '../src/StaleClosures.tsx'),
  route('/transition-use-state', '../src/TransitionUseState.tsx'),
  route('/transition-use-context-selector', '../src/TransitionUseContextSelector.tsx'),
  route('/transition-use-search', '../src/TransitionUseSearch.tsx'),
  route('/transition-redux', '../src/TransitionRedux.tsx'),
  route('/transition-zustand', '../src/TransitionZustand.tsx'),
  route('/custom-elements', '../src/CustomElements.tsx'),
  route('/media-source-extensions', '../src/MediaSourceExtensions.tsx'),
  route('/worker-offscreen-canvas', '../src/OffscreenCanvas.tsx'),
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
      srcDirectory: '_app',
      router: {
        virtualRouteConfig: routes,
        routesDirectory: '.',
      },
    }),
    nitroV2Plugin({ preset: process.env.TSS_TARGET }),
    react(),
    tailwindcss(),
  ],
});
