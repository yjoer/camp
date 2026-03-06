// oxlint-disable import/no-default-export
import tailwindcss from '@tailwindcss/vite';
import { tanstackStart } from '@tanstack/solid-start/plugin/vite';
import { rootRoute, route } from '@tanstack/virtual-file-routes';
import { nitro } from 'nitro/vite';
import { defineConfig } from 'vite';
import solid from 'vite-plugin-solid';

const routes = rootRoute('root.tsx', [
  route('/', '../src/gallery.tsx'),
  route('/stale-closures', '../src/stale-closures.tsx'),
  route('/transition-signal', '../src/transition-signal.tsx'),
]);

export default defineConfig({
  resolve: {
    tsconfigPaths: true,
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
    nitro({ preset: process.env.TSS_TARGET }),
    solid({ ssr: true }),
    tailwindcss(),
  ],
});
