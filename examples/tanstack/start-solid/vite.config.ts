import tailwindcss from '@tailwindcss/vite';
import { nitroV2Plugin } from '@tanstack/nitro-v2-vite-plugin';
import { tanstackStart } from '@tanstack/solid-start/plugin/vite';
import { rootRoute, route } from '@tanstack/virtual-file-routes';
import { defineConfig } from 'vite';
import solid from 'vite-plugin-solid';

const routes = rootRoute('root.tsx', [
  route('/', '../src/Gallery.tsx'), //
  route('/stale-closures', '../src/StaleClosures.tsx'),
  route('/transition-signal', '../src/TransitionSignal.tsx'),
]);

export default defineConfig({
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
    solid({ ssr: true }),
    tailwindcss(),
  ],
});
