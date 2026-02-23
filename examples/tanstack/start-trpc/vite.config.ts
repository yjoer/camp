// oxlint-disable import/no-default-export
import tailwindcss from '@tailwindcss/vite';
import { nitroV2Plugin } from '@tanstack/nitro-v2-vite-plugin';
import { tanstackStart } from '@tanstack/react-start/plugin/vite';
import { rootRoute, route } from '@tanstack/virtual-file-routes';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

const routes = rootRoute('root.tsx', [
  route('/', '../src/landing-page.tsx'),
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
    nitroV2Plugin({ preset: process.env.TSS_TARGET }),
    react({ babel: { configFile: true } }),
    tailwindcss(),
  ],
});
