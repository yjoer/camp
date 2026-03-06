// oxlint-disable import/no-default-export
import tailwindcss from '@tailwindcss/vite';
import { tanstackStart } from '@tanstack/react-start/plugin/vite';
import { rootRoute, route } from '@tanstack/virtual-file-routes';
import react from '@vitejs/plugin-react';
import { nitro } from 'nitro/vite';
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
    nitro({ preset: process.env.TSS_TARGET, serverEntry: false }),
    react({ babel: { configFile: true } }),
    tailwindcss(),
  ],
});
