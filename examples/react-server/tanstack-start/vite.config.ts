// oxlint-disable import/no-default-export
import { tanstackStart } from '@tanstack/react-start/plugin/vite';
import { rootRoute, route } from '@tanstack/virtual-file-routes';
import react from '@vitejs/plugin-react';
import { nitro } from 'nitro/vite';
import { defineConfig } from 'vite';

const routes = rootRoute('root.tsx', [
  route('/', '../home.tsx'),
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
    nitro(),
    react(),
  ],
});
