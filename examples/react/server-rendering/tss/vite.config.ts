// oxlint-disable import/no-default-export
import { tanstackStart } from '@tanstack/react-start/plugin/vite';
import { rootRoute, route } from '@tanstack/virtual-file-routes';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

const routes = rootRoute('_app/root.tsx', [
  route('/', 'home.tsx'),
]);

export default defineConfig({
  server: {
    port: 3000,
  },
  plugins: [
    tanstackStart({
      customViteReactPlugin: true,
      tsr: {
        virtualRouteConfig: routes,
        verboseFileRoutes: false,
        routesDirectory: '.',
        srcDirectory: '_app',
      },
    }),
    react(),
  ],
});
