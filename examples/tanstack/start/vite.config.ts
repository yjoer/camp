import tailwindcss from '@tailwindcss/vite';
import { nitroV2Plugin } from '@tanstack/nitro-v2-vite-plugin';
import { tanstackStart } from '@tanstack/react-start/plugin/vite';
import { rootRoute, route } from '@tanstack/virtual-file-routes';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

const routes = rootRoute('root.tsx', [
  route('/', '../src/Gallery.tsx'),
  route('/stale-closures', '../src/StaleClosures.tsx'),
  route('/external-stores/redux', '../src/external-stores/Redux.tsx'),
  route('/external-stores/redux-toolkit', '../src/external-stores/ReduxToolkit.tsx'),
  route('/external-stores/zustand', '../src/external-stores/Zustand.tsx'),
  route('/transition-use-state', '../src/TransitionUseState.tsx'),
  route('/transition-use-context-selector', '../src/TransitionUseContextSelector.tsx'),
  route('/transition-use-search', '../src/TransitionUseSearch.tsx'),
  route('/transition-redux', '../src/TransitionRedux.tsx'),
  route('/transition-zustand', '../src/TransitionZustand.tsx'),
  route('/list-virt-fixed-height', '../src/list-virt/FixedHeight.tsx'),
  route('/list-virt-dynamic-height', '../src/list-virt/DynamicHeight.tsx'),
  route('/custom-elements', '../src/CustomElements.tsx'),
  route('/media-source-extensions', '../src/MediaSourceExtensions.tsx'),
  route('/worker-offscreen-canvas', '../src/OffscreenCanvas.tsx'),
  route('/webgl/triangle', '../src/webgl/Triangle.tsx'),
  route('/webgl/rectangle', '../src/webgl/Rectangle.tsx'),
  route('/webgl/multiple-rectangles', '../src/webgl/MultipleRectangles.tsx'),
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
