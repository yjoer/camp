// oxlint-disable import/no-default-export
import babel from '@rolldown/plugin-babel';
import tailwindcss from '@tailwindcss/vite';
import { tanstackStart } from '@tanstack/react-start/plugin/vite';
import { rootRoute, route } from '@tanstack/virtual-file-routes';
import react from '@vitejs/plugin-react';
import { nitro } from 'nitro/vite';
import { defineConfig } from 'vite';

export const routes = rootRoute('root.tsx', [
	route('/', '../src/gallery.tsx'),
	route('/stale-closures', '../src/stale-closures.tsx'),
	route('/external-stores/redux', '../src/external-stores/redux.tsx'),
	route('/external-stores/redux-toolkit', '../src/external-stores/redux-toolkit.tsx'),
	route('/external-stores/zustand', '../src/external-stores/zustand.tsx'),
	route('/transition-use-state', '../src/transition-use-state.tsx'),
	route('/transition-use-context-selector', '../src/transition-use-context-selector.tsx'),
	route('/transition-use-search', '../src/transition-use-search.tsx'),
	route('/transition-redux', '../src/transition-redux.tsx'),
	route('/transition-zustand', '../src/transition-zustand.tsx'),
	route('/list-virt-fixed-height', '../src/list-virt/fixed-height.tsx'),
	route('/list-virt-dynamic-height', '../src/list-virt/dynamic-height.tsx'),
	route('/custom-elements', '../src/custom-elements.tsx'),
	route('/media-source-extensions', '../src/media-source-extensions.tsx'),
	route('/worker-offscreen-canvas', '../src/offscreen-canvas.tsx'),
	route('/webgl/triangle', '../src/webgl/triangle.tsx'),
	route('/webgl/rectangle', '../src/webgl/rectangle.tsx'),
	route('/webgl/multiple-rectangles', '../src/webgl/multiple-rectangles.tsx'),
	route('/ui/animations/sidebar', '../ui/animation-sidebar.tsx'),
]);

export const babelConfig = {
	plugins: [
		['@stylexjs/babel-plugin', {
			debug: process.env.NODE_ENV === 'development',
			unstable_moduleResolution: { type: 'commonJS' },
		}],
	],
	parserOpts: {
		plugins: ['jsx', 'typescript'],
	},
} satisfies Parameters<typeof babel>[0];

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
		react(),
		babel(babelConfig),
		tailwindcss(),
	],
});
