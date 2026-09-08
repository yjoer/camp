// oxlint-disable import/no-default-export
import babel from '@rolldown/plugin-babel';
import { tanstackStart } from '@tanstack/react-start/plugin/vite';
import { rootRoute, route } from '@tanstack/virtual-file-routes';
import react from '@vitejs/plugin-react';
import { nitro } from 'nitro/vite';
import { defineConfig } from 'vite-plus';

export const routes = rootRoute('root.tsx', [
	route('/', '../src/gallery.tsx'),
	route('/stale-closures', '../src/stale_closures.tsx'),
	route('/external-stores/redux', '../src/external_stores/redux.tsx'),
	route('/external-stores/redux-toolkit', '../src/external_stores/redux_toolkit.tsx'),
	route('/external-stores/zustand', '../src/external_stores/zustand.tsx'),
	route('/transition-use-state', '../src/transition_use_state.tsx'),
	route('/transition-use-context-selector', '../src/transition_use_context_selector.tsx'),
	route('/transition-use-search', '../src/transition_use_search.tsx'),
	route('/transition-redux', '../src/transition_redux.tsx'),
	route('/transition-zustand', '../src/transition_zustand.tsx'),
	route('/list-virt-fixed-height', '../src/list_virt/fixed_height.tsx'),
	route('/list-virt-dynamic-height', '../src/list_virt/dynamic_height.tsx'),
	route('/custom-elements', '../src/custom_elements.tsx'),
	route('/media-source-extensions', '../src/media_source_extensions.tsx'),
	route('/worker-offscreen-canvas', '../src/offscreen_canvas.tsx'),
	route('/webgl/triangle', '../src/webgl/triangle.tsx'),
	route('/webgl/rectangle', '../src/webgl/rectangle.tsx'),
	route('/webgl/multiple-rectangles', '../src/webgl/multiple_rectangles.tsx'),
	route('/ui/animations/sidebar', '../ui/animation_sidebar.tsx'),
	route('/ui/animations/sidebar-w', '../ui/animation_sidebar_w.tsx'),
	route('/ui/animations/disclosure', '../ui/animation_disclosure.tsx'),
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
	],
});
