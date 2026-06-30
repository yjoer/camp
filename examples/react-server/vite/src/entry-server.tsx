import type { RenderToPipeableStreamOptions } from 'react-dom/server';

import ReactDOMServer from 'react-dom/server';

import { App } from './app';

export function render() {
	return ReactDOMServer.renderToString(<App />);
}

export function render_stream(options: RenderToPipeableStreamOptions) {
	return ReactDOMServer.renderToPipeableStream(<App />, options);
}
