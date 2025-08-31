import ReactDOMServer from 'react-dom/server';

import App from './App';

import type { RenderToPipeableStreamOptions } from 'react-dom/server';

export function render() {
  return ReactDOMServer.renderToString(<App />);
}

export function renderStream(options: RenderToPipeableStreamOptions) {
  return ReactDOMServer.renderToPipeableStream(<App />, options);
}
