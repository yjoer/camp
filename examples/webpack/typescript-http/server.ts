// oxlint-disable no-console
import { createServer } from 'node:http';

import { router } from './router.ts';

const app = createServer(router);

app.listen(3000, '::', () => {
  console.log(`server is now listening on port 3000`);
});

const enabled = false;
if (enabled && import.meta.webpackHot) {
  import.meta.webpackHot.accept('./router.ts', async () => {
    console.log('♻️ HMR: Hot-Reloading');

    // const mod = await import('./router.ts');
    // app.removeListener('request', router);
    app.removeAllListeners('request');
    app.on('request', router);
    console.log('listeners', app.listenerCount('request'));
  });
}
