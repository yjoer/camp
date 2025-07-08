// oxlint-disable no-console
import { createServer } from 'node:http';

import { router } from './router.ts';

const app = createServer(router);

app.listen(3000, '::', () => {
  console.log(`server is now listening on port 3000`);
});

const enabled = true;
if (enabled && import.meta.webpackHot) {
  let oldRouter = router;

  import.meta.webpackHot.accept('./router.ts', async () => {
    console.log('♻️ HMR: Hot-Reloading');

    // const mod = await import('./router.ts');
    // app.removeAllListeners('request');
    app.removeListener('request', oldRouter);
    app.on('request', router);
    oldRouter = router;

    console.log('listeners', app.listenerCount('request'));
  });
}
