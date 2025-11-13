// oxlint-disable no-process-exit
import { app, dispose } from './_app/fastify.ts';
import { router } from './_app/router.ts';

app.register(router);

try {
  const address = await app.listen({ port: 3000 });
  app.log.info(`Server is now listening on ${address}`);
} catch (error) {
  app.log.error(error);
  process.exit(1);
}

if (import.meta.webpackHot) {
  import.meta.webpackHot.accept();
  import.meta.webpackHot.dispose(() => dispose());

  import.meta.webpackHot.addStatusHandler((status) => {
    if (status === 'fail') process.exit();
  });
}
