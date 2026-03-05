import { bcryptHash, chunked, hello, missingPackages, stream } from '../mods/handlers.ts';

import type { FastifyPluginAsyncZod } from 'fastify-type-provider-zod';

// oxlint-disable-next-line typescript/require-await
export const router: FastifyPluginAsyncZod = async (app, _opts) => {
  app.get('/', (request, reply) => {
    hello(request, reply);
  });

  app.get('/chunked', async (request, reply) => {
    await chunked(request, reply);
  });

  app.get('/stream', (request, reply) => {
    stream(request, reply);
  });

  app.get('/missing-packages', async (request, reply) => {
    await missingPackages(request, reply);
  });

  app.get('/bcrypt', async (request, reply) => {
    await bcryptHash(request, reply);
  });
};

if (import.meta.webpackHot) {
  import.meta.webpackHot.accept('../mods/handlers.ts');
}
