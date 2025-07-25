import { bcryptHash, chunked, hello, missingPackages, stream } from '../mods/handlers.ts';

import type { FastifyPluginAsyncZod } from 'fastify-type-provider-zod';

export const router: FastifyPluginAsyncZod = async (app, _opts) => {
  app.get('/', (request, reply) => {
    hello(request, reply);
  });

  app.get('/chunked', (request, reply) => {
    chunked(request, reply);
  });

  app.get('/stream', (request, reply) => {
    stream(request, reply);
  });

  app.get('/missing-packages', (request, reply) => {
    missingPackages(request, reply);
  });

  app.get('/bcrypt', (request, reply) => {
    bcryptHash(request, reply);
  });
};

if (import.meta.webpackHot) {
  import.meta.webpackHot.accept('../mods/handlers.ts');
}
