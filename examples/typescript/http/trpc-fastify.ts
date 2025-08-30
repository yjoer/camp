// oxlint-disable no-console
// oxlint-disable no-process-exit
import { initTRPC } from '@trpc/server';
import { fastifyTRPCPlugin } from '@trpc/server/adapters/fastify';
import { fastify } from 'fastify';

import type { FastifyTRPCPluginOptions } from '@trpc/server/adapters/fastify';

const t = initTRPC.create();
const router = t.router;
const procedure = t.procedure;

const appRouter = router({
  hello: procedure.query(function () {
    return { hello: 'world' };
  }),
});

const app = fastify({
  logger: false,
});

app.register(fastifyTRPCPlugin, {
  prefix: '/trpc',
  trpcOptions: {
    router: appRouter,
    onError({ path, error }) {
      console.error(`Error in tRPC handler on path '${path}':`, error);
    },
  } satisfies FastifyTRPCPluginOptions<typeof appRouter>['trpcOptions'],
});

try {
  const address = await app.listen({ port: 3000 });
  console.log(`server listening at ${address}`);
} catch (error) {
  app.log.error(error);
  process.exit(1);
}
