// oxlint-disable no-console
import { initTRPC } from '@trpc/server';
import { createHTTPServer } from '@trpc/server/adapters/standalone';

const t = initTRPC.create();
const router = t.router;
const procedure = t.procedure;

const appRouter = router({
  hello: procedure.query(function () {
    return { hello: 'world' };
  }),
});

const server = createHTTPServer({
  router: appRouter,
  basePath: '/trpc/',
});

server.listen({ port: 3000 });
console.log('server listening on port 3000');
