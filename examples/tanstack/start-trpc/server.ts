// oxlint-disable no-process-exit
// oxlint-disable no-console
import { fastifyCookie } from '@fastify/cookie';
import { fastifyCors } from '@fastify/cors';
import { initTRPC, TRPCError } from '@trpc/server';
import { fastifyTRPCPlugin } from '@trpc/server/adapters/fastify';
import { fastify } from 'fastify';

import type {
  CreateFastifyContextOptions,
  FastifyTRPCPluginOptions,
} from '@trpc/server/adapters/fastify';

const createContext = async (opts: CreateFastifyContextOptions) => {
  return {
    req: opts.req,
    res: opts.res,
  };
};

type Context = Awaited<ReturnType<typeof createContext>>;
const t = initTRPC.context<Context>().create();
const router = t.router;
const pub = t.procedure;

const auth = pub.use(async (opts) => {
  const { session_token } = opts.ctx.req.cookies;
  if (!session_token) throw new TRPCError({ code: 'UNAUTHORIZED' });

  return opts.next({ ctx: { session_token } });
});

const appRouter = router({
  public: pub.query(() => {
    return { message: 'hello' };
  }),
  protected: auth.query((opts) => {
    return { session_token: opts.ctx.session_token };
  }),
});

export type AppRouter = typeof appRouter;

const app = fastify();

await app.register(fastifyCors, {
  origin: [/localhost:\d+$/],
  credentials: true,
});

await app.register(fastifyCookie, {
  hook: 'onRequest',
});

app.register(fastifyTRPCPlugin, {
  prefix: 'trpc',
  trpcOptions: {
    router: appRouter,
    createContext,
    onError({ path, error }) {
      console.error(`Error in tRPC handler on path '${path}':`, error);
    },
  } satisfies FastifyTRPCPluginOptions<typeof appRouter>['trpcOptions'],
});

try {
  const address = await app.listen({ port: 3001 });
  console.log(`server listening at ${address}`);
} catch (error) {
  app.log.error(error);
  process.exit(1);
}
