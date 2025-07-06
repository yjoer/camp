import { fastify } from 'fastify';
import { serializerCompiler, validatorCompiler, ZodTypeProvider } from 'fastify-type-provider-zod';

export let app = createApp();

function createApp() {
  const fst = fastify({
    logger: true,
  }).withTypeProvider<ZodTypeProvider>();

  fst.setValidatorCompiler(validatorCompiler);
  fst.setSerializerCompiler(serializerCompiler);

  return fst;
}

export function dispose() {
  app.close();
  app = createApp();
}

export type FastifyInstance = typeof app;

type FastifyHandlerOptions = Parameters<typeof app.route>[0]['handler'] extends (
  request: infer Request,
  reply: infer Reply,
) => any
  ? { Reply: Reply; Request: Request }
  : never;

export type FastifyRequest = FastifyHandlerOptions['Request'];
export type FastifyReply = FastifyHandlerOptions['Reply'];
