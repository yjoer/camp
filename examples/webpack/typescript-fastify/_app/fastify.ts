import { fastify } from 'fastify';
import { serializerCompiler, validatorCompiler, ZodTypeProvider } from 'fastify-type-provider-zod';

export function createApp() {
	const fst = fastify({
		logger: true,
	}).withTypeProvider<ZodTypeProvider>();

	fst.setValidatorCompiler(validatorCompiler);
	fst.setSerializerCompiler(serializerCompiler);

	return fst;
}

export type FastifyInstance = ReturnType<typeof createApp>;

type FastifyHandlerOptions = Parameters<FastifyInstance['route']>[0]['handler'] extends (
	request: infer Request,
	reply: infer Reply,
) => any
	? { Reply: Reply; Request: Request }
	: never;

export type FastifyRequest = FastifyHandlerOptions['Request'];
export type FastifyReply = FastifyHandlerOptions['Reply'];
