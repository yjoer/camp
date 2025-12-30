// oxlint-disable no-console
// oxlint-disable no-process-exit
import { fastify } from 'fastify';

const app = fastify({
  logger: false,
});

app.get('/', function (request, reply) {
  reply.send({ hello: 'world' });
});

try {
  const address = await app.listen({ port: 3000 });
  console.log(`server listening at ${address}`);
} catch (error) {
  app.log.error(error);
  process.exit(1);
}
