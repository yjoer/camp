/* eslint-disable unicorn/no-process-exit */
/* eslint-disable unicorn/prefer-top-level-await */
import Fastify from 'fastify';

const fastify = Fastify({
  logger: true,
});

fastify.get('/', (request, reply) => {
  reply.send({ hello: 'world' });
});

(async () => {
  try {
    const address = await fastify.listen({ port: 3000 });
    fastify.log.info(`Server is now listening on ${address}`);
  } catch (error) {
    fastify.log.error(error);
    process.exit(1);
  }
})();

if (import.meta.webpackHot) {
  import.meta.webpackHot.accept();
  import.meta.webpackHot.dispose(() => fastify.close());

  import.meta.webpackHot.addStatusHandler((status) => {
    if (status === 'fail') process.exit();
  });
}
