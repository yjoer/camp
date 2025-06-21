/* eslint-disable unicorn/prefer-top-level-await */
// oxlint-disable no-process-exit
import { Readable } from 'node:stream';

import Fastify from 'fastify';

const fastify = Fastify({
  logger: true,
});

fastify.get('/', (request, reply) => {
  reply.send({ hello: 'world' });
});

fastify.get('/chunked', async (request, reply) => {
  reply.raw.write('<div>First</div>');
  await sleep(1000);

  reply.raw.write('<div>Second</div>');
  await sleep(1000);

  reply.raw.write('<div>Third</div>');
  reply.raw.write('<div>.</div>');
  reply.raw.end();
});

fastify.get('/stream', (request, reply) => {
  async function* generate() {
    yield '<div>First</div>';
    await sleep(1000);

    yield '<div>Second</div>';
    await sleep(1000);

    yield '<div>Third</div>';
    yield '<div>.</div>';
  }

  reply.send(Readable.from(generate()));
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

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

if (import.meta.webpackHot) {
  import.meta.webpackHot.accept();
  import.meta.webpackHot.dispose(() => fastify.close());

  import.meta.webpackHot.addStatusHandler((status) => {
    if (status === 'fail') process.exit();
  });
}
