/* eslint-disable unicorn/prefer-top-level-await */
// oxlint-disable no-process-exit
import { Readable } from 'node:stream';

import bcrypt from 'bcrypt';
import { fastify } from 'fastify';
import { serializerCompiler, validatorCompiler, ZodTypeProvider } from 'fastify-type-provider-zod';
import { z } from 'zod/v4';

const app = fastify({
  logger: true,
}).withTypeProvider<ZodTypeProvider>();

app.setValidatorCompiler(validatorCompiler);
app.setSerializerCompiler(serializerCompiler);

app.get('/', (request, reply) => {
  reply.send({ hello: 'world' });
});

app.get('/chunked', async (request, reply) => {
  reply.raw.write('<div>First</div>');
  await sleep(1000);

  reply.raw.write('<div>Second</div>');
  await sleep(1000);

  reply.raw.write('<div>Third</div>');
  reply.raw.write('<div>.</div>');
  reply.raw.end();
});

app.get('/stream', (request, reply) => {
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

app.get('/missing-packages', async (request, reply) => {
  try {
    // @ts-expect-error missing package
    // eslint-disable-next-line import-x/no-unresolved
    await import('missing-package');
  } catch (error) {
    reply.code(500).send({ error: error.message });
    return;
  }

  reply.send({});
});

app.get(
  '/bcrypt',
  {
    schema: {
      querystring: z.object({
        password: z.string(),
      }),
    },
  },
  async (request, reply) => {
    const { password } = request.query;

    const hash = await bcrypt.hash(password, 10);
    reply.send({ hash });
  },
);

(async () => {
  try {
    const address = await app.listen({ port: 3000 });
    app.log.info(`Server is now listening on ${address}`);
  } catch (error) {
    app.log.error(error);
    process.exit(1);
  }
})();

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

if (import.meta.webpackHot) {
  import.meta.webpackHot.accept();
  import.meta.webpackHot.dispose(() => app.close());

  import.meta.webpackHot.addStatusHandler((status) => {
    if (status === 'fail') process.exit();
  });
}
