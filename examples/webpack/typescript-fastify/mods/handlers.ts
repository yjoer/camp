import { Readable } from 'node:stream';

import bcrypt from 'bcrypt';

import type { FastifyReply, FastifyRequest } from 'fastify';

export function hello(request: FastifyRequest, reply: FastifyReply) {
  reply.send({ hello: 'world' });
}

export async function chunked(request: FastifyRequest, reply: FastifyReply) {
  reply.raw.write('<div>First</div>');
  await sleep(1000);

  reply.raw.write('<div>Second</div>');
  await sleep(1000);

  reply.raw.write('<div>Third</div>');
  reply.raw.write('<div>.</div>');
  reply.raw.end();
}

export async function stream(request: FastifyRequest, reply: FastifyReply) {
  async function* generate() {
    yield '<div>First</div>';
    await sleep(1000);

    yield '<div>Second</div>';
    await sleep(1000);

    yield '<div>Third</div>';
    yield '<div>.</div>';
  }

  reply.send(Readable.from(generate()));
}

export async function missingPackages(request: FastifyRequest, reply: FastifyReply) {
  try {
    // @ts-expect-error missing package
    // eslint-disable-next-line import-x/no-unresolved
    await import('missing-package');
  } catch (error) {
    reply.code(500).send({ error: error.message });
    return;
  }

  reply.send({});
}

export async function bcryptHash(password: string) {
  const hash = await bcrypt.hash(password, 10);
  return hash;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
