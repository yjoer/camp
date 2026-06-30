import type { FastifyReply, FastifyRequest } from 'fastify';

import bcrypt from 'bcrypt';
import { Readable } from 'node:stream';
import { z } from 'zod/v4';

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

export function stream(request: FastifyRequest, reply: FastifyReply) {
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
		await import('missing-package');
	} catch (error) {
		reply.code(500).send({ error: (error as Error).message });
		return;
	}

	reply.send({});
}

const BcryptHashInput = z.object({
	query: z.object({
		password: z.string().nonempty(),
	}),
});

export async function bcryptHash(request: FastifyRequest, reply: FastifyReply) {
	const result = BcryptHashInput.safeParse(request);
	if (!result.success) return reply.code(400).send(result.error.issues);

	const { password } = result.data.query;
	const hash = await bcrypt.hash(password, 10);

	reply.send({ hash });
}

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
