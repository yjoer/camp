// oxlint-disable no-process-exit
// oxlint-disable no-console
import { createServer } from 'node:http';

import { os } from '@orpc/server';
import { RPCHandler } from '@orpc/server/node';
import { fastify } from 'fastify';

const router = {
	hello: os.route({ method: 'GET' }).handler(function () {
		return { hello: 'world' };
	}),
};

const handler = new RPCHandler(router);

const app = fastify({
	logger: false,
	serverFactory: (fastify_handler) => {
		const server = createServer((req, res) => {
			void handler.handle(req, res, { prefix: '/rpc' }).then(({ matched }) => {
				if (matched) return;
				fastify_handler(req, res);
			});
		});

		return server;
	},
});

try {
	const address = await app.listen({ port: 3000, host: '0.0.0.0' });
	console.log(`server listening at ${address}`);
} catch (error) {
	app.log.error(error);
	process.exit(1);
}
