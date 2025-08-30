// oxlint-disable no-console
import { createServer } from 'node:http';

import { OpenAPIHandler } from '@orpc/openapi/node';
import { os } from '@orpc/server';

const router = {
  hello: os.route({ method: 'GET', path: '/' }).handler(function () {
    return { hello: 'world' };
  }),
};

const handler = new OpenAPIHandler(router);

const server = createServer(async (req, res) => {
  const result = await handler.handle(req, res, {
    context: { headers: req.headers },
  });

  if (!result.matched) {
    res.statusCode = 404;
    res.end('no procedure matched');
  }
});

server.listen({ port: 3000 });
console.log('server listening on port 3000');
