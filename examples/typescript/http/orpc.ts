// oxlint-disable no-console
import { createServer } from 'node:http';

import { os } from '@orpc/server';
import { RPCHandler } from '@orpc/server/node';

const router = {
  hello: os.route({ method: 'GET' }).handler(function () {
    return { hello: 'world' };
  }),
};

const handler = new RPCHandler(router);

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
