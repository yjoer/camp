// oxlint-disable no-console
import { createServer } from 'node:http';

const server = createServer((req, res) => {
  if (req.method === 'GET' && req.url === '/') {
    res.statusCode = 200;
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify({ hello: 'world' }));
  }
});

server.listen({ port: 3000 }, function () {
  console.log('server listening on port 3000');
});
