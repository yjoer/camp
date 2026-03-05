import type { IncomingMessage, ServerResponse } from 'node:http';

export function posts(req: IncomingMessage, res: ServerResponse) {
  res.writeHead(200);
  res.write('posts');
  res.end();
}
