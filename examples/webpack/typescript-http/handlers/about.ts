import type { IncomingMessage, ServerResponse } from 'node:http';

export async function about(req: IncomingMessage, res: ServerResponse) {
  res.writeHead(200);
  res.write('about');
  res.end();
}
