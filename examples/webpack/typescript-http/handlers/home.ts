import type { IncomingMessage, ServerResponse } from 'node:http';

export async function home(req: IncomingMessage, res: ServerResponse) {
  res.writeHead(200);
  res.write('home');
  res.end();
}
