import { os } from '@orpc/server';
import { RPCHandler } from '@orpc/server/fetch';
import { H3, serve } from 'h3';

const router = {
  hello: os.route({ method: 'GET' }).handler(function () {
    return { hello: 'world' };
  }),
};

const handler = new RPCHandler(router);
const app = new H3();

app.use('/rpc/**', async (event) => {
  const { matched, response } = await handler.handle(event.req, {
    prefix: '/rpc',
  });

  if (matched) {
    return response;
  }
});

serve(app, { port: 3000 });
