import { H3, serve } from 'h3';

const app = new H3();

app.get('/', function () {
  return { hello: 'world' };
});

serve(app, { port: 3000 });
