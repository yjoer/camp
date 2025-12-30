// oxlint-disable no-console
import express from 'express';

const app = express();

app.get('/', (req, res) => {
  res.send({ hello: 'world' });
});

app.listen(3000, function () {
  console.log('server listening on port 3000');
});
