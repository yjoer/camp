/* eslint-disable import-x/no-extraneous-dependencies */
import http from 'k6/http';

export const options = {
  vus: 250,
  duration: '30s',
};

const urls = {
  node: 'http://127.0.0.1:3000',
  express: 'http://127.0.0.1:3000',
  fastify: 'http://127.0.0.1:3000',
  'h-three': 'http://127.0.0.1:3000',
  trpc: 'http://127.0.0.1:3000/trpc/hello',
  'trpc-fastify': 'http://127.0.0.1:3000/trpc/hello',
  orpc: 'http://127.0.0.1:3000/hello',
  'orpc-fastify': 'http://127.0.0.1:3000/rpc/hello',
  'orpc-h-three': 'http://127.0.0.1:3000/rpc/hello',
  'orpc-api': 'http://127.0.0.1:3000',
};

export default function bench() {
  http.get(urls[__ENV.SERVER]);
}
