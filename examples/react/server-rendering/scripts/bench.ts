/* eslint-disable import-x/no-extraneous-dependencies */
// oxlint-disable import/no-default-export
import http from 'k6/http';

export const options = {
  vus: 250,
  duration: '30s',
};

const urls = {
  'vite': 'http://127.0.0.1:3000',
  'vite-stream': 'http://127.0.0.1:3000',
  'next-pages': 'http://127.0.0.1:3000',
  'next-app': 'http://127.0.0.1:3000/app',
  'tss': 'http://127.0.0.1:3000',
};

export default function bench() {
  http.get(urls[__ENV.SERVER]);
}
