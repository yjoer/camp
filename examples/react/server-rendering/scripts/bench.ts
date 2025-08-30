/* eslint-disable import-x/no-extraneous-dependencies */
import http from 'k6/http';

export const options = {
  vus: 250,
  duration: '30s',
};

const urls = {
  'vite': 'http://127.0.0.1:3000',
  'next-pages': 'http://127.0.0.1:3000',
  'next-app': 'http://127.0.0.1:3000/app',
  'tss': 'http://127.0.0.1:3000',
};

export default function bench() {
  http.get(urls[__ENV.SERVER]);
}
