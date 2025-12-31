/* eslint-disable import-x/no-extraneous-dependencies */
/* eslint-disable perfectionist/sort-switch-case */
import http from 'k6/http';

const url = __ENV.URL;
const method = __ENV.METHOD || 'GET';
const body = __ENV.BODY || undefined;
const vus = Number.parseInt(__ENV.VUS || '250');
const duration = __ENV.DURATION || '30s';

export const options = {
  vus,
  duration,
};

export default function bench() {
  switch (method) {
    case 'GET': {
      http.get(url);
      break;
    }
    case 'POST': {
      http.post(url, body, { headers: { 'Content-Type': 'application/json' } });
      break;
    }
    case 'PUT': {
      http.put(url, body, { headers: { 'Content-Type': 'application/json' } });
      break;
    }
    case 'DELETE': {
      http.del(url);
      break;
    }
  }
}

export function handleSummary(data: Record<string, unknown>) {
  return {
    '.build/summary.json': JSON.stringify(data),
  };
}
