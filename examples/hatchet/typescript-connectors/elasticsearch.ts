import { Client } from '@elastic/elasticsearch';

export const es = new Client({
  nodes: process.env.ELASTICSEARCH_NODES?.split(','),
  auth: {
    apiKey: process.env.ELASTICSEARCH_API_KEY ?? '',
  },
});
