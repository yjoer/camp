import { Client, ClusterConnectionPool } from '@elastic/elasticsearch';

export const es = new Client({
  nodes: process.env.ELASTICSEARCH_NODES?.split(','),
  ConnectionPool: ClusterConnectionPool,
  auth: {
    apiKey: process.env.ELASTICSEARCH_API_KEY ?? '',
  },
});
