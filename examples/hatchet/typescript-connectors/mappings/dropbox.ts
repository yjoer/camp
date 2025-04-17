import { es } from '../elasticsearch.ts';

async function v1() {
  const exists = await es.indices.exists({ index: 'dropbox' });
  if (exists) return;

  await es.indices.create({
    index: 'dropbox',
    mappings: {
      dynamic: false,
      properties: {
        id: { type: 'keyword' },
        name: { type: 'text' },
        path_lower: { type: 'keyword' },
        path_display: { type: 'keyword' },
        revision: { type: 'long' },
        size: { type: 'long' },
        type: { type: 'keyword' },
        updated_at: { type: 'date' },
        url: { type: 'keyword' },
        body: { type: 'text' },
      },
    },
  });
}

await v1();
