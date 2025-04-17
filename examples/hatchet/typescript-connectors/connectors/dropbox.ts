import { ConcurrencyLimitStrategy } from '@hatchet-dev/typescript-sdk';

import { es } from '../elasticsearch.ts';
import { hatchet } from '../hatchet-client.ts';

const baseUrl = 'https://api.dropboxapi.com';

export const dropbox = hatchet.workflow({
  name: 'dropbox',
  onCrons: ['*/5 * * * *'],
  concurrency: {
    expression: 'dropbox',
    maxRuns: 1,
    limitStrategy: ConcurrencyLimitStrategy.CANCEL_NEWEST,
  },
});

const getAccessToken = dropbox.task({
  name: 'get_access_token',
  fn: async () => {
    const refreshToken = process.env.DROPBOX_REFRESH_TOKEN;
    if (!refreshToken) throw new Error('Refresh token is missing');

    const clientId = process.env.DROPBOX_CLIENT_ID;
    if (!clientId) throw new Error('Client ID is missing');

    const clientSecret = process.env.DROPBOX_CLIENT_SECRET;
    if (!clientSecret) throw new Error('Client secret is missing');

    const response = await fetch(`${baseUrl}/oauth2/token`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        grant_type: 'refresh_token',
        refresh_token: refreshToken,
        client_id: clientId,
        client_secret: clientSecret,
      }),
    });

    if (!response.ok) throw new Error(`Error fetching access token: ${response.body}`);
    const data = await response.json();

    return data;
  },
});

const listFolder = dropbox.task({
  name: 'list_folder',
  parents: [getAccessToken],
  fn: async (input, ctx) => {
    const token = await ctx.parentOutput(getAccessToken);

    ctx.log('Retrieving a list of files and folders recursively.');
    let response = await fetch(`${baseUrl}/2/files/list_folder`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token.access_token}`,
      },
      body: JSON.stringify({
        path: '',
        recursive: true,
      }),
    });

    if (!response.ok) throw new Error(`Error fetching data: ${response.body}`);

    let data = await response.json();
    if (data.entries.length === 0) throw new Error('No entries found');

    const entriesAll: any[] = [];
    entriesAll.push(...data.entries);

    let { has_more: hasMore, cursor } = data;

    while (hasMore) {
      ctx.log('Retrieving a list of files and folders recursively (cont).');
      response = await fetch(`${baseUrl}/2/files/list_folder/continue`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token.access_token}`,
        },
        body: JSON.stringify({
          cursor,
        }),
      });

      if (!response.ok) throw new Error(`Error fetching data: ${response.body}`);

      data = await response.json();
      if (data.entries.length === 0) throw new Error('No entries found');

      entriesAll.push(...data.entries);
      ({ has_more: hasMore, cursor } = data);
    }

    ctx.log(`Total entries found: ${entriesAll.length}`);

    return {
      entries: entriesAll,
    };
  },
});

const enrichEntries = dropbox.task({
  name: 'enrich_entries_with_metadata',
  parents: [listFolder],
  fn: async (input, ctx) => {
    const token = await ctx.parentOutput(getAccessToken);
    const { entries } = await ctx.parentOutput(listFolder);

    const paperFiles = entries.filter((f) => f.name.endsWith('.paper'));
    if (paperFiles.length === 0) return entries;

    ctx.log(`Retrieving metadata for ${paperFiles.length} files.`);
    const response = await fetch(`${baseUrl}/2/sharing/get_file_metadata/batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token.access_token}`,
      },
      body: JSON.stringify({
        files: paperFiles.map((f) => f.path_lower),
      }),
    });

    if (!response.ok) throw new Error(`Error fetching metadata: ${response.body}`);

    const files = await response.json();
    if (files.length === 0) throw new Error('No files found');

    const filesMap = new Map();

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      filesMap.set(file.result.id, file.result);
    }

    for (let i = 0; i < entries.length; i++) {
      if (!filesMap.has(entries[i].id)) continue;
      entries[i].preview_url = filesMap.get(entries[i].id).preview_url;
    }

    return {
      entries,
    };
  },
});

dropbox.task({
  name: 'export_and_index_paper_files',
  parents: [enrichEntries],
  executionTimeout: '10m',
  fn: async (input, ctx) => {
    const token = await ctx.parentOutput(getAccessToken);
    const { entries } = await ctx.parentOutput(enrichEntries);

    const paperFiles = entries.filter((f) => f.name.endsWith('.paper'));
    if (paperFiles.length === 0) return;

    for (let i = 0; i < paperFiles.length; i++) {
      const file = paperFiles[i];

      ctx.log(`Exporting ${file.path_lower} to markdown.`);
      const response = await fetch(`https://content.dropboxapi.com/2/files/export`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token.access_token}`,
          'Dropbox-API-Arg': JSON.stringify({
            path: file.path_lower,
            export_format: 'markdown',
          }),
        },
      });

      if (!response.ok) throw new Error(`Failed to export ${file.path_lower}: ${response.body}`);

      const result = JSON.parse(response.headers.get('Dropbox-Api-Result') ?? '{}');
      const metadata = result.export_metadata;
      if (!metadata) throw new Error(`Failed to get metadata for ${file.path_lower}`);

      ctx.log(`Indexing ${file.path_lower}.`);
      await es.update({
        id: file.id,
        index: 'dropbox',
        doc: {
          id: file.id,
          name: file.name,
          path_lower: file.path_lower,
          path_display: file.path_display,
          revision: metadata.paper_revision,
          size: metadata.size,
          type: file['.tag'],
          updated_at: file.server_modified,
          url: file.preview_url,
          body: await response.text(),
        },
        doc_as_upsert: true,
      });
    }
  },
});
