import { ConcurrencyLimitStrategy } from '@hatchet-dev/typescript-sdk';

import { es } from '../elasticsearch.ts';
import { hatchet } from '../hatchet-client.ts';

const base_url = 'https://api.dropboxapi.com';

export const dropbox = hatchet.workflow({
	name: 'dropbox',
	onCrons: ['*/5 * * * *'],
	concurrency: {
		expression: "'dropbox'",
		maxRuns: 1,
		limitStrategy: ConcurrencyLimitStrategy.CANCEL_NEWEST,
	},
});

const get_access_token = dropbox.task({
	name: 'get_access_token',
	fn: async () => {
		const refresh_token = process.env.DROPBOX_REFRESH_TOKEN;
		if (!refresh_token) throw new Error('Refresh token is missing');

		const client_id = process.env.DROPBOX_CLIENT_ID;
		if (!client_id) throw new Error('Client ID is missing');

		const client_secret = process.env.DROPBOX_CLIENT_SECRET;
		if (!client_secret) throw new Error('Client secret is missing');

		const response = await fetch(`${base_url}/oauth2/token`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/x-www-form-urlencoded',
			},
			body: new URLSearchParams({
				grant_type: 'refresh_token',
				refresh_token,
				client_id,
				client_secret,
			}),
		});

		if (!response.ok) throw new Error(`Error fetching access token: ${JSON.stringify(response.body, undefined, 2)}`);
		const data = await response.json() as { access_token: string };

		return data;
	},
});

const list_folder = dropbox.task({
	name: 'list_folder',
	parents: [get_access_token],
	fn: async (input, ctx) => {
		const token = await ctx.parentOutput(get_access_token);

		void ctx.logger.info('Retrieving a list of files and folders recursively.');
		let response = await fetch(`${base_url}/2/files/list_folder`, {
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

		if (!response.ok) throw new Error(`Error fetching data: ${JSON.stringify(response.body, undefined, 2)}`);

		let data = await response.json() as { cursor: string; entries: FileMetadata[]; has_more: boolean };
		if (data.entries.length === 0) throw new Error('No entries found');

		const entries = [...data.entries];
		let { has_more, cursor } = data;

		while (has_more) {
			void ctx.logger.info('Retrieving a list of files and folders recursively (cont).');
			response = await fetch(`${base_url}/2/files/list_folder/continue`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					'Authorization': `Bearer ${token.access_token}`,
				},
				body: JSON.stringify({
					cursor,
				}),
			});

			if (!response.ok) throw new Error(`Error fetching data: ${JSON.stringify(response.body, undefined, 2)}`);

			data = await response.json() as { cursor: string; entries: FileMetadata[]; has_more: boolean };
			if (data.entries.length === 0) throw new Error('No entries found');

			entries.push(...data.entries);
			({ has_more, cursor } = data);
		}

		void ctx.logger.info(`Total entries found: ${entries.length}`);

		return {
			entries,
		};
	},
});

const enrich_entries = dropbox.task({
	name: 'enrich_entries_with_metadata',
	parents: [list_folder],
	fn: async (input, ctx) => {
		const token = await ctx.parentOutput(get_access_token);
		const { entries } = await ctx.parentOutput(list_folder);

		const paper_files = entries.filter(f => f.name.endsWith('.paper'));
		if (paper_files.length === 0) return entries;

		void ctx.logger.info(`Retrieving metadata for ${paper_files.length} files.`);
		const response = await fetch(`${base_url}/2/sharing/get_file_metadata/batch`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'Authorization': `Bearer ${token.access_token}`,
			},
			body: JSON.stringify({
				files: paper_files.map(f => f.path_lower),
			}),
		});

		if (!response.ok) throw new Error(`Error fetching metadata: ${JSON.stringify(response.body, undefined, 2)}`);

		const files = await response.json() as { result: { id: string; preview_url: string } }[];
		if (files.length === 0) throw new Error('No files found');

		const files_map = new Map<string, typeof files[0]['result']>();
		for (const file of files) files_map.set(file.result.id, file.result);

		for (let i = 0; i < entries.length; i++) {
			if (!files_map.has(entries[i].id)) continue;
			entries[i].preview_url = files_map.get(entries[i].id)!.preview_url;
		}

		return {
			entries,
		};
	},
});

dropbox.task({
	name: 'export_and_index_paper_files',
	parents: [enrich_entries],
	executionTimeout: '10m',
	fn: async (input, ctx) => {
		const token = await ctx.parentOutput(get_access_token);
		const { entries } = await ctx.parentOutput(enrich_entries);

		const paper_files = entries.filter(f => f.name.endsWith('.paper'));
		if (paper_files.length === 0) return;

		for (let i = 0; i < paper_files.length; i++) {
			const file = paper_files[i];

			void ctx.logger.info(`Exporting ${file.path_lower} to markdown.`);
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

			if (!response.ok) throw new Error(`Failed to export ${file.path_lower}: ${JSON.stringify(response.body, undefined, 2)}`);

			const result = response.headers.get('Dropbox-Api-Result');
			if (!result) throw new Error(`Failed to get export metadata for ${file.path_lower}`);

			const { export_metadata } = JSON.parse(result) as { export_metadata: ExportMetadata };

			void ctx.logger.info(`Indexing ${file.path_lower}.`);
			await es.update({
				id: file.id,
				index: 'dropbox',
				doc: {
					id: file.id,
					name: file.name,
					path_lower: file.path_lower,
					path_display: file.path_display,
					revision: export_metadata.paper_revision,
					size: export_metadata.size,
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

type FileMetadata = {
	'.tag': 'file' | 'folder';
	'id': string;
	'name': string;
	'path_lower': string;
	'path_display': string;
	'server_modified': string;
	'preview_url': string;
};

type ExportMetadata = {
	name: string;
	size: number;
	export_hash: string;
	paper_revision: number;
};
