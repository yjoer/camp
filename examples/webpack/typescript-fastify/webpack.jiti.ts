/* eslint-disable import-x/no-extraneous-dependencies */
// oxlint-disable import/no-default-export
import { get_server_config } from '@camp-org/webpack/config';

const mode = process.env.NODE_ENV === 'production' ? 'production' : 'development';

const config = get_server_config({
	entry: ['./server.ts'],
	mode,
	project_path: import.meta.dirname,
	config_path: import.meta.filename,
});

export default config;
