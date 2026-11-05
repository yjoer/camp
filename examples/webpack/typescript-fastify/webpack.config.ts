import { get_server_config } from '@camp-org/webpack/config';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const mode = process.env.NODE_ENV === 'production' ? 'production' : 'development';

const config = get_server_config({
	entry: ['./server.ts'],
	mode,
	project_path: path.dirname(fileURLToPath(import.meta.url)),
	config_path: fileURLToPath(import.meta.url),
});

export default config;
