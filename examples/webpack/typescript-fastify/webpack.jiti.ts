/* eslint-disable import-x/no-extraneous-dependencies */
// oxlint-disable import/no-default-export
import type { get_server_config as GSC } from '@camp-org/webpack/config';

import * as pkg from '@camp-org/webpack/config';

// @ts-expect-error jiti
// oxlint-disable-next-line namespace
const { get_server_config } = pkg.default as { get_server_config: typeof GSC };

const mode = process.env.NODE_ENV === 'production' ? 'production' : 'development';

const config = get_server_config({
	entry: ['./server.ts'],
	mode,
	project_path: import.meta.dirname,
	config_path: import.meta.filename,
});

export default config;
