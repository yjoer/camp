#!/usr/bin/env node
// oxlint-disable no-console
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

/* eslint-disable-next-line import-x/no-extraneous-dependencies */
import { Generator, getConfig } from '@tanstack/router-generator';

import type { VirtualRootRoute } from '@tanstack/virtual-file-routes';

const cwd = process.cwd();
const vite_config_file = path.resolve(cwd, 'vite.config.ts');
const manifest_file = path.resolve(cwd, 'package.json');
try {
	await fs.access(vite_config_file);
	await fs.access(manifest_file);
} catch {
	console.error('vite.config.ts or package.json not found in the current directory');
	process.exit(1);
}

const mod = await import(pathToFileURL(vite_config_file).href) as { routes?: VirtualRootRoute };
if (!mod.routes) {
	console.error('no routes export found in vite.config.ts');
	process.exit(1);
}

const manifest = JSON.parse(await fs.readFile(manifest_file, 'utf8')) as { dependencies?: Record<string, string> };
let framework: 'react' | 'solid' | undefined;
if (manifest.dependencies?.['@tanstack/react-start']) {
	framework = 'react';
} else if (manifest.dependencies?.['@tanstack/solid-start']) {
	framework = 'solid';
} else {
	console.error('no @tanstack/*-start dependency found in package.json');
	process.exit(1);
}

const config = getConfig({
	target: framework,
	virtualRouteConfig: mod.routes,
	routesDirectory: '.',
	generatedRouteTree: './routeTree.gen.ts',
	routeTreeFileFooter: [[
		"import type { getRouter } from './router.tsx'",
		`import type { createStart } from '@tanstack/${framework}-start'`,
		`declare module '@tanstack/${framework}-start' {`,
		'  interface Register {',
		'    ssr: true',
		'    router: Awaited<ReturnType<typeof getRouter>>',
		'  }',
		'}',
	].join('\n')],
}, '_app');

const generator = new Generator({ config, root: '_app' });
await generator.run();
