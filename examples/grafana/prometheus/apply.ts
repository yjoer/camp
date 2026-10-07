import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { manifest } from './pgbouncer.ts';

const json = JSON.stringify(manifest().build(), undefined, 2);
const tmp = mkdtempSync(path.join(tmpdir(), 'pgbouncer-exporter'));
const file = path.join(tmp, 'dashboard.json');

try {
	writeFileSync(file, json, 'utf8');
	execFileSync('gcx', ['resources', 'push', 'dashboards', '--path', file], { stdio: 'inherit' });
} finally {
	rmSync(tmp, { recursive: true, force: true });
}
