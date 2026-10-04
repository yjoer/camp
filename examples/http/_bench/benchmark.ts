// oxlint-disable typescript/require-await
import type { ChildProcess, SpawnOptions } from 'node:child_process';

import { program } from 'commander';
import { deepStrictEqual } from 'node:assert';
import { spawn, spawnSync } from 'node:child_process';
import { appendFileSync, mkdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { pidtree } from 'pidtree';

const TARGET_MEASUREMENTS = 10;
const MAX_STD_RATIO = 0.05;
const IQR_MULTIPLIER = 1.5;

const script_dir = import.meta.dirname;
const http_dir = path.resolve(script_dir, '..');
const react_server_dir = path.resolve(http_dir, '..', 'react-server');

const base_url = 'http://127.0.0.1:3000';
const results_file = path.join(script_dir, '.build', 'results.jsonl');

async function main() {
	program
	.option('--server <prefix...>', 'filter servers by name prefixes')
	.option('--warmup <count>', 'warmup rounds per endpoint', w => Number.parseInt(w, 10), 1)
	.option('--verify-only', 'verify servers without benchmarking');

	program.parse();
	const opts = program.opts<{ server?: string[]; warmup: number; verifyOnly?: boolean }>();

	for (const server of servers) {
		if (opts.server?.some(prefix => server.name.startsWith(prefix)) === false) continue;
		process.stdout.write(`building ${server.group}/${server.name}\n`);
		await server.build();
	}

	for (const server of servers) {
		if (opts.server?.some(prefix => server.name.startsWith(prefix)) === false) continue;

		process.stdout.write(`starting ${server.group}/${server.name}\n`);
		const child = await server.setup();

		try {
			for (const endpoint of server.endpoints) {
				process.stdout.write(`verifying ${endpoint.path} of ${server.group}/${server.name}\n`);
				await _verify_endpoint(endpoint);
			}
		} finally {
			process.stdout.write(`stopping ${server.group}/${server.name}\n`);
			await server.teardown(child);
		}
	}

	if (opts.verifyOnly) return;

	for (const server of servers) {
		if (opts.server?.some(prefix => server.name.startsWith(prefix)) === false) continue;

		process.stdout.write(`starting ${server.group}/${server.name}\n`);
		const child = await server.setup();

		try {
			for (const endpoint of server.endpoints) {
				for (let i = 0; i < opts.warmup; i += 1) {
					process.stdout.write(`warming up ${endpoint.path} of ${server.group}/${server.name}\n`);
					await server.measure(endpoint);
				}

				let measurements: number[] = [];
				for (;;) {
					process.stdout.write(`measuring ${endpoint.path} of ${server.group}/${server.name}`);
					measurements.push(await server.measure(endpoint));

					const { n, outliers, max, mean, std, ratio } = _stats(measurements);
					process.stdout.write(`n: ${n}, outliers: ${outliers.join(',')}, mean: ${mean}, std: ${std}, std_ratio: ${ratio}\n`);

					if (measurements.length >= TARGET_MEASUREMENTS && outliers.includes(max)) {
						process.stdout.write(`excluding peak ${max} from the distribution underestimates the mean, recollecting ${endpoint.path} of ${server.group}/${server.name}\n`);
						measurements = [max];
						continue;
					}

					if (measurements.length >= TARGET_MEASUREMENTS && ratio > MAX_STD_RATIO) {
						process.stdout.write(`std ratio ${ratio} exceeds ${MAX_STD_RATIO}, recollecting ${endpoint.path} of ${server.group}/${server.name}\n`);
						measurements = [];
						continue;
					}

					if (n >= TARGET_MEASUREMENTS) break;
				}

				const { mean, std } = _stats(measurements);
				mkdirSync(path.join(script_dir, '.build'), { recursive: true });
				appendFileSync(results_file, `${JSON.stringify({
					group: server.group,
					server_name: server.name,
					endpoint_name: endpoint.path,
					language: server.language,
					timestamp: new Date().toISOString(),
					measurements,
					mean_rps: mean,
					std_rps: std,
					mean_latency_us: 1_000_000 / mean,
				})}\n`);
			}
		} finally {
			process.stdout.write(`stopping ${server.group}/${server.name}\n`);
			await server.teardown(child);
		}
	}
}

async function _verify_endpoint(endpoint: Endpoint) {
	const deadline = Date.now() + 30_000;

	for (;;) {
		let response: Response | undefined;

		try {
			response = await fetch(
				`${base_url}${endpoint.path}`,
				endpoint.method === 'GET' ? {
					method: endpoint.method,
				} : {
					method: endpoint.method,
					headers: { 'content-type': 'application/json' },
					body: JSON.stringify(endpoint.body ?? '{}'),
				},
			);
		} catch {
			response = undefined;
		}

		if (response === undefined) {
			if (Date.now() >= deadline) throw new Error(`${endpoint.path} did not become ready in time`);
			await new Promise(resolve => setTimeout(resolve, 500));
			continue;
		}

		if (!response.ok) throw new Error(`${endpoint.path} returned ${response.status}`);

		if (endpoint.expected_json !== undefined) {
			deepStrictEqual(await response.json(), endpoint.expected_json);
		}

		if (endpoint.expected_regex?.test(await response.text()) === false) {
			throw new Error(`${endpoint.path} response did not match ${endpoint.expected_regex}`);
		}

		return;
	}
}

function _stats(measurements: number[]) {
	const sorted = measurements.toSorted((a, b) => a - b);
	const lower = _percentile(sorted, 25);
	const upper = _percentile(sorted, 75);

	const range = upper - lower;
	const lower_fence = lower - IQR_MULTIPLIER * range;
	const upper_fence = upper + IQR_MULTIPLIER * range;

	let n = 0, mean = 0, m2 = 0;
	let max = -Infinity;
	const outliers: number[] = [];
	for (const x of measurements) {
		if (x > max) max = x;
		if (x < lower_fence || x > upper_fence) {
			outliers.push(x);
			continue;
		}

		// welford's algorithm
		n += 1;
		const delta = x - mean;
		mean += delta / n;
		const delta2 = x - mean;
		m2 += delta * delta2;
	}

	const std = Math.sqrt(m2 / n);
	return { n, outliers, max, mean, std, ratio: std / mean };
}

const _percentile = (array: number[], p: number) => {
	const pos = (array.length - 1) * p / 100;
	const lo = Math.floor(pos);
	const hi = Math.ceil(pos);
	return array[lo] + (pos - lo) * (array[hi] - array[lo]);
};

interface Server {
	group: string;
	name: string;
	language: string;
	endpoints: Endpoint[];
	build(): Promise<void>;
	setup(): Promise<ChildProcess>;
	measure(endpoint: Endpoint): Promise<number>;
	teardown(child: ChildProcess): Promise<void>;
}

interface Endpoint {
	path: string;
	method: 'GET' | 'POST' | 'PUT' | 'DELETE';
	body?: unknown;
	expected_json?: unknown;
	expected_regex?: RegExp;
}

const node: Server = {
	group: 'http',
	name: 'node',
	language: 'typescript',
	endpoints: [
		{ path: '/', method: 'GET', expected_json: { hello: 'world' } },
	],
	async build() {},
	async setup() {
		return _spawn('node', ['server.ts'], { cwd: path.join(http_dir, 'node'), stdio: 'inherit' });
	},
	async measure(endpoint: Endpoint) {
		return _measure_k6(endpoint).metrics.http_reqs.values.rate;
	},
	async teardown(child: ChildProcess) {
		await _kill_tree_wait(child);
	},
};

const express: Server = {
	group: 'http',
	name: 'express',
	language: 'typescript',
	endpoints: [
		{ path: '/', method: 'GET', expected_json: { hello: 'world' } },
	],
	async build() {},
	async setup() {
		return _spawn('node', ['server.ts'], { cwd: path.join(http_dir, 'express'), stdio: 'inherit' });
	},
	async measure(endpoint: Endpoint) {
		return _measure_k6(endpoint).metrics.http_reqs.values.rate;
	},
	async teardown(child: ChildProcess) {
		await _kill_tree_wait(child);
	},
};

const fastify: Server = {
	group: 'http',
	name: 'fastify',
	language: 'typescript',
	endpoints: [
		{ path: '/', method: 'GET', expected_json: { hello: 'world' } },
	],
	async build() {},
	async setup() {
		return _spawn('node', ['server.ts'], { cwd: path.join(http_dir, 'fastify'), stdio: 'inherit' });
	},
	async measure(endpoint: Endpoint) {
		return _measure_k6(endpoint).metrics.http_reqs.values.rate;
	},
	async teardown(child: ChildProcess) {
		await _kill_tree_wait(child);
	},
};

const h_three: Server = {
	group: 'http',
	name: 'h-three',
	language: 'typescript',
	endpoints: [
		{ path: '/', method: 'GET', expected_json: { hello: 'world' } },
	],
	async build() {},
	async setup() {
		return _spawn('node', ['server.ts'], { cwd: path.join(http_dir, 'h-three'), stdio: 'inherit' });
	},
	async measure(endpoint: Endpoint) {
		return _measure_k6(endpoint).metrics.http_reqs.values.rate;
	},
	async teardown(child: ChildProcess) {
		await _kill_tree_wait(child);
	},
};

const trpc: Server = {
	group: 'http',
	name: 'trpc',
	language: 'typescript',
	endpoints: [
		{ path: '/trpc/hello', method: 'GET', expected_json: { result: { data: { hello: 'world' } } } },
	],
	async build() {},
	async setup() {
		return _spawn('node', ['node.ts'], { cwd: path.join(http_dir, 'trpc'), stdio: 'inherit' });
	},
	async measure(endpoint: Endpoint) {
		return _measure_k6(endpoint).metrics.http_reqs.values.rate;
	},
	async teardown(child: ChildProcess) {
		await _kill_tree_wait(child);
	},
};

const trpc_fastify: Server = {
	group: 'http',
	name: 'trpc-fastify',
	language: 'typescript',
	endpoints: [
		{ path: '/trpc/hello', method: 'GET', expected_json: { result: { data: { hello: 'world' } } } },
	],
	async build() {},
	async setup() {
		return _spawn('node', ['fastify.ts'], { cwd: path.join(http_dir, 'trpc'), stdio: 'inherit' });
	},
	async measure(endpoint: Endpoint) {
		return _measure_k6(endpoint).metrics.http_reqs.values.rate;
	},
	async teardown(child: ChildProcess) {
		await _kill_tree_wait(child);
	},
};

const orpc: Server = {
	group: 'http',
	name: 'orpc',
	language: 'typescript',
	endpoints: [
		{ path: '/hello', method: 'GET', expected_json: { json: { hello: 'world' } } },
	],
	async build() {},
	async setup() {
		return _spawn('node', ['node.ts'], { cwd: path.join(http_dir, 'orpc'), stdio: 'inherit' });
	},
	async measure(endpoint: Endpoint) {
		return _measure_k6(endpoint).metrics.http_reqs.values.rate;
	},
	async teardown(child: ChildProcess) {
		await _kill_tree_wait(child);
	},
};

const orpc_fastify: Server = {
	group: 'http',
	name: 'orpc-fastify',
	language: 'typescript',
	endpoints: [
		{ path: '/rpc/hello', method: 'GET', expected_json: { json: { hello: 'world' } } },
	],
	async build() {},
	async setup() {
		return _spawn('node', ['fastify.ts'], { cwd: path.join(http_dir, 'orpc'), stdio: 'inherit' });
	},
	async measure(endpoint: Endpoint) {
		return _measure_k6(endpoint).metrics.http_reqs.values.rate;
	},
	async teardown(child: ChildProcess) {
		await _kill_tree_wait(child);
	},
};

const orpc_h_three: Server = {
	group: 'http',
	name: 'orpc-h-three',
	language: 'typescript',
	endpoints: [
		{ path: '/rpc/hello', method: 'GET', expected_json: { json: { hello: 'world' } } },
	],
	async build() {},
	async setup() {
		return _spawn('node', ['h_three.ts'], { cwd: path.join(http_dir, 'orpc'), stdio: 'inherit' });
	},
	async measure(endpoint: Endpoint) {
		return _measure_k6(endpoint).metrics.http_reqs.values.rate;
	},
	async teardown(child: ChildProcess) {
		await _kill_tree_wait(child);
	},
};

const orpc_openapi_node: Server = {
	group: 'http',
	name: 'orpc-openapi-node',
	language: 'typescript',
	endpoints: [
		{ path: '/', method: 'GET', expected_json: { hello: 'world' } },
	],
	async build() {},
	async setup() {
		return _spawn('node', ['openapi_node.ts'], { cwd: path.join(http_dir, 'orpc'), stdio: 'inherit' });
	},
	async measure(endpoint: Endpoint) {
		return _measure_k6(endpoint).metrics.http_reqs.values.rate;
	},
	async teardown(child: ChildProcess) {
		await _kill_tree_wait(child);
	},
};

const fastapi: Server = {
	group: 'http',
	name: 'fastapi',
	language: 'python',
	endpoints: [
		{ path: '/', method: 'GET', expected_json: { hello: 'world' } },
	],
	async build() {},
	async setup() {
		return _spawn('uv', ['run', 'server.py'], { cwd: path.join(http_dir, 'fastapi'), stdio: 'inherit' });
	},
	async measure(endpoint: Endpoint) {
		return _measure_k6(endpoint).metrics.http_reqs.values.rate;
	},
	async teardown(child: ChildProcess) {
		await _kill_tree_wait(child);
	},
};

const spring_boot: Server = {
	group: 'http',
	name: 'spring-boot',
	language: 'java',
	endpoints: [
		{ path: '/', method: 'GET', expected_json: { hello: 'world' } },
	],
	async build() {
		spawnSync('gradle', ['bootJar'], { cwd: path.join(http_dir, 'spring-boot'), stdio: 'inherit' });
	},
	async setup() {
		return _spawn('java', ['-XX:ActiveProcessorCount=1', '-jar', '.build/libs/spring-boot.jar'], { cwd: path.join(http_dir, 'spring-boot'), stdio: 'inherit' });
	},
	async measure(endpoint: Endpoint) {
		return _measure_k6(endpoint).metrics.http_reqs.values.rate;
	},
	async teardown(child: ChildProcess) {
		await _kill_tree_wait(child);
	},
};

const axum: Server = {
	group: 'http',
	name: 'axum',
	language: 'rust',
	endpoints: [
		{ path: '/', method: 'GET', expected_json: { hello: 'world' } },
	],
	async build() {
		spawnSync('cargo', ['build', '--release'], { cwd: path.join(http_dir, 'axum'), stdio: 'inherit' });
	},
	async setup() {
		return _spawn('cargo', ['run', '--release'], { cwd: path.join(http_dir, 'axum'), stdio: 'inherit' });
	},
	async measure(endpoint: Endpoint) {
		return _measure_k6(endpoint).metrics.http_reqs.values.rate;
	},
	async teardown(child: ChildProcess) {
		await _kill_tree_wait(child);
	},
};

const axum_connect: Server = {
	group: 'http',
	name: 'axum-connect',
	language: 'rust',
	endpoints: [
		{ path: '/hello.v1.HelloService/Hello', method: 'POST', body: {}, expected_json: { hello: 'world' } },
	],
	async build() {
		spawnSync('cargo', ['build', '--release'], { cwd: path.join(http_dir, 'axum-connect'), stdio: 'inherit' });
	},
	async setup() {
		return _spawn('cargo', ['run', '--release'], { cwd: path.join(http_dir, 'axum-connect'), stdio: 'inherit' });
	},
	async measure(endpoint: Endpoint) {
		return _measure_k6(endpoint).metrics.http_reqs.values.rate;
	},
	async teardown(child: ChildProcess) {
		await _kill_tree_wait(child);
	},
};

const react_server_vite: Server = {
	group: 'react-server',
	name: 'vite',
	language: 'typescript',
	endpoints: [
		{ path: '/', method: 'GET', expected_regex: /Hello, World!/ },
	],
	async build() {
		spawnSync('yarn', ['build'], { cwd: path.join(react_server_dir, 'vite'), stdio: 'inherit' });
	},
	async setup() {
		return _spawn('yarn', ['start'], { cwd: path.join(react_server_dir, 'vite'), stdio: 'inherit' });
	},
	async measure(endpoint: Endpoint) {
		return _measure_k6(endpoint).metrics.http_reqs.values.rate;
	},
	async teardown(child: ChildProcess) {
		await _kill_tree_wait(child);
	},
};

const react_server_vite_stream: Server = {
	group: 'react-server',
	name: 'vite-stream',
	language: 'typescript',
	endpoints: [
		{ path: '/', method: 'GET', expected_regex: /Hello, World!/ },
	],
	async build() {
		spawnSync('yarn', ['build'], { cwd: path.join(react_server_dir, 'vite'), stdio: 'inherit' });
	},
	async setup() {
		return _spawn('yarn', ['start:stream'], { cwd: path.join(react_server_dir, 'vite'), stdio: 'inherit' });
	},
	async measure(endpoint: Endpoint) {
		return _measure_k6(endpoint).metrics.http_reqs.values.rate;
	},
	async teardown(child: ChildProcess) {
		await _kill_tree_wait(child);
	},
};

const react_server_next_pages: Server = {
	group: 'react-server',
	name: 'next-pages',
	language: 'typescript',
	endpoints: [
		{ path: '/', method: 'GET', expected_regex: /Hello, World!/ },
	],
	async build() {
		spawnSync('yarn', ['build'], { cwd: path.join(react_server_dir, 'next'), stdio: 'inherit' });
	},
	async setup() {
		return _spawn('yarn', ['start'], { cwd: path.join(react_server_dir, 'next'), stdio: 'inherit' });
	},
	async measure(endpoint: Endpoint) {
		return _measure_k6(endpoint).metrics.http_reqs.values.rate;
	},
	async teardown(child: ChildProcess) {
		await _kill_tree_wait(child);
	},
};

const react_server_next_app: Server = {
	group: 'react-server',
	name: 'next-app',
	language: 'typescript',
	endpoints: [
		{ path: '/app', method: 'GET', expected_regex: /Hello, World!/ },
	],
	async build() {
		spawnSync('yarn', ['build'], { cwd: path.join(react_server_dir, 'next'), stdio: 'inherit' });
	},
	async setup() {
		return _spawn('yarn', ['start'], { cwd: path.join(react_server_dir, 'next'), stdio: 'inherit' });
	},
	async measure(endpoint: Endpoint) {
		return _measure_k6(endpoint).metrics.http_reqs.values.rate;
	},
	async teardown(child: ChildProcess) {
		await _kill_tree_wait(child);
	},
};

const react_server_tanstack_start: Server = {
	group: 'react-server',
	name: 'tanstack-start',
	language: 'typescript',
	endpoints: [
		{ path: '/', method: 'GET', expected_regex: /Hello, World!/ },
	],
	async build() {
		spawnSync('yarn', ['build'], { cwd: path.join(react_server_dir, 'tanstack-start'), stdio: 'inherit' });
	},
	async setup() {
		return _spawn('yarn', ['start'], { cwd: path.join(react_server_dir, 'tanstack-start'), stdio: 'inherit' });
	},
	async measure(endpoint: Endpoint) {
		return _measure_k6(endpoint).metrics.http_reqs.values.rate;
	},
	async teardown(child: ChildProcess) {
		await _kill_tree_wait(child);
	},
};

export const servers: Server[] = [
	node,
	express,
	fastify,
	h_three,
	trpc,
	trpc_fastify,
	orpc,
	orpc_fastify,
	orpc_h_three,
	orpc_openapi_node,
	fastapi,
	spring_boot,
	axum,
	axum_connect,
	react_server_vite,
	react_server_vite_stream,
	react_server_next_pages,
	react_server_next_app,
	react_server_tanstack_start,
];

function _spawn(cmd: string, args: string[], options: SpawnOptions) {
	return spawn(cmd, args, options).on('exit', (code) => {
		if (code !== null && code !== 0 && code !== 1) throw new Error(`${cmd} exited with code ${code}`);
	});
}

async function _kill_tree_wait(child: ChildProcess) {
	if (child.pid === undefined) return;
	const pids = await pidtree(child.pid, { root: true });
	for (const pid of pids) process.kill(pid);

	const deadline = Date.now() + 10_000;
	for (;;) {
		const alive = pids.filter((pid) => {
			try {
				process.kill(pid, 0);
				return true;
			} catch {
				return false;
			}
		});

		if (alive.length === 0) return;
		if (Date.now() >= deadline) throw new Error(`timed out waiting for ${alive.join(',')} to exit`);
		await new Promise(resolve => setTimeout(resolve, 200));
	}
}

function _measure_k6(endpoint: Endpoint) {
	spawnSync('k6', [
		'run',
		'--env', `URL=${base_url}${endpoint.path}`,
		'--env', `METHOD=${endpoint.method}`,
		'--env', 'VUS=250',
		'--env', 'DURATION=10s',
		...(endpoint.body === undefined ? [] : ['--env', `BODY=${JSON.stringify(endpoint.body)}`]),
		path.join(script_dir, 'k_six.ts'),
	], { stdio: 'inherit' });

	return JSON.parse(readFileSync(path.join(script_dir, '.build', 'summary.json'), 'utf8')) as {
		metrics: {
			http_reqs: {
				values: { rate: number };
			};
		};
	};
}

await main();
