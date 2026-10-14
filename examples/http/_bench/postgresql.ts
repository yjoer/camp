// oxlint-disable typescript/require-await
import type { OptionValues } from 'commander';

import { program } from 'commander';
import { spawnSync } from 'node:child_process';
import { appendFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { z } from 'zod';

import { stats } from './lib.ts';
import { setup_pgbench } from './setup_pgbench.ts';

const TOTAL_TRANSACTIONS = 1000;
const TARGET_MEASUREMENTS = 10;
const MAX_STD_RATIO = 0.05;

const script_dir = import.meta.dirname;
const node_pgbench_dir = path.resolve(script_dir, '..', '..', 'database', 'node_postgres');
const results_file = path.join(script_dir, '.build', 'postgresql_results.jsonl');

async function main() {
	program
	.option('-r, --runner <name>', 'node, pgbench')
	.option('-p, --port <port>', 'port to run (5432 direct, 6432 pgbouncer)')
	.option('-c, --client <name>', 'small (5-100 Δ5), large (100-1000 Δ100)')
	.option('-t, --transactions <count>', 'transactions per client (default 1000/clients)')
	.option('-s, --scale <factor>', 'scaling factor')

	.action(async () => {
		const opts = RunOptions.parse(program.opts());

		const clients = { small: [5, 100, 5], large: [100, 1000, 100] } as const;
		const runners = [node, pgbench];
		const experiments: Experiment[] = [];

		for (const runner of runners) {
			if (opts.runner !== undefined && runner.name !== opts.runner) continue;

			for (const port of [5432, 6432]) {
				if (opts.port !== undefined && port !== opts.port) continue;

				for (const [client, [from, to, step]] of Object.entries(clients)) {
					if (opts.client !== undefined && client !== opts.client) continue;

					for (let clients = from; clients <= to; clients += step) {
						const t = Math.max(1, Math.ceil(TOTAL_TRANSACTIONS / clients));
						experiments.push({ runner, port, clients, transactions: opts.transactions ?? t, scale: opts.scale });
					}
				}
			}
		}

		await _run(experiments);
	});

	program
	.command('init')
	.description('setup pgbench tables')
	.option('-s, --scale <factor>', 'scaling factor')
	.option('--unlogged', 'create tables as unlogged')
	.option('--fillfactor <percent>', 'table fill factor percent')
	.action(async (options: OptionValues) => {
		const opts = InitOptions.parse(options);
		await setup_pgbench({ scale: opts.scale, unlogged_tables: opts.unlogged, fillfactor: opts.fillfactor });
	});

	await program.parseAsync();
}

const InitOptions = z.object({
	scale: z.coerce.number().int().min(1).default(100),
	unlogged: z.boolean().default(false),
	fillfactor: z.coerce.number().int().min(10).max(100).optional(),
});

const RunOptions = z.object({
	runner: z.enum(['node', 'pgbench']).optional(),
	port: z.coerce.number().int().min(1).max(65_535).optional(),
	client: z.enum(['small', 'large']).optional(),
	transactions: z.coerce.number().int().min(1).optional(),
	scale: z.coerce.number().int().min(1).default(100),
});

async function _run(experiments: Experiment[]) {
	for (const experiment of experiments) {
		const runner = experiment.runner;
		process.stdout.write(`measuring ${runner.name} clients=${experiment.clients} port=${experiment.port}\n`);

		let rate_samples: number[] = [];
		let latency_samples: number[] = [];
		for (;;) {
			const result = await runner.measure(experiment);
			rate_samples.push(result.rate);
			latency_samples.push(result.avg);

			const { n, outliers, max, mean, std, ratio } = stats(rate_samples);
			process.stdout.write(`n: ${n.toString().padStart(2, ' ')}, mean: ${mean.toFixed(4)}, std: ${std.toFixed(4)}, std_ratio: ${ratio.toFixed(4)}, outliers: ${outliers.join(',')}\n`);

			if (rate_samples.length >= TARGET_MEASUREMENTS && outliers.includes(max)) {
				process.stdout.write(`excluding peak ${max} from the distribution underestimates the mean, recollecting ${runner.name} clients=${experiment.clients} port=${experiment.port}\n`);
				const max_index = rate_samples.indexOf(max);
				rate_samples = [max];
				latency_samples = [latency_samples[max_index]];
				continue;
			}

			if (rate_samples.length >= TARGET_MEASUREMENTS && ratio > MAX_STD_RATIO) {
				process.stdout.write(`std ratio ${ratio} exceeds ${MAX_STD_RATIO}, recollecting ${runner.name} clients=${experiment.clients} port=${experiment.port}\n`);
				rate_samples = [];
				latency_samples = [];
				continue;
			}

			if (n >= TARGET_MEASUREMENTS) break;
		}

		const { mean, std } = stats(rate_samples);
		const { mean: mean_latency, std: std_latency } = stats(latency_samples);
		mkdirSync(path.join(script_dir, '.build'), { recursive: true });
		appendFileSync(results_file, `${JSON.stringify({
			runner: runner.name,
			port: experiment.port,
			clients: experiment.clients,
			transactions: experiment.transactions,
			scale: experiment.scale,
			timestamp: new Date().toISOString(),
			rate_samples,
			mean_rate: mean,
			std_rate: std,
			latency_samples,
			mean_latency_ms: mean_latency,
			std_latency_ms: std_latency,
		})}\n`);
	}
}

interface Experiment {
	runner: Runner;
	port: number;
	clients: number;
	transactions: number;
	scale: number;
}

interface Runner {
	name: string;
	measure(params: Omit<Experiment, 'runner'>): Promise<{ rate: number; avg: number }>;
}

const node: Runner = {
	name: 'node',
	async measure(params: Omit<Experiment, 'runner'>) {
		const result = spawnSync('node', [
			'pgbench.ts',
			'-p', params.port.toString(),
			'-c', params.clients.toString(),
			'-t', params.transactions.toString(),
			'-s', params.scale.toString(),
		], { cwd: node_pgbench_dir });

		if (result.status !== 0) throw new Error(`node pgbench failed: ${result.stderr.toString()}`);
		const output = JSON.parse(result.stdout.toString()) as {
			metrics: { transactions: { rate: number; avg: number } };
		};
		return { rate: output.metrics.transactions.rate, avg: output.metrics.transactions.avg };
	},
};

const pgbench: Runner = {
	name: 'pgbench',
	async measure(params: Omit<Experiment, 'runner'>) {
		const result = spawnSync('pgbench', [
			'-h', process.env.DB_HOST ?? '127.0.0.1',
			'-p', params.port.toString(),
			'-d', process.env.DB_DATABASE ?? 'postgres',
			'-U', process.env.DB_USER ?? 'postgres',
			'-c', params.clients.toString(),
			'-t', params.transactions.toString(),
		], { env: { ...process.env, PGPASSWORD: process.env.DB_PASSWORD ?? 'postgres' } });

		if (result.status !== 0) throw new Error(`pgbench failed: ${result.stderr.toString()}`);
		const text = `${result.stdout.toString()}\n${result.stderr.toString()}`;
		const rate = Number(text.match(/tps\s*=\s*([\d.]+)/)?.[1]);
		const avg = Number(text.match(/latency average\s*=\s*([\d.]+)\s*ms/)?.[1]);
		if (!Number.isFinite(rate) || !Number.isFinite(avg)) throw new Error(`could not parse pgbench output: ${text}`);
		return { rate, avg };
	},
};

await main();
