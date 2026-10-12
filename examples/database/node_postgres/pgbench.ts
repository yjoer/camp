import type { Pool } from 'pg';

import { program } from 'commander';
import { Pool as PgPool } from 'pg';
import { z } from 'zod';

const NACCOUNTS = 100_000;
const NBRANCHES = 1;
const NTELLERS = 10;

async function main() {
	program
	.option('-p, --port <port>', 'database port')
	.option('-c, --clients <count>', 'number of pooled clients')
	.option('-t, --transactions <count>', 'number of transactions each client runs')
	.option('-s, --scale <factor>', 'scaling factor')
	.option('--pipeline', 'send queries without waiting for each result');

	program.parse();
	const opts = CliOptions.parse(program.opts());

	const pool = new PgPool({
		host: process.env.DB_HOST ?? '127.0.0.1',
		port: opts.port,
		user: process.env.DB_USER ?? 'postgres',
		password: process.env.DB_PASSWORD ?? 'postgres',
		database: process.env.DB_DATABASE ?? 'postgres',
		max: opts.clients,
		min: opts.clients,
		pipeline: opts.pipeline,
	});

	try {
		await Promise.all(Array.from({ length: opts.clients }, async () => {
			const client = await pool.connect();
			client.release();
		}));

		const start = performance.now();
		const results = await Promise.all(Array.from({ length: opts.clients }, async () => {
			const timings: number[] = [];
			for (let i = 0; i < opts.transactions; i += 1) timings.push(await transact(pool, opts));
			return timings;
		}));
		const elapsed = (performance.now() - start) / 1000;

		const samples = results.flat();
		const mean = samples.reduce((a, b) => a + b, 0) / samples.length;

		const output = {
			metrics: {
				options: opts,
				transactions: {
					rate: opts.clients * opts.transactions / elapsed,
					avg: mean,
				},
			},
		};
		process.stdout.write(`${JSON.stringify(output)}\n`);
	} finally {
		await pool.end();
	}
}

const CliOptions = z.object({
	port: z.coerce.number().int().min(1).max(65_535).prefault(process.env.DB_PORT ?? '6432'),
	clients: z.coerce.number().int().min(1).default(1),
	transactions: z.coerce.number().int().min(1).default(10),
	scale: z.coerce.number().int().min(1).default(1),
	pipeline: z.boolean().default(false),
});

interface TransactOptions {
	scale: number;
}

async function transact(pool: Pool, options: TransactOptions): Promise<number> {
	const aid = random_int(1, NACCOUNTS * options.scale);
	const bid = random_int(1, NBRANCHES * options.scale);
	const tid = random_int(1, NTELLERS * options.scale);
	const delta = random_int(-5000, 5000);

	const client = await pool.connect();
	const start = performance.now();

	try {
		await client.query('begin');
		await client.query('update pgbench_accounts set abalance = abalance + $1 where aid = $2', [delta, aid]);
		await client.query('select abalance from pgbench_accounts where aid = $1', [aid]);
		await client.query('update pgbench_tellers set tbalance = tbalance + $1 where tid = $2', [delta, tid]);
		await client.query('update pgbench_branches set bbalance = bbalance + $1 where bid = $2', [delta, bid]);
		await client.query('insert into pgbench_history (tid, bid, aid, delta, mtime) values ($1, $2, $3, $4, current_timestamp)', [tid, bid, aid, delta]);
		await client.query('commit');
		return performance.now() - start;
	} catch (error) {
		await client.query('rollback');
		throw error;
	} finally {
		client.release();
	}
}

function random_int(min: number, max: number): number {
	return Math.floor(Math.random() * (max - min + 1)) + min;
}

await main();
