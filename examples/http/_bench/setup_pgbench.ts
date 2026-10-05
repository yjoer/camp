import type { Pool } from 'pg';

import { Client } from 'pg';

export interface SetupPgbenchOptions {
	scale?: number;
	unlogged_tables?: boolean;
	fillfactor?: number;
}

export async function setup_pgbench(options: SetupPgbenchOptions = {}): Promise<void> {
	const pg = new Client({
		host: process.env.DB_HOST ?? '127.0.0.1',
		port: Number.parseInt(process.env.DB_PORT ?? '6432', 10),
		user: process.env.DB_USER ?? 'postgres',
		password: process.env.DB_PASSWORD ?? 'postgres',
		database: process.env.DB_DATABASE ?? 'postgres',
	});

	await pg.connect();

	try {
		await drop_tables(pg);
		await create_tables(pg, options);
		await generate_data_client_side(pg, options);
		await vacuum_tables(pg);
		await create_primary_keys(pg);
		await create_foreign_keys(pg);
	} finally {
		await pg.end();
	}
}

const SCALE_32BIT_THRESHOLD = 20_000;
const NBRANCHES = 1;
const NTELLERS = 10;
const NACCOUNTS = 100_000;

async function drop_tables(pg: Client | Pool): Promise<void> {
	await pg.query(`drop table if exists pgbench_accounts, pgbench_branches, pgbench_history, pgbench_tellers`);
}

interface CreateTableOptions {
	scale?: number;
	unlogged_tables?: boolean;
	fillfactor?: number;
}

async function create_tables(pg: Client | Pool, options: CreateTableOptions = {}): Promise<void> {
	const scale = options.scale ?? 1;
	const unlogged = options.unlogged_tables ? ' unlogged' : '';

	const fillfactor = options.fillfactor ?? 100;
	const fillfactor_clause = ` with (fillfactor=${fillfactor})`;

	const tables = [{
		table: 'pgbench_history',
		sm_cols: 'tid int,bid int,aid int,delta int,mtime timestamp,filler char(22)',
		big_cols: 'tid int,bid int,aid bigint,delta int,mtime timestamp,filler char(22)',
		use_fillfactor: false,
	}, {
		table: 'pgbench_tellers',
		sm_cols: 'tid int not null,bid int,tbalance int,filler char(84)',
		big_cols: 'tid int not null,bid int,tbalance int,filler char(84)',
		use_fillfactor: true,
	}, {
		table: 'pgbench_accounts',
		sm_cols: 'aid int not null,bid int,abalance int,filler char(84)',
		big_cols: 'aid bigint not null,bid int,abalance int,filler char(84)',
		use_fillfactor: true,
	}, {
		table: 'pgbench_branches',
		sm_cols: 'bid int not null,bbalance int,filler char(88)',
		big_cols: 'bid int not null,bbalance int,filler char(88)',
		use_fillfactor: true,
	}];

	for (const { table, sm_cols, big_cols, use_fillfactor } of tables) {
		const cols = scale >= SCALE_32BIT_THRESHOLD ? big_cols : sm_cols;
		const query = `create${unlogged} table ${table}(${cols})${use_fillfactor ? fillfactor_clause : ''}`;
		await pg.query(query);
	}
}

interface GenerateDataOptions {
	scale?: number;
}

async function generate_data_client_side(pg: Client | Pool, options: GenerateDataOptions = {}): Promise<void> {
	const scale = options.scale ?? 1;
	await pg.query('begin');

	try {
		await pg.query(`truncate table pgbench_accounts, pgbench_branches, pgbench_history, pgbench_tellers`);

		const values_branches: string[] = [];
		for (let i = 0; i < NBRANCHES * scale; i += 1) values_branches.push(`(${i + 1}, 0)`);
		await pg.query(`insert into pgbench_branches (bid, bbalance) values ${values_branches.join(',')}`);

		const values_tellers: string[] = [];
		for (let i = 0; i < NTELLERS * scale; i += 1) values_tellers.push(`(${i + 1}, ${Math.floor(i / NTELLERS) + 1}, 0)`);
		await pg.query(`insert into pgbench_tellers (tid, bid, tbalance) values ${values_tellers.join(',')}`);

		const values_accounts: string[] = [];
		for (let i = 0; i < NACCOUNTS * scale; i += 1) values_accounts.push(`(${i + 1}, ${Math.floor(i / NACCOUNTS) + 1}, 0, '')`);
		await pg.query(`insert into pgbench_accounts (aid, bid, abalance, filler) values ${values_accounts.join(',')}`);

		await pg.query('commit');
	} catch (error) {
		await pg.query('rollback');
		throw error;
	}
}

async function vacuum_tables(pg: Client | Pool): Promise<void> {
	await pg.query('vacuum analyze pgbench_branches');
	await pg.query('vacuum analyze pgbench_tellers');
	await pg.query('vacuum analyze pgbench_accounts');
	await pg.query('vacuum analyze pgbench_history');
}

async function create_primary_keys(pg: Client | Pool): Promise<void> {
	await pg.query('alter table pgbench_branches add primary key (bid)');
	await pg.query('alter table pgbench_tellers add primary key (tid)');
	await pg.query('alter table pgbench_accounts add primary key (aid)');
}

async function create_foreign_keys(pg: Client | Pool): Promise<void> {
	const keys = [
		'alter table pgbench_tellers add constraint pgbench_tellers_bid_fkey foreign key (bid) references pgbench_branches',
		'alter table pgbench_accounts add constraint pgbench_accounts_bid_fkey foreign key (bid) references pgbench_branches',
		'alter table pgbench_history add constraint pgbench_history_bid_fkey foreign key (bid) references pgbench_branches',
		'alter table pgbench_history add constraint pgbench_history_tid_fkey foreign key (tid) references pgbench_tellers',
		'alter table pgbench_history add constraint pgbench_history_aid_fkey foreign key (aid) references pgbench_accounts',
	];

	for (const key of keys) await pg.query(key);
}
