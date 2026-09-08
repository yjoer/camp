import type { RouterClient } from '@orpc/server';

import { createORPCClient } from '@orpc/client';
import { RPCLink } from '@orpc/client/message-port';
import { MessageType } from '@orpc/standard-server-peer';
import * as stylex from '@stylexjs/stylex';
import { createFileRoute } from '@tanstack/react-router';
import { useEffect, useRef, useState } from 'react';

import type { router } from '@/lib/canvas_worker';

import { button_styles } from '@/components/button';
import CanvasWorker from '@/lib/canvas_worker?worker';

export const Route = createFileRoute('/worker-offscreen-canvas')({
	component: OffscreenCanvas,
});

function OffscreenCanvas() {
	const ref = useRef<HTMLDivElement>(null!);
	const worker_ref = useRef<CanvasWorkerClient>(null);
	const [date, set_date] = useState('');

	const handle_click = () => {
		if (!worker_ref.current) return;

		void worker_ref.current.get_date().then((date) => {
			set_date(date.toISOString());
		});
	};

	useEffect(() => {
		const canvas = document.createElement('canvas');
		ref.current.append(canvas);

		const offscreen = canvas.transferControlToOffscreen();
		const worker = get_worker_client();
		worker_ref.current = worker;

		transferables.add(offscreen);
		void worker.render({ canvas: offscreen });

		return () => {
			canvas.remove();
		};
	}, []);

	return (
		<div sx={styles.container}>
			<div>
				<span sx={styles.label}>Date</span>
				<div sx={styles.date}>{date}</div>
				<div sx={styles.actions}>
					<button sx={button_styles.base} onClick={handle_click}>
						Get Date
					</button>
				</div>
			</div>
			<div ref={ref}>
				<span sx={styles.label}>Canvas</span>
			</div>
		</div>
	);
}

const styles = stylex.create({
	container: {
		display: 'flex',
		flexDirection: 'column',
		gap: 16,
		marginBlock: 4,
		marginInline: 8,
	},
	label: {
		backgroundColor: '#ffdd00',
	},
	date: {
		height: 24,
	},
	actions: {
		marginTop: 4,
	},
});

// oxlint-disable-next-line typescript/no-unnecessary-type-arguments
const transferables = new WeakSet<Transferable>();

function get_worker_client(): CanvasWorkerClient {
	const link = new RPCLink({
		port: new CanvasWorker(),
		experimental_transfer: (message) => {
			const [_id, type, payload] = message;
			if (type !== MessageType.REQUEST) return [];

			const transfer: Transferable[] = [];
			const body = payload.body as { json: Record<string, unknown> } | undefined;
			for (const v of Object.values(body?.json ?? {})) {
				if (transferables.has(v as object)) transfer.push(v as Transferable);
			}

			return transfer;
		},
	});

	return createORPCClient(link);
}

type CanvasWorkerClient = RouterClient<typeof router>;
