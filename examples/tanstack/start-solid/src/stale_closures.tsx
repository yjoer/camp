/* eslint-disable react-hooks/refs */
import * as stylex from '@stylexjs/stylex';
import { createFileRoute } from '@tanstack/solid-router';
import { createSignal, onCleanup, onMount } from 'solid-js';

import { button_styles } from '@/components/button';

export const Route = createFileRoute('/stale-closures')({
	component: StaleClosures,
});

function StaleClosures() {
	return (
		<div {...stylex.props(styles.container)}>
			<Signals />
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
});

function Signals() {
	const [count, set_count] = createSignal(0);
	let timed_log_ref!: HTMLDivElement;
	let log_ref!: HTMLDivElement;
	let interval: ReturnType<typeof setInterval>;

	onMount(() => {
		interval = setInterval(() => {
			timed_log_ref.textContent = `Timed Log: ${count()}`;
		}, 1000);
	});

	onCleanup(() => {
		clearInterval(interval);
	});

	const handle_click = () => {
		log_ref.textContent = `Log: ${count()}`;
	};

	return (
		<div>
			<span {...stylex.props(signals_styles.label)}>Signals</span>
			<div>Count: {count()}</div>
			<div ref={timed_log_ref}>Timed Log: </div>
			<div ref={log_ref}>Log: </div>
			<div {...stylex.props(signals_styles.actions)}>
				<button onClick={() => set_count(prev => prev + 1)} {...stylex.props(button_styles.base)}>
					Increment
				</button>
				<button onClick={handle_click} {...stylex.props(button_styles.base)}>
					Log
				</button>
			</div>
		</div>
	);
}

const signals_styles = stylex.create({
	label: {
		backgroundColor: '#ffdd00',
	},
	actions: {
		display: 'flex',
		gap: 8,
		marginTop: 4,
	},
});
