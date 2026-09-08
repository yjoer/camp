/* eslint-disable react-hooks/exhaustive-deps */
import * as stylex from '@stylexjs/stylex';
import { createFileRoute } from '@tanstack/react-router';
import { useCallback, useEffect, useEffectEvent, useRef, useState } from 'react';

import { button_styles } from '@/components/button';

export const Route = createFileRoute('/stale-closures')({
	component: StaleClosures,
});

function StaleClosures() {
	return (
		<div sx={styles.container}>
			<DependencyArray />
			<EffectEvents />
			<RefSync />
			<StateRefHook />
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

function DependencyArray() {
	const timedLogRef = useRef<HTMLDivElement>(null!);
	const timedLogStaleRef = useRef<HTMLDivElement>(null!);
	const logRef = useRef<HTMLDivElement>(null!);
	const logStaleRef = useRef<HTMLDivElement>(null!);

	const [count, setCount] = useState(0);

	useEffect(() => {
		const interval = setInterval(() => {
			timedLogRef.current.textContent = `Timed Log: ${count}`;
		}, 1000);

		return () => {
			clearInterval(interval);
		};
	}, [count]);

	useEffect(() => {
		const interval = setInterval(() => {
			timedLogStaleRef.current.textContent = `Timed Log - Stale: ${count}`;
		}, 1000);

		return () => {
			clearInterval(interval);
		};
	}, []);

	const handleClick = useCallback(() => {
		logRef.current.textContent = `Log: ${count}`;
	}, [count]);

	const handleClickStale = useCallback(() => {
		logStaleRef.current.textContent = `Log - Stale: ${count}`;
	}, []);

	return (
		<div>
			<span sx={da_styles.label}>Dependency Array</span>
			<div>Count: {count}</div>
			<div ref={timedLogRef}>Timed Log:</div>
			<div ref={timedLogStaleRef}>Timed Log - Stale:</div>
			<div ref={logRef}>Log:</div>
			<div ref={logStaleRef}>Log - Stale:</div>
			<div sx={da_styles.actions}>
				<button sx={button_styles.base} onClick={() => setCount(prev => prev + 1)}>
					Increment
				</button>
				<button sx={button_styles.base} onClick={handleClick}>
					Log
				</button>
				<button sx={button_styles.base} onClick={handleClickStale}>
					Log - Stale
				</button>
			</div>
		</div>
	);
}

const da_styles = stylex.create({
	label: {
		backgroundColor: '#ffdd00',
	},
	actions: {
		display: 'flex',
		gap: 8,
		marginTop: 4,
	},
});

function EffectEvents() {
	const [count, setCount] = useState(0);
	const timedLogRef = useRef<HTMLDivElement>(null!);

	const onInterval = useEffectEvent(() => {
		timedLogRef.current.textContent = `Timed Log: ${count}`;
	});

	useEffect(() => {
		const interval = setInterval(() => {
			onInterval();
		}, 1000);

		return () => {
			clearInterval(interval);
		};
	}, []);

	return (
		<div>
			<span sx={ee_styles.label}>Effect Events</span>
			<div>Count: {count}</div>
			<div ref={timedLogRef}>Timed Log:</div>
			<div sx={ee_styles.actions}>
				<button sx={button_styles.base} onClick={() => setCount(prev => prev + 1)}>
					Increment
				</button>
			</div>
		</div>
	);
}

const ee_styles = stylex.create({
	label: {
		backgroundColor: '#ffdd00',
	},
	actions: {
		display: 'flex',
		gap: 8,
		marginTop: 4,
	},
});

function RefSync() {
	const [count, setCount] = useState(0);
	const countRef = useRef(count);

	const timedLogRef = useRef<HTMLDivElement>(null!);
	const logRef = useRef<HTMLDivElement>(null!);

	useEffect(() => {
		const interval = setInterval(() => {
			timedLogRef.current.textContent = `Timed Log: ${countRef.current}`;
		}, 1000);

		return () => {
			clearInterval(interval);
		};
	}, []);

	const handleClick = useCallback(() => {
		logRef.current.textContent = `Log: ${countRef.current}`;
	}, []);

	const handleIncrement = useCallback(() => {
		setCount(prev => prev + 1);
		countRef.current += 1;
	}, []);

	return (
		<div>
			<span sx={rs_styles.label}>Ref Sync</span>
			<div>Count: {count}</div>
			<div ref={timedLogRef}>Timed Log:</div>
			<div ref={logRef}>Log:</div>
			<div sx={rs_styles.actions}>
				<button sx={button_styles.base} onClick={handleIncrement}>
					Increment
				</button>
				<button sx={button_styles.base} onClick={handleClick}>
					Log
				</button>
			</div>
		</div>
	);
}

const rs_styles = stylex.create({
	label: {
		backgroundColor: '#ffdd00',
	},
	actions: {
		display: 'flex',
		gap: 8,
		marginTop: 4,
	},
});

function StateRefHook() {
	const [count, setCount] = useStateRef(0);

	const timedLogRef = useRef<HTMLDivElement>(null!);
	const logRef = useRef<HTMLDivElement>(null!);

	useEffect(() => {
		const interval = setInterval(() => {
			timedLogRef.current.textContent = `Timed Log: ${count.current}`;
		}, 1000);

		return () => {
			clearInterval(interval);
		};
	}, []);

	const handleClick = useCallback(() => {
		logRef.current.textContent = `Log: ${count.current}`;
	}, []);

	const handleIncrement = useCallback(() => {
		setCount(count.current + 1);
	}, []);

	return (
		<div>
			<span sx={srh_styles.label}>State Ref Hook</span>
			<div>Count: {count.current}</div>
			<div ref={timedLogRef}>Timed Log:</div>
			<div ref={logRef}>Log:</div>
			<div sx={srh_styles.actions}>
				<button sx={button_styles.base} onClick={handleIncrement}>
					Increment
				</button>
				<button sx={button_styles.base} onClick={handleClick}>
					Log
				</button>
			</div>
		</div>
	);
}

const srh_styles = stylex.create({
	label: {
		backgroundColor: '#ffdd00',
	},
	actions: {
		display: 'flex',
		gap: 8,
		marginTop: 4,
	},
});

function useStateRef<T>(value: T): [React.RefObject<T>, (newState: T) => void] {
	const ref = useRef(value);
	const [, forceRender] = useState(false);

	function setState(newState: T) {
		if (Object.is(ref.current, newState)) return;

		ref.current = newState;
		forceRender(prev => !prev);
	}

	return [ref, setState];
}
