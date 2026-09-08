import * as stylex from '@stylexjs/stylex';
import { createFileRoute } from '@tanstack/react-router';
import { useLayoutEffect, useRef, useState } from 'react';

import { button_styles } from '@/components/button';

export const Route = createFileRoute('/ui/animations/disclosure')({
	component: AnimationDisclosure,
});

function AnimationDisclosure() {
	return (
		<div sx={styles.container}>
			<DetailsMaxHeight />
			<DetailsHeight />
			<DetailsWAAPI />
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

function DetailsMaxHeight() {
	const ref = useRef<HTMLDivElement | null>(null);
	const [state, set_state] = useState<'opening' | 'opened' | 'closing' | 'closed'>('closed');

	const handle_toggle = () => {
		if (state === 'closed' || state == 'closing') set_state('opening');
		else set_state('closing');
	};

	useLayoutEffect(() => {
		if (!ref.current) return;
		const el = ref.current;

		if (state !== 'opening' && state !== 'closing') return;

		if (state === 'opening') {
			void el.clientHeight; // reflow
			el.style.maxHeight = '72px';
		} else {
			el.style.maxHeight = '';
		}

		const handle_transition_end = () => {
			if (state === 'opening') set_state('opened');
			if (state === 'closing') set_state('closed');
		};

		el.addEventListener('transitionend', handle_transition_end);
		return () => el.removeEventListener('transitionend', handle_transition_end);
	}, [state]);

	return (
		<div>
			<span sx={dmh_styles.label}>Max Height</span>
			<div sx={dmh_styles.actions}>
				<button sx={button_styles.base} onClick={handle_toggle}>
					Toggle
				</button>
			</div>
			{ state !== 'closed' && (
				<div ref={ref} sx={dmh_styles.details}>
					<Details />
				</div>
			)}
		</div>
	);
}

const dmh_styles = stylex.create({
	label: {
		backgroundColor: '#ffa500',
	},
	actions: {
		marginBlock: 4,
	},
	details: {
		maxHeight: 0,
		overflow: 'hidden',
		transition: 'max-height 250ms',
	},
});

function DetailsHeight() {
	const ref = useRef<HTMLDivElement | null>(null);
	const [state, set_state] = useState<'opening' | 'opened' | 'closing' | 'closed'>('closed');

	const handle_toggle = () => {
		if (state === 'closed' || state == 'closing') set_state('opening');
		else set_state('closing');
	};

	useLayoutEffect(() => {
		if (!ref.current) return;
		const el = ref.current;

		if (state !== 'opening' && state !== 'closing') return;

		if (state === 'opening') {
			el.style.height = '0px';
			el.style.height = `${el.scrollHeight}px`; // reflow
		} else {
			el.style.height = '0px';
		}

		const handle_transition_end = () => {
			if (state === 'opening') set_state('opened');
			if (state === 'closing') set_state('closed');
		};

		el.addEventListener('transitionend', handle_transition_end);
		return () => el.removeEventListener('transitionend', handle_transition_end);
	}, [state]);

	return (
		<div>
			<span sx={dh_styles.label}>Height</span>
			<div sx={dh_styles.actions}>
				<button sx={button_styles.base} onClick={handle_toggle}>
					Toggle
				</button>
			</div>
			{ state !== 'closed' && (
				<div ref={ref} sx={dh_styles.details}>
					<Details />
				</div>
			)}
		</div>
	);
}

const dh_styles = stylex.create({
	label: {
		backgroundColor: '#ffa500',
	},
	actions: {
		marginBlock: 4,
	},
	details: {
		overflow: 'hidden',
		transition: 'height 250ms',
	},
});

function DetailsWAAPI() {
	const ref = useRef<HTMLDivElement | null>(null);
	const animation_ref = useRef<Animation | null>(null);
	const [state, set_state] = useState<'opening' | 'opened' | 'closing' | 'closed'>('closed');

	const handle_toggle = () => {
		if (state === 'closed' || state == 'closing') set_state('opening');
		else set_state('closing');
	};

	useLayoutEffect(() => {
		if (!ref.current) return;
		const el = ref.current;

		if (state !== 'opening' && state !== 'closing') return;

		const effect = animation_ref.current?.effect as KeyframeEffect | undefined;
		if (!animation_ref.current || effect?.target !== el) {
			animation_ref.current = el.animate([{ height: '0px' }, { height: `${el.clientHeight}px` }], {
				duration: 250,
			});
			animation_ref.current.pause();
		}

		animation_ref.current.playbackRate = state === 'opening' ? 1 : -1;
		animation_ref.current.play();

		void animation_ref.current.finished
		.then(() => {
			animation_ref.current?.commitStyles();
			animation_ref.current?.cancel();
			if (state === 'opening') set_state('opened');
			if (state === 'closing') set_state('closed');
		});
	}, [state]);

	return (
		<div>
			<span sx={dw_styles.label}>WAAPI</span>
			<div sx={dw_styles.actions}>
				<button sx={button_styles.base} onClick={handle_toggle}>
					Toggle
				</button>
			</div>
			{ state !== 'closed' && (
				<div ref={ref} sx={dw_styles.details}>
					<Details />
				</div>
			)}
		</div>
	);
}

const dw_styles = stylex.create({
	label: {
		backgroundColor: '#ffa500',
	},
	actions: {
		marginBlock: 4,
	},
	details: {
		overflow: 'hidden',
	},
});

function Details() {
	return (
		<div sx={details_styles.base}>
			🎁
		</div>
	);
}

const details_styles = stylex.create({
	base: {
		display: 'inline-block',
		padding: 24,
		backgroundColor: 'oklch(97% 0 0)',
	},
});
