import * as stylex from '@stylexjs/stylex';
import { createFileRoute } from '@tanstack/react-router';
import { useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

import { button_styles } from '@/components/button';

export const Route = createFileRoute('/ui/animations/sidebar-w')({
	component: AnimationSidebarW,
});

function AnimationSidebarW() {
	return (
		<div sx={styles.container}>
			<SidebarMaxWidth />
			<SidebarWidth />
			<SidebarWAAPI />
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

function SidebarMaxWidth() {
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
			void el.clientWidth; // reflow
			el.style.maxWidth = '300px';
		} else {
			el.style.maxWidth = '0px';
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
			<span sx={smw_styles.label}>Max Width</span>
			<div sx={smw_styles.actions}>
				<button sx={button_styles.base} onClick={handle_toggle}>
					Toggle
				</button>
			</div>
			{state !== 'closed' && createPortal(
				<div sx={smw_styles.portal}>
					<div ref={ref} sx={smw_styles.sidebar_container}>
						<Sidebar />
					</div>
				</div>,
				document.body,
			)}
		</div>
	);
}

const smw_styles = stylex.create({
	label: {
		backgroundColor: '#ffa500',
	},
	actions: {
		marginBlock: 4,
	},
	portal: {
		position: 'absolute',
		inset: 0,
		pointerEvents: 'none',
	},
	sidebar_container: {
		position: 'absolute',
		top: 0,
		right: 0,
		bottom: 0,
		maxWidth: 0,
		overflow: 'hidden',
		transition: 'max-width 250ms',
	},
});

function SidebarWidth() {
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
			el.style.width = '0px';
			el.style.width = `${el.scrollWidth}px`;
		} else {
			el.style.width = '0px';
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
			<span sx={sw_styles.label}>Width</span>
			<div sx={sw_styles.actions}>
				<button sx={button_styles.base} onClick={handle_toggle}>
					Toggle
				</button>
			</div>
			{state !== 'closed' && createPortal(
				<div sx={sw_styles.portal}>
					<div ref={ref} sx={sw_styles.sidebar_container}>
						<Sidebar />
					</div>
				</div>,
				document.body,
			)}
		</div>
	);
}

const sw_styles = stylex.create({
	label: {
		backgroundColor: '#ffa500',
	},
	actions: {
		marginBlock: 4,
	},
	portal: {
		position: 'absolute',
		inset: 0,
		pointerEvents: 'none',
	},
	sidebar_container: {
		position: 'absolute',
		top: 0,
		right: 0,
		bottom: 0,
		width: 0,
		overflow: 'hidden',
		transition: 'width 250ms',
	},
});

function SidebarWAAPI() {
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
			animation_ref.current = el.animate([{ width: '0px' }, { width: `${el.scrollWidth}px` }], {
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
			<span sx={swa_styles.label}>WAAPI</span>
			<div sx={swa_styles.actions}>
				<button sx={button_styles.base} onClick={handle_toggle}>
					Toggle
				</button>
			</div>
			{state !== 'closed' && createPortal(
				<div sx={swa_styles.portal}>
					<div ref={ref} sx={swa_styles.sidebar_container}>
						<Sidebar />
					</div>
				</div>,
				document.body,
			)}
		</div>
	);
}

const swa_styles = stylex.create({
	label: {
		backgroundColor: '#ffa500',
	},
	actions: {
		marginBlock: 4,
	},
	portal: {
		position: 'absolute',
		inset: 0,
		pointerEvents: 'none',
	},
	sidebar_container: {
		position: 'absolute',
		top: 0,
		right: 0,
		bottom: 0,
		width: 0,
		overflow: 'hidden',
	},
});

function Sidebar() {
	return (
		<div sx={sidebar_styles.base}>
			{Array.from({ length: 5 }).map((_, idx) => (
				<div key={idx} sx={sidebar_styles.item}>
					<div sx={sidebar_styles.item_icon} />
					<div sx={sidebar_styles.item_label} />
				</div>
			))}
		</div>
	);
}

const sidebar_styles = stylex.create({
	base: {
		display: 'flex',
		flexDirection: 'column',
		gap: 8,
		width: 'max-content',
		height: '100%',
		paddingBlock: 8,
		borderLeftColor: 'oklch(92% 0 0 / 1)',
		borderLeftStyle: 'solid',
		borderLeftWidth: 1,
	},
	item: {
		display: 'flex',
		gap: 8,
		paddingBlock: 4,
		paddingInline: 8,
	},
	item_icon: {
		width: 24,
		height: 24,
		backgroundColor: 'oklch(92% 0 0)',
	},
	item_label: {
		width: 240,
		height: 24,
		backgroundColor: 'oklch(92% 0 0)',
	},
});
