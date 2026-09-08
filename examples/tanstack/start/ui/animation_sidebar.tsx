import * as stylex from '@stylexjs/stylex';
import { createFileRoute } from '@tanstack/react-router';
import { useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

import { button_styles } from '@/components/button';

export const Route = createFileRoute('/ui/animations/sidebar')({
	component: AnimationSidebar,
});

function AnimationSidebar() {
	return (
		<div sx={styles.container}>
			<SidebarTransition />
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

function SidebarTransition() {
	const ref = useRef<HTMLDivElement | null>(null);
	const [state, set_state] = useState<'closed' | 'closing' | 'opened' | 'opening'>('closed');

	const handle_toggle = () => {
		if (state === 'closed' || state == 'closing') set_state('opening');
		else set_state('closing');
	};

	useLayoutEffect(() => {
		if (!ref.current) return;
		const el = ref.current;

		if (state !== 'opening' && state !== 'closing') return;

		if (state === 'opening') {
			el.style.transition = 'transform 250ms';
			void el.clientWidth; // reflow
			el.style.transform = 'translateX(0)';
		} else {
			el.style.transform = 'translateX(100%)';
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
			<span sx={st_styles.label}>CSS Transform</span>
			<div sx={st_styles.actions}>
				<button sx={button_styles.base} onClick={handle_toggle}>
					Toggle
				</button>
			</div>
			{state !== 'closed' && createPortal(
				<div sx={st_styles.portal}>
					<Sidebar ref={ref} />
				</div>,
				document.body,
			)}
		</div>
	);
}

const st_styles = stylex.create({
	label: {
		backgroundColor: '#ffa500',
	},
	actions: {
		marginTop: 4,
	},
	portal: {
		position: 'absolute',
		inset: 0,
		overflow: 'hidden',
		pointerEvents: 'none',
	},
});

function SidebarWAAPI() {
	const ref = useRef<HTMLDivElement | null>(null);
	const animation_ref = useRef<Animation | null>(null);
	const [state, set_state] = useState<'closed' | 'closing' | 'opened' | 'opening'>('closed');

	const handle_toggle = () => {
		if (state === 'closed' || state === 'closing') set_state('opening');
		else set_state('closing');
	};

	useLayoutEffect(() => {
		if (!ref.current) return;
		const el = ref.current;

		if (state !== 'opening' && state !== 'closing') return;

		const effect = animation_ref.current?.effect as KeyframeEffect | undefined;
		if (!animation_ref.current || effect?.target !== el) {
			animation_ref.current = el.animate(
				[{ transform: 'translateX(100%)' }, { transform: 'translateX(0)' }],
				{
					duration: 250,
					easing: 'ease-in-out',
				},
			);
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
			<span sx={sw_styles.label}>WAAPI Transform</span>
			<div sx={sw_styles.actions}>
				<button sx={button_styles.base} onClick={handle_toggle}>
					Toggle
				</button>
			</div>
			{state !== 'closed' && createPortal(
				<div sx={sw_styles.portal}>
					<Sidebar ref={ref} />
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
		marginTop: 4,
	},
	portal: {
		position: 'absolute',
		inset: 0,
		overflow: 'hidden',
		pointerEvents: 'none',
	},
});

interface SidebarProps {
	ref: React.Ref<HTMLDivElement>;
}

function Sidebar({ ref }: SidebarProps) {
	return (
		<div ref={ref} sx={sidebar_styles.base}>
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
		position: 'absolute',
		top: 0,
		right: 0,
		display: 'flex',
		flexDirection: 'column',
		height: '100%',
		borderLeftColor: 'oklch(92% 0 0 / 1)',
		borderLeftStyle: 'solid',
		borderLeftWidth: 1,
		transform: 'translateX(100%)',
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
