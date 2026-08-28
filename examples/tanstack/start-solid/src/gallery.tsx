import * as stylex from '@stylexjs/stylex';
import { createFileRoute, Link } from '@tanstack/solid-router';
import { For } from 'solid-js';

export const Route = createFileRoute('/')({
	component: Gallery,
});

const sections = [
	{
		label: 'Managing State',
		children: [
			{ to: '/stale-closures', label: 'Stale Closures' },
		],
	},
	{
		label: 'Concurrent Rendering',
		children: [
			{ to: '/transition-signal', label: 'useTransition with createSignal' },
		],
	},
];

function Gallery() {
	return (
		<div {...stylex.props(styles.container)}>
			<For each={sections}>
				{(section) => {
					return (
						<div>
							<span {...stylex.props(styles.label)}>{section.label}</span>
							<For each={section.children}>
								{(child) => {
									return <Link to={child.to} {...stylex.props(styles.link)}>{child.label}</Link>;
								}}
							</For>
						</div>
					);
				}}
			</For>
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
	link: {
		display: 'block',
	},
});
