import * as stylex from '@stylexjs/stylex';
import { createFileRoute } from '@tanstack/solid-router';
import { createSignal, For, useTransition } from 'solid-js';

import { button_styles } from '@/components/button';

export const Route = createFileRoute('/transition-signal')({
	component: TransitionSignal,
});

function TransitionSignal() {
	const [page, set_page] = createSignal(1);
	const [page_slow, set_page_slow] = createSignal(1);

	const [pending, start_transition] = useTransition();

	const handle_click = () => {
		set_page(prev => prev + 1);

		void start_transition(() => {
			set_page_slow(prev => prev + 1);
		});
	};

	return (
		<div {...stylex.props(styles.container)}>
			<div>Page: {page()}</div>
			<div>Pending: {pending() ? 'true' : 'false'}</div>
			<button onClick={handle_click} {...stylex.props(button_styles.base)}>
				Next Page
			</button>
			<Posts page={page_slow()} />
		</div>
	);
}

const styles = stylex.create({
	container: {
		marginBlock: 4,
		marginInline: 8,
	},
});

interface PostProps {
	page: number;
}

function Posts(props: PostProps) {
	const posts = () => {
		return Array.from({ length: 10 }).map((_, i) => {
			const post_id = (props.page - 1) * 10 + i + 1;
			return post_id;
		});
	};

	return (
		<div {...stylex.props(posts_styles.container)}>
			<For each={posts()}>
				{(post) => {
					return <SlowPost post_id={post} />;
				}}
			</For>
		</div>
	);
}

const posts_styles = stylex.create({
	container: {
		marginTop: 16,
	},
});

interface SlowPostProps {
	post_id: number;
}

function SlowPost(props: SlowPostProps) {
	return (
		<div>
			{block()}
			Slow Post #{props.post_id}
		</div>
	);
}

const block = () => {
	const start_time = performance.now();
	while (performance.now() - start_time < 50);

	return null;
};
