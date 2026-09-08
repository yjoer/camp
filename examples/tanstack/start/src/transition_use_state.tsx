/* eslint-disable react-hooks/purity */
// oxlint-disable no-console
import * as stylex from '@stylexjs/stylex';
import { createFileRoute } from '@tanstack/react-router';
import { memo, useState, useTransition } from 'react';

import { button_styles } from '@/components/button';

export const Route = createFileRoute('/transition-use-state')({
	component: TransitionUseState,
});

function TransitionUseState() {
	console.log('render');
	const [page, set_page] = useState(1);
	const [page_slow, set_page_slow] = useState(1);

	const [is_pending, start_transition] = useTransition();

	const handle_click = () => {
		set_page(prev => prev + 1);

		start_transition(() => {
			set_page_slow(prev => prev + 1);
		});
	};

	return (
		<div sx={styles.container}>
			<div>Page: {page}</div>
			<div>Pending: {is_pending ? 'true' : 'false'}</div>
			<button sx={button_styles.base} onClick={handle_click}>
				Next Page
			</button>
			<Posts page={page_slow} />
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

const Posts = memo(function Posts({ page }: PostProps) {
	return (
		<div sx={posts_styles.container}>
			{Array.from({ length: 10 }, (_, i) => {
				const post_id = (page - 1) * 10 + i + 1;
				return <SlowPost key={post_id} post_id={post_id} />;
			})}
		</div>
	);
});

const posts_styles = stylex.create({
	container: {
		marginTop: 16,
	},
});

interface SlowPostProps {
	post_id: number;
}

function SlowPost({ post_id }: SlowPostProps) {
	const start_time = performance.now();
	while (performance.now() - start_time < 50);

	return <div>Slow Post #{post_id}</div>;
}
