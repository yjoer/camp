// oxlint-disable no-console
import { faker } from '@faker-js/faker';
import * as stylex from '@stylexjs/stylex';
import { createFileRoute } from '@tanstack/react-router';
import { useVirtualizer } from '@tanstack/react-virtual';
import { split } from 'canvas-hypertxt';
import { useEffectEvent, useLayoutEffect, useRef, useState } from 'react';
import { varPreLine } from 'uwrap';

export const Route = createFileRoute('/list-virt-dynamic-height')({
	validateSearch: (search: Record<string, unknown>) => {
		return {
			wrap: search.wrap as 'canvas-hypertxt' | 'uwrap' | undefined,
		};
	},
	component: DynamicHeight,
});

const heights = Array.from({ length: 1000 }, () => {
	return Math.floor(Math.random() * 200) + 150;
});

const titles = Array.from({ length: 1000 }, (_, index) => {
	faker.seed(index);
	return faker.book.title();
});

function DynamicHeight() {
	const { wrap } = Route.useSearch();

	const ref = useRef<HTMLDivElement>(null!);
	const [cell_width, set_cell_width] = useState(0);
	const [lines, set_lines] = useState<number[]>(null!);

	const uwrap_count = useEffectEvent(() => {
		if (wrap && wrap !== 'uwrap') return;

		const ctx = document.createElement('canvas').getContext('2d')!;
		ctx.font = "16px 'Roboto Variable', sans-serif";
		const width = ref.current.clientWidth / 6 - (16 + 8 * 5) / 6;
		const { count } = varPreLine(ctx);

		const t0 = performance.now();
		const lines = titles.map((title) => {
			return count(title, width);
		});
		const t1 = performance.now();

		console.log(`uwrap count: ${(t1 - t0).toFixed(2)} ms`);

		set_cell_width(width);
		set_lines(lines);
	});

	const canvas_hypertxt_split = useEffectEvent(() => {
		if (wrap !== 'canvas-hypertxt') return;

		const ctx = document.createElement('canvas').getContext('2d')!;
		ctx.font = "16px 'Roboto Variable', sans-serif";
		const width = ref.current.clientWidth / 6 - (16 + 8 * 5) / 6;

		const t0 = performance.now();
		const lines = titles.map((title) => {
			const splits = split(ctx, title, ctx.font, width, false);
			return splits.length;
		});
		const t1 = performance.now();

		console.log(`canvas-hypertxt split: ${(t1 - t0).toFixed(2)} ms`);

		set_cell_width(width);
		set_lines(lines);
	});

	useLayoutEffect(() => {
		uwrap_count();
		canvas_hypertxt_split();
	}, []);

	// eslint-disable-next-line react-hooks/incompatible-library
	const virtualizer = useVirtualizer({
		count: 1000,
		getScrollElement: () => ref.current,
		estimateSize: (index) => {
			return heights[index] + lines[index] * 20;
		},
		enabled: !!lines,
		gap: 8,
		lanes: 6,
		overscan: 1,
	});

	return (
		<div ref={ref} sx={styles.container}>
			<div style={{ height: virtualizer.getTotalSize() }} sx={styles.masonry}>
				{virtualizer.getVirtualItems().map((row) => {
					return (
						<div
							key={row.index}
							style={{
								position: 'absolute',
								width: cell_width,
								height: row.size,
								translate: `${row.lane * cell_width + row.lane * 8}px ${row.start}px`,
							}}>
							<div style={{ height: heights[row.index] }} sx={styles.image} />
							<div sx={styles.title}>{titles[row.index]}</div>
						</div>
					);
				})}
			</div>
		</div>
	);
}

const styles = stylex.create({
	container: {
		height: '100dvh',
		paddingBlock: 4,
		paddingInline: 8,
		overflow: 'auto',
		scrollbarGutter: 'stable',
	},
	masonry: {
		position: 'relative',
	},
	image: {
		width: '100%',
		backgroundColor: 'oklch(92.2% 0 0)',
	},
	title: {
		lineHeight: 1.25,
	},
});
