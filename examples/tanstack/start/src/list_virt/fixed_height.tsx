import { faker } from '@faker-js/faker';
import * as stylex from '@stylexjs/stylex';
import { createFileRoute } from '@tanstack/react-router';

export const Route = createFileRoute('/list-virt-fixed-height')({
	validateSearch: (search: Record<string, unknown>) => {
		return {
			variant: (search.variant as 'content-visibility' | 'default' | undefined) ?? 'default',
		};
	},
	component: FixedHeight,
});

const titles = Array.from({ length: 10_000 }, (_, index) => {
	faker.seed(index);
	return faker.book.title();
});

function FixedHeight() {
	const { variant } = Route.useSearch();

	return (
		<div sx={styles.container}>
			<div sx={styles.grid}>
				{titles.map((title, index) => (
					<div
						key={index}
						style={{ ...(variant === 'content-visibility' && { contentVisibility: 'auto' }) }}>
						<div sx={styles.image} />
						<div sx={styles.title}>{title}</div>
					</div>
				))}
			</div>
		</div>
	);
}

const styles = stylex.create({
	container: {
		paddingBlock: 4,
		paddingInline: 8,
		scrollbarGutter: 'stable',
	},
	grid: {
		display: 'grid',
		gridTemplateColumns: 'repeat(6, minmax(0, 1fr))',
		gap: 8,
	},
	image: {
		width: '100%',
		height: 160,
		backgroundColor: 'oklch(92.2% 0 0)',
	},
	title: {
		lineHeight: 1.25,
	},
});
