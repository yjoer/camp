import type { CompiledStyles, InlineStyles, StyleXArray } from '@stylexjs/stylex';

declare module 'react' {
	interface HTMLAttributes<T> {
		sx?: StyleXArray<
			| (CompiledStyles | null | undefined)
			| [CompiledStyles, InlineStyles]
			| boolean
		>;
	}
}
