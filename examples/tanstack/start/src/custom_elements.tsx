// oxlint-disable no-namespace
import * as stylex from '@stylexjs/stylex';
import { createFileRoute } from '@tanstack/react-router';
import { createIsomorphicFn } from '@tanstack/react-start';

export const Route = createFileRoute('/custom-elements')({
	component: CustomElements,
});

function CustomElements() {
	return (
		<div sx={styles.container}>
			<custom-element>This is a custom element!</custom-element>
		</div>
	);
}

const styles = stylex.create({
	container: {
		marginBlock: 4,
		marginInline: 8,
	},
});

createIsomorphicFn().client(() => {
	return class Component extends HTMLElement {
		#internals: ElementInternals;
		#controller!: AbortController;

		static {
			const tag = 'custom-element';
			if (!customElements.get(tag)) customElements.define(tag, Component);
		}

		constructor() {
			super();
			this.#internals = this.attachInternals();
		}

		get isReady() {
			return this.#internals.states.has('--ready');
		}

		connectedCallback() {
			if (this.isReady) return;
			this.#internals.states.add('--ready');
			this.#controller = new AbortController();

			this.addEventListener('mouseenter', () => {
				this.style.backgroundColor = 'oklch(97% 0 0)';
			}, { signal: this.#controller.signal });

			this.addEventListener('mouseleave', () => {
				this.style.backgroundColor = '';
			}, { signal: this.#controller.signal });
		}

		disconnectedCallback() {
			this.#controller.abort();
		}
	};
})();

declare module 'react/jsx-runtime' {
	namespace JSX {
		interface IntrinsicElements {
			'custom-element': any;
		}
	}
}
