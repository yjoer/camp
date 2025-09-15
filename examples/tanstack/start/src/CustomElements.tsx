// oxlint-disable no-namespace
import { createIsomorphicFn } from '@tanstack/react-start';

export const Route = createFileRoute({
  component: CustomElements,
});

function CustomElements() {
  return (
    <div className="mx-2 my-1">
      <custom-element>This is a custom element!</custom-element>
    </div>
  );
}

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

      this.addEventListener(
        'mouseenter',
        () => {
          this.classList.add('bg-neutral-100');
        },
        { signal: this.#controller.signal },
      );

      this.addEventListener(
        'mouseleave',
        () => {
          this.classList.remove('bg-neutral-100');
        },
        { signal: this.#controller.signal },
      );
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
