/* eslint-disable react-hooks/refs */
import * as stylex from '@stylexjs/stylex';
import { createFileRoute } from '@tanstack/solid-router';
import { createSignal, onCleanup, onMount } from 'solid-js';

import { button_styles } from '../components/button';

export const Route = createFileRoute('/stale-closures')({
  component: StaleClosures,
});

function StaleClosures() {
  return (
    <div class="mx-2 my-1 flex flex-col gap-4">
      <Signals />
    </div>
  );
}

function Signals() {
  const [count, set_count] = createSignal(0);
  let timed_log_ref!: HTMLDivElement;
  let log_ref!: HTMLDivElement;
  let interval: ReturnType<typeof setInterval>;

  onMount(() => {
    interval = setInterval(() => {
      timed_log_ref.textContent = `Timed Log: ${count()}`;
    }, 1000);
  });

  onCleanup(() => {
    clearInterval(interval);
  });

  const handle_click = () => {
    log_ref.textContent = `Log: ${count()}`;
  };

  return (
    <div>
      <span class="bg-[#ffdd00]">Signals</span>
      <div>Count: {count()}</div>
      <div ref={timed_log_ref}>Timed Log: </div>
      <div ref={log_ref}>Log: </div>
      <div class="mt-1 flex gap-2">
        <button onClick={() => set_count((prev) => prev + 1)} {...stylex.props(button_styles.base)}>
          Increment
        </button>
        <button onClick={handle_click} {...stylex.props(button_styles.base)}>
          Log
        </button>
      </div>
    </div>
  );
}
