/* eslint-disable react-hooks/refs */
import { createFileRoute } from '@tanstack/solid-router';
import { createSignal, onCleanup, onMount } from 'solid-js';

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
        <button
          class="cursor-pointer rounded bg-black/4 px-3 py-1 transition-transform active:scale-96"
          onClick={() => set_count((prev) => prev + 1)}>
          Increment
        </button>
        <button
          class="cursor-pointer rounded bg-black/4 px-3 py-1 transition-transform active:scale-96"
          onClick={handle_click}>
          Log
        </button>
      </div>
    </div>
  );
}
