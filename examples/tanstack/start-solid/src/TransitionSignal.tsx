import { createFileRoute } from '@tanstack/solid-router';
import { createSignal, For, useTransition } from 'solid-js';

export const Route = createFileRoute('/transition-signal')({
  component: TransitionSignal,
});

function TransitionSignal() {
  const [page, set_page] = createSignal(1);
  const [page_slow, set_page_slow] = createSignal(1);

  const [pending, start_transition] = useTransition();

  const handle_click = () => {
    set_page((prev) => prev + 1);

    start_transition(() => {
      set_page_slow((prev) => prev + 1);
    });
  };

  return (
    <div class="mx-2 my-1">
      <div>Page: {page()}</div>
      <div>Pending: {pending() ? 'true' : 'false'}</div>
      <button class="mt-1 font-semibold" onClick={handle_click}>
        Next Page
      </button>
      <Posts page={page_slow()} />
    </div>
  );
}

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
    <div class="mt-4">
      <For each={posts()}>
        {(post) => {
          return <SlowPost post_id={post} />;
        }}
      </For>
    </div>
  );
}

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
  let start_time = performance.now();
  while (performance.now() - start_time < 50) {
    //
  }

  return null;
};
