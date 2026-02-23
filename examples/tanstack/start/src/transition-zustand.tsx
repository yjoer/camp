/* eslint-disable react-hooks/purity */
// oxlint-disable no-console
import * as stylex from '@stylexjs/stylex';
import { createFileRoute } from '@tanstack/react-router';
import { useState, useTransition } from 'react';
import { create } from 'zustand';

import { button_styles } from '@/components/button';

export const Route = createFileRoute('/transition-zustand')({
  component: TransitionZustand,
});

function TransitionZustand() {
  console.log('render');

  const [store] = useState(create_store);

  return (
    <div className="mx-2 my-1">
      <SettingsPanel store={store} />
      <Posts store={store} />
    </div>
  );
}

interface SettingsPanelProps {
  store: ReturnType<typeof create_store>;
}

function SettingsPanel({ store }: SettingsPanelProps) {
  const page = store(state => state.page);
  const set_page = store(state => state.set_page);
  const set_page_slow = store(state => state.set_page_slow);

  const [is_pending, start_transition] = useTransition();

  const handle_click = () => {
    set_page();

    start_transition(() => {
      set_page_slow();
    });
  };

  return (
    <>
      <div>Page: {page}</div>
      <div>Pending: {is_pending ? 'true' : 'false'}</div>
      <button onClick={handle_click} {...stylex.props(button_styles.base)}>
        Next Page
      </button>
    </>
  );
}

interface PostsProps {
  store: ReturnType<typeof create_store>;
}

const Posts = function Posts({ store }: PostsProps) {
  const page = store(state => state.page_slow);

  return (
    <div className="mt-4">
      {Array.from({ length: 10 }, (_, i) => {
        const post_id = (page - 1) * 10 + i + 1;
        return <SlowPost key={post_id} post_id={post_id} />;
      })}
    </div>
  );
};

interface SlowPostProps {
  post_id: number;
}

function SlowPost({ post_id }: SlowPostProps) {
  let start_time = performance.now();
  while (performance.now() - start_time < 50);

  return <div>Slow Post #{post_id}</div>;
}

interface State {
  page: number;
  page_slow: number;
  set_page: () => void;
  set_page_slow: () => void;
}

const create_store = () => {
  return create<State>(set => ({
    page: 1,
    page_slow: 1,
    set_page: () => set(state => ({ page: state.page + 1 })),
    set_page_slow: () => set(state => ({ page_slow: state.page_slow + 1 })),
  }));
};
