/* eslint-disable react-hooks/purity */
// oxlint-disable no-console
import * as stylex from '@stylexjs/stylex';
import { createFileRoute } from '@tanstack/react-router';
import { useTransition } from 'react';
import { create } from 'zustand';

import { button_styles } from '@/components/button';

export const Route = createFileRoute('/transition-zustand')({
  component: TransitionZustand,
});

function TransitionZustand() {
  console.log('render');

  return (
    <div className="mx-2 my-1">
      <SettingsPanel />
      <Posts />
    </div>
  );
}

function SettingsPanel() {
  const page = useStore((state) => state.page);
  const set_page = useStore((state) => state.set_page);
  const set_page_slow = useStore((state) => state.set_page_slow);

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

const Posts = function Posts() {
  const page = useStore((state) => state.page_slow);

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

const useStore = create<State>((set) => ({
  page: 1,
  page_slow: 1,
  set_page: () => set((state) => ({ page: state.page + 1 })),
  set_page_slow: () => set((state) => ({ page_slow: state.page_slow + 1 })),
}));
