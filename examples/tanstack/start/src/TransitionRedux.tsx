/* eslint-disable react-hooks/purity */
// oxlint-disable no-console
import * as stylex from '@stylexjs/stylex';
import { createFileRoute } from '@tanstack/react-router';
import { useState, useTransition } from 'react';
import { Provider } from 'react-redux';

import { button_styles } from '@/components/button';
import { setPage, setPageSlow } from '@/state/TransitionReduxSlice';
import { create_store, useAppDispatch, useAppSelector } from '@/state/index';

import type { AppStore } from '@/state/index';

export const Route = createFileRoute('/transition-redux')({
  component: TransitionRedux,
});

function TransitionRedux() {
  console.log('render');

  const [store] = useState<AppStore>(create_store);

  return (
    <Provider store={store}>
      <div className="mx-2 my-1">
        <SettingsPanel />
        <Posts />
      </div>
    </Provider>
  );
}

function SettingsPanel() {
  const page = useAppSelector((state) => state['transition-redux'].page);

  const dispatch = useAppDispatch();

  const [is_pending, start_transition] = useTransition();

  const handle_click = () => {
    dispatch(setPage());

    start_transition(() => {
      dispatch(setPageSlow());
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
  const page = useAppSelector((state) => state['transition-redux'].pageSlow);

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
