/* eslint-disable react-hooks/purity */
// oxlint-disable no-console
import * as stylex from '@stylexjs/stylex';
import { createFileRoute } from '@tanstack/react-router';
import { useTransition } from 'react';

import { button_styles } from '@/components/button';
import { setPage, setPageSlow } from '@/state/TransitionReduxSlice';
import { useAppDispatch, useAppSelector } from '@/state/index';

export const Route = createFileRoute('/transition-redux')({
  component: TransitionRedux,
});

function TransitionRedux() {
  console.log('render');

  return (
    <div className="mx-2 my-1">
      <SettingsPanel />
      <Posts />
    </div>
  );
}

function SettingsPanel() {
  const page = useAppSelector((state) => state['transition-redux'].page);

  const dispatch = useAppDispatch();

  const [isPending, startTransition] = useTransition();

  const handleClick = () => {
    dispatch(setPage());

    startTransition(() => {
      dispatch(setPageSlow());
    });
  };

  return (
    <>
      <div>Page: {page}</div>
      <div>Pending: {isPending ? 'true' : 'false'}</div>
      <button onClick={handleClick} {...stylex.props(button_styles.base)}>
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
        const postId = (page - 1) * 10 + i + 1;
        return <SlowPost key={postId} postId={postId} />;
      })}
    </div>
  );
};

interface SlowPostProps {
  postId: number;
}

function SlowPost({ postId }: SlowPostProps) {
  let startTime = performance.now();
  while (performance.now() - startTime < 50) {
    //
  }

  return <div>Slow Post #{postId}</div>;
}
