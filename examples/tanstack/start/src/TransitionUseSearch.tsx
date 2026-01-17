/* eslint-disable react-hooks/purity */
// oxlint-disable no-console
import * as stylex from '@stylexjs/stylex';
import { createFileRoute } from '@tanstack/react-router';
import { memo, useState, useTransition } from 'react';

import { button_styles } from '@/components/button';

export const Route = createFileRoute('/transition-use-search')({
  component: TransitionSearch,
  validateSearch: (search: Record<string, unknown>) => {
    return {
      page: (search.page as string) || '1',
    };
  },
});

function TransitionSearch() {
  console.log('render');
  const { page } = Route.useSearch();
  const [page_slow, set_page_slow] = useState(Number.parseInt(page, 10) || 1);

  const navigate = Route.useNavigate();

  const [is_pending, start_transition] = useTransition();

  const handle_click = () => {
    navigate({
      search: (prev) => ({ page: (Number(prev.page) + 1).toString() }),
    });

    start_transition(() => {
      set_page_slow((prev) => (prev ? prev + 1 : 1));
    });
  };

  return (
    <div className="mx-2 my-1">
      <div>Page: {page}</div>
      <div>Pending: {is_pending ? 'true' : 'false'}</div>
      <button onClick={handle_click} {...stylex.props(button_styles.base)}>
        Next Page
      </button>
      <Posts page={page_slow} />
    </div>
  );
}

interface PostProps {
  page: number;
}

const Posts = memo(function Posts({ page }: PostProps) {
  return (
    <div className="mt-4">
      {Array.from({ length: 10 }, (_, i) => {
        const post_id = (page - 1) * 10 + i + 1;
        return <SlowPost key={post_id} post_id={post_id} />;
      })}
    </div>
  );
});

interface SlowPostProps {
  post_id: number;
}

function SlowPost({ post_id }: SlowPostProps) {
  let start_time = performance.now();
  while (performance.now() - start_time < 50);

  return <div>Slow Post #{post_id}</div>;
}
