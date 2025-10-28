/* eslint-disable react-hooks/purity */
// oxlint-disable no-console
import { createFileRoute } from '@tanstack/react-router';
import { memo, useState, useTransition } from 'react';

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
  const [pageSlow, setPageSlow] = useState(Number.parseInt(page, 10) || 1);

  const navigate = Route.useNavigate();

  const [isPending, startTransition] = useTransition();

  const handleClick = () => {
    navigate({
      search: (prev) => ({ page: (Number(prev.page) + 1).toString() }),
    });

    startTransition(() => {
      setPageSlow((prev) => (prev ? prev + 1 : 1));
    });
  };

  return (
    <div className="mx-2 my-1">
      <div>Page: {page}</div>
      <div>Pending: {isPending ? 'true' : 'false'}</div>
      <button className="mt-1 font-semibold" onClick={handleClick}>
        Next Page
      </button>
      <Posts page={pageSlow} />
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
        const postId = (page - 1) * 10 + i + 1;
        return <SlowPost key={postId} postId={postId} />;
      })}
    </div>
  );
});

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
