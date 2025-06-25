// oxlint-disable no-console
import { memo, useState, useTransition } from 'react';

export const Route = createFileRoute({
  component: TransitionUseState,
});

function TransitionUseState() {
  console.log('render');
  const [page, setPage] = useState(1);
  const [pageSlow, setPageSlow] = useState(1);

  const [isPending, startTransition] = useTransition();

  const handleClick = () => {
    setPage((prev) => prev + 1);

    startTransition(() => {
      setPageSlow((prev) => prev + 1);
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
