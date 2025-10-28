/* eslint-disable react-hooks/purity */
// oxlint-disable no-console
import { createFileRoute } from '@tanstack/react-router';
import { useTransition } from 'react';
import { create } from 'zustand';

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
  const setPage = useStore((state) => state.setPage);
  const setPageSlow = useStore((state) => state.setPageSlow);

  const [isPending, startTransition] = useTransition();

  const handleClick = () => {
    setPage();

    startTransition(() => {
      setPageSlow();
    });
  };

  return (
    <>
      <div>Page: {page}</div>
      <div>Pending: {isPending ? 'true' : 'false'}</div>
      <button className="mt-1 font-semibold" onClick={handleClick}>
        Next Page
      </button>
    </>
  );
}

const Posts = function Posts() {
  const page = useStore((state) => state.pageSlow);

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

interface State {
  page: number;
  pageSlow: number;
  setPage: () => void;
  setPageSlow: () => void;
}

const useStore = create<State>((set) => ({
  page: 1,
  pageSlow: 1,
  setPage: () => set((state) => ({ page: state.page + 1 })),
  setPageSlow: () => set((state) => ({ pageSlow: state.pageSlow + 1 })),
}));
