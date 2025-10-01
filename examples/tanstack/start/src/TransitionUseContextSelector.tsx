/* eslint-disable react-hooks/purity */
// oxlint-disable no-console
import { useState, useTransition } from 'react';
import { createContext, useContextSelector, useContextUpdate } from 'use-context-selector';

export const Route = createFileRoute({
  component: TransitionUseContextSelector,
});

function TransitionUseContextSelector() {
  console.log('render');

  return (
    <Provider>
      <div className="mx-2 my-1">
        <SettingsPanel />
        <Posts />
      </div>
    </Provider>
  );
}

function SettingsPanel() {
  const page = useContextSelector(Context, (v) => v.page);
  const setPage = useContextSelector(Context, (v) => v.setPage);
  const setPageSlow = useContextSelector(Context, (v) => v.setPageSlow);

  const update = useContextUpdate(Context);

  const [isPending, startTransition] = useTransition();

  const handleClick = () => {
    setPage((prev) => prev + 1);

    startTransition(() => {
      update(() => {
        setPageSlow((prev) => prev + 1);
      });
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
  const page = useContextSelector(Context, (v) => v.pageSlow);

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

interface ContextProps {
  page: number;
  pageSlow: number;
  setPage: React.Dispatch<React.SetStateAction<number>>;
  setPageSlow: React.Dispatch<React.SetStateAction<number>>;
}

const Context = createContext<ContextProps>({
  page: 1,
  pageSlow: 1,
  setPage: () => {},
  setPageSlow: () => {},
});

interface ProviderProps {
  children: React.ReactNode;
}

function Provider({ children }: ProviderProps) {
  const [page, setPage] = useState(1);
  const [pageSlow, setPageSlow] = useState(1);

  return (
    <Context.Provider value={{ page, setPage, pageSlow, setPageSlow }}>{children}</Context.Provider>
  );
}
