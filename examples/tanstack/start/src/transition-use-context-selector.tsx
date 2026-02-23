/* eslint-disable react-hooks/purity */
// oxlint-disable no-console
import * as stylex from '@stylexjs/stylex';
import { createFileRoute } from '@tanstack/react-router';
import { useState, useTransition } from 'react';
import { createContext, useContextSelector, useContextUpdate } from 'use-context-selector';

import { button_styles } from '@/components/button';

export const Route = createFileRoute('/transition-use-context-selector')({
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
  const page = useContextSelector(Context, v => v.page);
  const set_page = useContextSelector(Context, v => v.set_page);
  const set_page_slow = useContextSelector(Context, v => v.set_page_slow);

  const update = useContextUpdate(Context);

  const [is_pending, start_transition] = useTransition();

  const handle_click = () => {
    set_page(prev => prev + 1);

    start_transition(() => {
      update(() => {
        set_page_slow(prev => prev + 1);
      });
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
  const page = useContextSelector(Context, v => v.page_slow);

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

interface ContextProps {
  page: number;
  page_slow: number;
  set_page: React.Dispatch<React.SetStateAction<number>>;
  set_page_slow: React.Dispatch<React.SetStateAction<number>>;
}

const Context = createContext<ContextProps>({
  page: 1,
  page_slow: 1,
  set_page: () => {},
  set_page_slow: () => {},
});

interface ProviderProps {
  children: React.ReactNode;
}

function Provider({ children }: ProviderProps) {
  const [page, set_page] = useState(1);
  const [page_slow, set_page_slow] = useState(1);

  return (
    <Context.Provider value={{ page, set_page, page_slow, set_page_slow }}>
      {children}
    </Context.Provider>
  );
}
