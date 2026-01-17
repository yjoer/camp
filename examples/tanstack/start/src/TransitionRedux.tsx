/* eslint-disable react-hooks/purity */
// oxlint-disable no-console
import { configureStore, createSlice } from '@reduxjs/toolkit';
import * as stylex from '@stylexjs/stylex';
import { createFileRoute } from '@tanstack/react-router';
import { useState, useTransition } from 'react';
import { Provider, useDispatch, useSelector, useStore } from 'react-redux';

import { button_styles } from '@/components/button';

import type { Action, ThunkAction } from '@reduxjs/toolkit';
import type { TypedUseSelectorHook } from 'react-redux';

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
  const page = useAppSelector((state) => state.posts.page);

  const dispatch = useAppDispatch();

  const [is_pending, start_transition] = useTransition();

  const handle_click = () => {
    dispatch(set_page());

    start_transition(() => {
      dispatch(set_page_slow());
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
  const page = useAppSelector((state) => state.posts.page_slow);

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

const posts_slice = createSlice({
  name: 'posts',
  initialState: {
    page: 1,
    page_slow: 1,
  },
  reducers: {
    set_page: (state) => {
      state.page += 1;
    },
    set_page_slow: (state) => {
      state.page_slow += 1;
    },
  },
});

const { set_page, set_page_slow } = posts_slice.actions;

function create_store() {
  const reducer = {
    posts: posts_slice.reducer,
  };

  return configureStore({ reducer });
}

type AppStore = ReturnType<typeof create_store>;
type RootState = ReturnType<AppStore['getState']>;
type AppDispatch = AppStore['dispatch'];
export type AppThunk<ReturnType = void> = ThunkAction<ReturnType, RootState, unknown, Action>;

const useAppDispatch: () => AppDispatch = useDispatch;
const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;
export const useAppStore: () => AppStore = useStore;
