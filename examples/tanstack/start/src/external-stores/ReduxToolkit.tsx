import { configureStore, createAction, createAsyncThunk, createSlice } from '@reduxjs/toolkit';
import * as stylex from '@stylexjs/stylex';
import { createFileRoute } from '@tanstack/react-router';
import { useState } from 'react';
import { Provider, useDispatch, useSelector, useStore } from 'react-redux';

import { button_styles } from '@/components/button';

import type { PayloadAction } from '@reduxjs/toolkit';

export const Route = createFileRoute('/external-stores/redux-toolkit')({
  component: ReduxToolkit,
});

function ReduxToolkit() {
  const [store] = useState<AppStore>(create_store);

  return (
    <Provider store={store}>
      <Counter />
    </Provider>
  );
}

function Counter() {
  const dispatch = useAppDispatch();

  const loading = useAppSelector(state => state.counter.loading);
  const count = useAppSelector(state => state.counter.value);

  return (
    <div className="mx-2 my-1">
      <div className="flex items-center gap-2">
        <button onClick={() => dispatch(increment())} {...stylex.props(button_styles.base)}>
          Increment
        </button>
        <span className="min-w-12 text-center">{count}</span>
        <button onClick={() => dispatch(decrement())} {...stylex.props(button_styles.base)}>
          Decrement
        </button>
      </div>
      <div className="mt-2 flex flex-col items-start gap-1">
        <button onClick={() => dispatch(increment_async())} {...stylex.props(button_styles.base)}>
          Async
        </button>
        <button onClick={() => dispatch(increment_async_a())} {...stylex.props(button_styles.base)}>
          {loading ? 'Loading' : 'Async Fulfilled'}
        </button>
        <button onClick={() => dispatch(increment_async_b())} {...stylex.props(button_styles.base)}>
          {loading ? 'Loading' : 'Async Rejected'}
        </button>
      </div>
    </div>
  );
}

const decrement = createAction('counter/decrement');

const counter_slice = createSlice({
  name: 'counter',
  initialState: {
    loading: false,
    value: 0,
  },
  reducers: {
    increment: (state) => {
      state.value += 1;
    },
    increment_by_amount: (state, action: PayloadAction<number>) => {
      state.value += action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(decrement, (state) => {
        state.value -= 1;
      })
      .addCase(increment_async_a.pending, (state) => {
        state.loading = true;
      })
      .addCase(increment_async_a.fulfilled, (state, action) => {
        state.value += action.payload;
        state.loading = false;
      })
      .addCase(increment_async_b.pending, (state) => {
        state.loading = true;
      })
      .addCase(increment_async_b.rejected, (state) => {
        state.value = -1;
        state.loading = false;
      });
  },
});

const { increment, increment_by_amount } = counter_slice.actions;

function increment_async() {
  return async (dispatch: AppDispatch) => {
    await new Promise(resolve => setTimeout(resolve, 500));
    dispatch(increment_by_amount(2));
  };
}

const increment_async_a = createAsyncThunk('counter/increment_async_a', async (_, thunk) => {
  thunk.dispatch(increment_by_amount(1));
  await new Promise(resolve => setTimeout(resolve, 500));
  return 1;
});

const increment_async_b = createAsyncThunk('counter/increment_async_b', async (_, thunk) => {
  thunk.dispatch(increment_by_amount(1));
  await new Promise((_, reject) => setTimeout(() => reject(new Error('failed')), 500));
});

function create_store() {
  return configureStore({
    reducer: {
      counter: counter_slice.reducer,
    },
  });
}

type AppStore = ReturnType<typeof create_store>;
type RootState = ReturnType<AppStore['getState']>;
type AppDispatch = AppStore['dispatch'];

const useAppDispatch = useDispatch.withTypes<AppDispatch>();
const useAppSelector = useSelector.withTypes<RootState>();
export const useAppStore = useStore.withTypes<AppStore>();
