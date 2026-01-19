/* eslint-disable perfectionist/sort-switch-case */
/* eslint-disable perfectionist/sort-union-types */
import * as stylex from '@stylexjs/stylex';
import { createFileRoute } from '@tanstack/react-router';
import { useState } from 'react';
import { Provider, useDispatch, useSelector, useStore } from 'react-redux';
import { applyMiddleware, combineReducers, legacy_createStore } from 'redux';
import { thunk } from 'redux-thunk';

import { button_styles } from '@/components/button';

export const Route = createFileRoute('/external-stores/redux')({
  component: Redux,
});

function Redux() {
  const [store] = useState<AppStore>(create_store);

  return (
    <Provider store={store}>
      <Counter />
    </Provider>
  );
}

function Counter() {
  const dispatch = useAppDispatch();

  const loading = useAppSelector((state) => state.counter.loading);
  const count = useAppSelector((state) => state.counter.value);

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

const INCREMENT = 'INCREMENT';
const DECREMENT = 'DECREMENT';
const INCREMENT_BY_AMOUNT = 'INCREMENT_BY_AMOUNT';

const INCREMENT_PENDING = 'INCREMENT_PENDING';
const INCREMENT_FULFILLED = 'INCREMENT_FULFILLED';
const INCREMENT_REJECTED = 'INCREMENT_REJECTED';

type CounterAction =
  | { type: typeof INCREMENT }
  | { type: typeof DECREMENT }
  | { type: typeof INCREMENT_BY_AMOUNT; payload: number }
  | { type: typeof INCREMENT_PENDING }
  | { type: typeof INCREMENT_FULFILLED; payload: number }
  | { type: typeof INCREMENT_REJECTED };

function increment(): CounterAction {
  return { type: INCREMENT };
}

function increment_by_amount(value: number): CounterAction {
  return { type: INCREMENT_BY_AMOUNT, payload: value };
}

function decrement(): CounterAction {
  return { type: DECREMENT };
}

function increment_async() {
  return async (dispatch: AppDispatch) => {
    await new Promise((resolve) => setTimeout(resolve, 500));
    dispatch(increment_by_amount(2));
  };
}

function increment_async_a() {
  return async (dispatch: AppDispatch) => {
    dispatch({ type: INCREMENT_PENDING });
    dispatch(increment_by_amount(1));

    try {
      await new Promise((resolve) => setTimeout(resolve, 500));
      dispatch({ type: INCREMENT_FULFILLED, payload: 1 });
    } catch {
      dispatch({ type: INCREMENT_REJECTED });
    }
  };
}

function increment_async_b() {
  return async (dispatch: AppDispatch) => {
    dispatch({ type: INCREMENT_PENDING });
    dispatch(increment_by_amount(1));

    try {
      await new Promise((_, reject) => setTimeout(() => reject(new Error('failed')), 500));
      dispatch({ type: INCREMENT_FULFILLED, payload: 1 });
    } catch {
      dispatch({ type: INCREMENT_REJECTED });
    }
  };
}

const initial_state = {
  loading: false,
  value: 0,
};

function counter_reducer(state = initial_state, action: CounterAction) {
  switch (action.type) {
    case INCREMENT: {
      return { ...state, value: state.value + 1 };
    }
    case DECREMENT: {
      return { ...state, value: state.value - 1 };
    }
    case INCREMENT_BY_AMOUNT: {
      return { ...state, value: state.value + action.payload };
    }
    case INCREMENT_PENDING: {
      return { ...state, loading: true };
    }
    case INCREMENT_FULFILLED: {
      return { ...state, loading: false, value: state.value + action.payload };
    }
    case INCREMENT_REJECTED: {
      return { ...state, loading: false, value: -1 };
    }
    default: {
      return state;
    }
  }
}

function create_store() {
  const middlewares = applyMiddleware(thunk);

  const reducer = combineReducers({
    counter: counter_reducer,
  });

  return legacy_createStore(reducer, undefined, middlewares);
}

type AppStore = ReturnType<typeof create_store>;
type RootState = ReturnType<AppStore['getState']>;
type AppDispatch = AppStore['dispatch'];

const useAppDispatch = useDispatch.withTypes<AppDispatch>();
const useAppSelector = useSelector.withTypes<RootState>();
export const useAppStore = useStore.withTypes<AppStore>();
