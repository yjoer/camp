import * as stylex from '@stylexjs/stylex';
import { createFileRoute } from '@tanstack/react-router';
import { useState } from 'react';
import { create } from 'zustand';

import { button_styles } from '@/components/button';

import type { StateCreator } from 'zustand';

export const Route = createFileRoute('/external-stores/zustand')({
  component: Zustand,
});

function Zustand() {
  const [store] = useState<AppStore>(create_store);

  return <Counter store={store} />;
}

interface CounterProps {
  store: AppStore;
}

function Counter({ store }: CounterProps) {
  const increment = store(state => state.increment);
  const decrement = store(state => state.decrement);
  const increment_async = store(state => state.increment_async);
  const increment_async_a = store(state => state.increment_async_a);
  const increment_async_b = store(state => state.increment_async_b);

  const loading = store(state => state.loading);
  const count = store(state => state.value);

  return (
    <div className="mx-2 my-1">
      <div className="flex items-center gap-2">
        <button onClick={() => increment()} {...stylex.props(button_styles.base)}>
          Increment
        </button>
        <span className="min-w-12 text-center">{count}</span>
        <button onClick={() => decrement()} {...stylex.props(button_styles.base)}>
          Decrement
        </button>
      </div>
      <div className="mt-2 flex flex-col items-start gap-1">
        <button onClick={() => increment_async()} {...stylex.props(button_styles.base)}>
          Async
        </button>
        <button onClick={() => increment_async_a()} {...stylex.props(button_styles.base)}>
          {loading ? 'Loading' : 'Async Fulfilled'}
        </button>
        <button onClick={() => increment_async_b()} {...stylex.props(button_styles.base)}>
          {loading ? 'Loading' : 'Async Rejected'}
        </button>
      </div>
    </div>
  );
}

interface CounterSlice {
  loading: boolean;
  value: number;
  increment: () => void;
  increment_by_amount: (value: number) => void;
  decrement: () => void;
  increment_async: () => Promise<void>;
  increment_async_a: () => Promise<void>;
  increment_async_b: () => Promise<void>;
}

const counter_slice: StateCreator<CounterSlice> = (set, get) => ({
  loading: false,
  value: 0,
  increment: () => set(state => ({ value: state.value + 1 })),
  decrement: () => set(state => ({ value: state.value - 1 })),
  increment_by_amount: (value: number) => set(state => ({ value: state.value + value })),
  increment_async: async () => {
    await new Promise(resolve => setTimeout(resolve, 500));
    get().increment_by_amount(2);
  },
  increment_async_a: async () => {
    set({ loading: true });
    get().increment_by_amount(1);

    await new Promise(resolve => setTimeout(resolve, 500));
    set(state => ({ loading: false, value: state.value + 1 }));
  },
  increment_async_b: async () => {
    set({ loading: true });
    get().increment_by_amount(1);

    try {
      await new Promise((_, reject) => setTimeout(() => reject(new Error('failed')), 500));
    } catch {
      set({ loading: false, value: -1 });
    }
  },
});

function create_store() {
  return create<CounterSlice>((...a) => ({
    ...counter_slice(...a),
  }));
}

type AppStore = ReturnType<typeof create_store>;
