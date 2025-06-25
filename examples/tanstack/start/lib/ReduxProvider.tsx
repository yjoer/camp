import { useRef } from 'react';
import { Provider } from 'react-redux';

import { makeStore } from '@/state/index';

import type { AppStore } from '@/state/index';

interface ReduxProviderProps {
  children: React.ReactNode;
}

function ReduxProvider({ children }: ReduxProviderProps) {
  const storeRef = useRef<AppStore>(null);

  if (!storeRef.current) {
    storeRef.current = makeStore();
  }

  return <Provider store={storeRef.current}>{children}</Provider>;
}

export default ReduxProvider;
