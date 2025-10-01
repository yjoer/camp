import { useState } from 'react';
import { Provider } from 'react-redux';

import { makeStore } from '@/state/index';

import type { AppStore } from '@/state/index';

interface ReduxProviderProps {
  children: React.ReactNode;
}

function ReduxProvider({ children }: ReduxProviderProps) {
  const [store] = useState<AppStore>(makeStore());

  return <Provider store={store}>{children}</Provider>;
}

export default ReduxProvider;
