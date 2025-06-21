import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import App from './App';

// oxlint-disable-next-line prefer-query-selector
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
