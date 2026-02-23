// oxlint-disable import/no-default-export
import type { Preview } from '@storybook/react-vite';

import './global.css';

const preview: Preview = {
  parameters: {
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/i,
      },
    },
  },
};

export default preview;
