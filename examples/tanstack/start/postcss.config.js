// oxlint-disable import/no-default-export
import stylex from '@stylexjs/postcss-plugin';
import autoprefixer from 'autoprefixer';

/** @type {import('postcss-load-config').Config} */
const config = {
  plugins: [
    stylex({
      include: ['_components/**/*.{ts,tsx}', 'src/**/*.{ts,tsx}', 'ui/**/*.{ts,tsx}'],
      useCSSLayers: true,
    }),
    autoprefixer,
  ],
};

export default config;
