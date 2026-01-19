/* eslint-disable import-x/no-extraneous-dependencies */
import stylex from '@stylexjs/postcss-plugin';
import autoprefixer from 'autoprefixer';

/** @type {import('postcss-load-config').Config} */
const config = {
  plugins: [
    stylex({
      include: ['_components/**/*.{ts,tsx}', 'src/**/*.{ts,tsx}'],
      useCSSLayers: true,
    }),
    autoprefixer,
  ],
};

export default config;
