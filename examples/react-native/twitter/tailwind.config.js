/* eslint-disable unicorn/prefer-module */
// oxlint-disable no-require-imports
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './App.tsx', //
    './components/**/*.{js,jsx,ts,tsx}',
  ],
  presets: [
    require('nativewind/preset'), //
  ],
  theme: {
    extend: {},
  },
  plugins: [],
};
