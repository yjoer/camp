/** @type {import("prettier").Config} */
export default {
  plugins: [
    // '@prettier/plugin-oxc',
    'prettier-plugin-packagejson',
    'prettier-plugin-tailwindcss',
    'prettier-plugin-classnames',
    'prettier-plugin-merge',
  ],
  bracketSameLine: true,
  printWidth: 100,
  quoteProps: 'consistent',
  singleQuote: true,
  tailwindFunctions: ['clsx', 'cva'],
  customFunctions: ['clsx', 'cva'],
  endingPosition: 'absolute',
};
