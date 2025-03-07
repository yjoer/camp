/** @type {import("prettier").Config} */
module.exports = {
  plugins: [
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
  endingPosition: 'absolute-with-indent',
  experimentalOptimization: true,
};
