/** @type {import("prettier").Config} */
module.exports = {
  plugins: [
    'prettier-plugin-packagejson',
    'prettier-plugin-tailwindcss', //
  ],
  bracketSameLine: true,
  printWidth: 100,
  quoteProps: 'consistent',
  singleQuote: true,
  tailwindFunctions: ['clsx'],
};
