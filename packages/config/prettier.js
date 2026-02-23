/** @type {import("prettier").Config} */
export const prettier_config = {
  plugins: [
    // '@prettier/plugin-oxc',
    '@prettier/plugin-xml',
    'prettier-plugin-packagejson',
    // 'prettier-plugin-tailwindcss',
    // 'prettier-plugin-classnames',
    // 'prettier-plugin-merge',
    'prettier-plugin-java',
  ],
  bracketSameLine: true,
  printWidth: 100,
  quoteProps: 'as-needed',
  singleQuote: true,
  //
  xmlWhitespaceSensitivity: 'ignore',
  // tailwindFunctions: ['clsx', 'cva'],
  // customFunctions: ['clsx', 'cva'],
  // endingPosition: 'absolute',
};
