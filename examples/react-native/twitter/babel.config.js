/* eslint-disable unicorn/prefer-module */
// oxlint-disable no-anonymous-default-export
module.exports = function (api) {
  api.cache(true);
  return {
    presets: [
      ['babel-preset-expo', { jsxImportSource: 'nativewind' }], //
      'nativewind/babel',
    ],
  };
};
