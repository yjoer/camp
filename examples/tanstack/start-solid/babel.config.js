export default function babel(api) {
  api.cache(true);

  return {
    plugins: [
      ['@stylexjs/babel-plugin', {
        debug: process.env.NODE_ENV === 'development',
        unstable_moduleResolution: { type: 'commonJS' },
      }],
    ],
    parserOpts: {
      plugins: ['jsx', 'typescript'],
    },
  };
}
