/** @type {import('lint-staged').Configuration} */
module.exports = {
  '*.{js,cjs,mjs,jsx,ts,tsx}': [
    'eslint --cache --cache-location node_modules/.cache/.eslintcache --max-warnings 0',
  ],
  '*.{jsx,tsx,css,scss}': [
    'stylelint --cache --cache-location node_modules/.cache/.stylelintcache --max-warnings 0',
  ],
  'package.json': [
    'sort-package-json --check', //
  ],
  '*.{js,cjs,mjs,jsx,ts,tsx,json,html,css,scss,md,mdx}': [
    'prettier --check', //
  ],
  '*.py': [
    'uv run ruff check',
    'uv run ruff check --select I',
    'uv run ruff format --check',
    'uv run mypy',
  ],
};
