/** @type {import('lint-staged').Configuration} */
export const lint_staged_config = {
  '*.{js,cjs,mjs,jsx,ts,tsx}': [
    'eslint --cache --cache-location node_modules/.cache/.eslintcache --max-warnings 0',
  ],
  '*.{jsx,tsx,css,scss}': [
    'stylelint --cache --cache-location node_modules/.cache/.stylelintcache --max-warnings 0',
  ],
  'package.json': [
    'sort-package-json --check',
  ],
  '*.{css,scss,html,json,md,mdx,yml,yaml}': [
    'prettier --check',
  ],
  '*.py': [
    'uv run ruff check',
    'uv run ruff check --select I',
    // 'uv run ruff format --check',
    // 'uv run mypy',
  ],
};
