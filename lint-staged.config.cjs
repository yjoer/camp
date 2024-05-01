module.exports = {
  '*.py': [
    'rye run ruff check',
    'rye run ruff check --select I',
    'rye run ruff format --check',
    'rye run mypy',
  ],
};
