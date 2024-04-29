module.exports = {
  '*.py': [
    'rye run ruff check', //
    'rye run ruff format --check',
    'rye run mypy',
  ],
};
