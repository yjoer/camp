module.exports = {
  '*.py': [
    'uv run ruff check',
    'uv run ruff check --select I',
    'uv run ruff format --check',
    'uv run mypy',
  ],
  '*.rs': [
    () => 'cargo clippy -- --deny warnings', //
    () => 'cargo fmt --check',
  ],
};
