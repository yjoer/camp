import fnmatch
from pathlib import Path


def pytest_ignore_collect(collection_path: Path) -> bool | None:
    path = str(collection_path)

    if fnmatch.fnmatch(path, "**/solutions/**/*.py"):
        if path.endswith("_pipeline.py"):
            return False

        return True

    return None
