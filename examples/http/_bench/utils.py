import subprocess
from collections.abc import Callable
from collections.abc import Generator
from contextlib import contextmanager

import numpy as np
import psutil


def stop_server(process: subprocess.Popen) -> None:
	parent = psutil.Process(process.pid)
	children = parent.children(recursive=True)
	children.append(parent)

	for p in children:
		p.terminate()

	psutil.wait_procs(children, timeout=5)


def remove_outliers(values: list, k: float = 1.5) -> list:
	arr = np.array(values)

	Q1, Q3 = np.percentile(arr, [25, 75])
	IQR = Q3 - Q1

	lower = Q1 - k * IQR
	upper = Q3 + k * IQR
	mask = (arr >= lower) & (arr <= upper)

	return arr[mask].tolist()


@contextmanager
def status() -> Generator[Callable[[str], None]]:
	prev_len = 0

	def update(text: str) -> None:
		nonlocal prev_len

		padding = " " * max(0, prev_len - len(text))
		print(f"{text}{padding}", end="\r", flush=True)
		prev_len = len(text)

	try:
		yield update
	finally:
		print(" " * prev_len, end="\r", flush=True)
