from IPython import get_ipython


def is_notebook() -> bool:
  shell = get_ipython().__class__.__name__
  return shell == "ZMQInteractiveShell"


class StopExecutionError(Exception):
  def _render_traceback_(self) -> None:
    pass


def stop_execution() -> None:
  # while raising an exception can conditionally stop the execution of a
  # notebook cell, it also halts the execution of programs run through the
  # terminal.
  raise StopExecutionError
