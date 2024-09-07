from IPython import get_ipython


def is_notebook():
    shell = get_ipython().__class__.__name__

    if shell == "ZMQInteractiveShell":
        return True
    else:
        return False


class StopExecutionException(Exception):
    def _render_traceback_(self):
        pass


def stop_execution():
    """
    While raising an exception can conditionally stop the execution of a notebook cell,
    it also halts the execution of programs run through the terminal.
    """
    raise StopExecutionException()
