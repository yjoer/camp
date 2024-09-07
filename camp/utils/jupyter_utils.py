from IPython import get_ipython


def is_notebook():
    shell = get_ipython().__class__.__name__

    if shell == "ZMQInteractiveShell":
        return True
    else:
        return False
