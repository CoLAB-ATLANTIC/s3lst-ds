def is_jupyter() -> bool:
    """
    Check if currently running in a Jupyter notebook.

    Returns
    -------

    bool
        `True` if running in a Jupyter notebook, `False` otherwise.

    This functions was extracted from a [StackOverflow
    answer](https://stackoverflow.com/a/39662359/4382986).
    """
    try:
        shell = get_ipython().__class__.__name__  # type: ignore
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter
