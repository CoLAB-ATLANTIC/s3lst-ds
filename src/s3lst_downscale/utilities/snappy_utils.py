import os
from importlib import import_module
from types import ModuleType

from s3lst_downscale.utilities.logging_utils import RichLogger


def import_esa_snappy(logger: RichLogger | None = None) -> ModuleType:
    """
    Import `esa_snappy` package and suppress respective future warnings.

    NOTE: the import of `esa_snappy` retrieves Java logging `WARNING` and `INFO`
    messages. These are irrelevant and seem to be insuppressible. For more details read
    [this thread](https://forum.step.esa.int/t/snap-gpt-warning/43343/4).

    Parameters
    ----------
    logger : RichLogger or None, default=None
        A rich logger for showing the progress of the processing.

    Returns
    -------
    ModuleType
        The imported `esa_snappy` module.

    Raises
    ------
    ImportError
        If `esa_snappy` is not installed.
    """

    try:
        # ---> Import snappy package while suppressing Java logging messages
        # NOTE: the import of `esa_snappy` retrieves Java logging `WARNING` and `INFO`
        # messages. These are actually irrelevant. For more details read [this
        # thread](https://forum.step.esa.int/t/snap-gpt-warning/43343/4).

        # Open the operating system's null device
        # NOTE: Any output written to this file is discarded immediately.
        # NOTE: https://docs.python.org/3/library/os.html#os.open
        devnull = os.open(
            path=os.devnull,
            # Use write-only mode
            flags=os.O_WRONLY,
        )

        # Get duplicates of the current stdout (which has file descriptor fd=1) and
        # stderr (which has file descriptor fd=2) so they can be restored later.
        # NOTE: https://docs.python.org/3/library/os.html#os.dup
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)

        try:
            # Regard the null dev as stdout (1) and stderr (2) so that any output
            # written to them is discarded.
            # NOTE: https://docs.python.org/3/library/os.html#os.dup2
            os.dup2(fd=devnull, fd2=1)
            os.dup2(fd=devnull, fd2=2)

            # Import snappy package
            esa_snappy = import_module("esa_snappy")

        finally:
            # Restore stdout and stderr
            # NOTE: https://docs.python.org/3/library/os.html#os.dup2
            os.dup2(fd=old_stdout, fd2=1)
            os.dup2(fd=old_stderr, fd2=2)

            # Close the the duplicates and the null device
            os.close(old_stdout)
            os.close(old_stderr)
            os.close(devnull)

        # ---> Supress snappy warnings by setting minimum logging level to SEVERE
        # instead of INFO
        LogManager = esa_snappy.jpy.get_type("java.util.logging.LogManager")
        Level = esa_snappy.jpy.get_type("java.util.logging.Level")
        snap_logger = LogManager.getLogManager().getLogger("org.esa.snap")
        snap_logger.setLevel(Level.SEVERE)

        # ---> Suppress snappy printed messages
        System = esa_snappy.jpy.get_type("java.lang.System")
        PrintStream = esa_snappy.jpy.get_type("java.io.PrintStream")
        NullStream = esa_snappy.jpy.get_type(
            "org.apache.commons.io.output.NullOutputStream"
        )
        System.setOut(PrintStream(NullStream()))
        System.setErr(PrintStream(NullStream()))

        return esa_snappy

    except ImportError as e:
        if logger is not None:
            logger.error(
                "[bold red]Error importing `esa_snappy` package."
                + f"\nError message: {e}"
                + "\nRequested utilities from the `esa_snappy` package are unavailable"
                + " because the package is not installed in the uv environment."
                + "\nRun `source scripts/install/install_snap.sh` to install it."
                + "[/bold red]",
            )
        raise ImportError(
            "Error importing `esa_snappy` package."
            + f"\nError message: {e}"
            + "\nRequested utilities from the `esa_snappy` package are unavailable"
            + " because the package is not installed in the uv environment."
            + "\nRun `source scripts/install/install_snap.sh` to install it."
        )
