import logging
from pathlib import Path
from typing import ClassVar, Literal

from rich import get_console
from rich.console import Console, RenderableType
from rich.logging import RichHandler
from rich.text import Text

from s3lst_ds.utilities.iter_utils import union


class RichLogger:
    """
    A `logging.Logger`-based class that uses a Rich console and that may also write to
    file.

    Parameters
    ----------

    name : str, default="root"
        Name of the base logger. If a name was not issued, the base root logger (of name
        `"root"`) is considered.

    level : {50, 40, 30, 20, 10, 0}
        Minimum [logging
        level](https://docs.python.org/3/library/logging.html#logging-levels) of the
        base logger. Defaults to the level of root (`logging.WARNING` (`30`) if the
        level of root was not changed) in case of base logger creation or to the level
        of an already existing base logger in case of reuse.

    log_mode : {None, "console", "file", "both"}, default="both"
        Mode for logging:
        - `None`, no logging is done.
        - `"console"`, only Rich console logging is considered.
        - `"file"`, only file logging is considered.
        - `"both"`, both Rich console and file logging are considered.

    rich_handler : RichHandler or None
        The Rich console logging handler used by the base logger. If `None`, no Rich
        console logging was set.

    file_handlers : dict[Path, logging.FileHandler]
        File logging handlers used by the base logger if any, keyed by by the paths to
        the respective files (`file_paths`). If empty, no file logging was set.

    file_paths : list[Path]
        Paths to the logging files (regardless of the handlers being set or not).

    file_modes : dict[Path, str]
        Modes for file logging (regardless of the handlers being set or not) (e.g. `"w"`
        for writing (overwriting) and `"a"` for appending) keyed by the paths to the
        respective files (`file_paths`).

    base_logger : logging.Logger
        The base logger.

    base_warnings_logger : logging.Logger
        A logger for the warnings of the warning package. This logger shares the same
        handlers as the base logger.

    console: Console
        The Rich console (regardless of the handler being set or not).

    Attributes
    ----------
    level_mapper : dict[int or str or None, int or None]
        Class-level mapping between logging level aliases and their corresponding
        numeric levels. Numeric levels are mapped to themselves, logging level names are
        mapped to their corresponding numeric levels, and `None` is mapped to `None`.
    """

    # Mapper between logging level aliases and their corresponding values
    level_mapper: ClassVar[dict[int | str | None, int | None]] = {
        0: 0,
        10: 10,
        20: 20,
        30: 30,
        40: 40,
        50: 50,
        "notset": 0,
        "debug": 10,
        "info": 20,
        "warning": 30,
        "error": 40,
        "critical": 50,
        None: None,
    }

    def __init__(
        self,
        name: str | None = None,
        level: Literal[
            50,
            40,
            30,
            20,
            10,
            0,
            "critical",
            "error",
            "warning",
            "info",
            "debug",
            "notset",
        ]
        | None = None,
        console: Console | None = None,
        file_path: Path | None = None,
        file_mode: str = "w",
        log_mode: Literal["console", "file", "both"] | None = "both",
    ) -> None:
        """
        Initialize instance by getting a base logger with the issued `name`, setting its
        minimum logging level to `level`, including a Rich logging handler using the
        issued `console` as well as a file logging handler that writes to file at
        `file_path` considering the issued `file_mode`. Note that Rich console logging
        is only considered if `log_mode` corresponds to `"console"` or `"both"`.
        Furthermore, file logging is only considered if `log_mode` corresponds to
        `"file"` or `"both"`.

        Parameters
        ----------

        name : str or None, default=None
            Name of the base logger. If issued and a base logger with such name already
            exists, it will be used and updated instead of recreated from scratch. If
            not issued, the base root logger (called `"root"`) is used. Note that the
            base root logger always exists.

        level : {50, 40, 30, 20, 10, 0, "critical", "error", "warning", "info", "debug",
            "notset", None}, default=None
            Minimum [logging
            level](https://docs.python.org/3/library/logging.html#logging-levels) (e.g.
            `logging.INFO` (`20`)) to set. Regardless of whether the base logger already
            exists or not, if a level is issued, it will be set in the base logger. If
            not issued, the level of the base logger will be kept. Note that the smaller
            the logging level, the larger the logged information (which will further
            include less important one). Note that if `level` is not issued, the level
            of the base logger defaults to the one of root (`logging.WARNING` (`30`) if
            the level of root was not changed) in case of base logger creation or to the
            level of an already existing base logger in case of reuse. The logging
            levels are as follows (from the least to the most important):

            - `logging.NOTSET`, `0`, `"notset"`;
            - `logging.DEBUG`, `10`, `"debug"`;
            - `logging.INFO`, `20`, `"info"`;
            - `logging.WARNING`, `30`, `"warning"`;
            - `logging.ERROR`, `40`, `"error"`;
            - `logging.CRITICAL`, `50`, `"critical"`.

        console: Console or None, default=None
            A Rich console object to use for console logging. Regardless of whether the
            base logger already exists or not, if a console is issued, it will be set in
            the base logger. If not issued and the base logger already exists and
            contains a Rich console, this latter will be kept. If it does not contain, a
            new Rich console is set. Note that if the base logger contains multiple Rich
            console handlers, only the first one is considered. Also, note that a Rich
            console handler is only considered if `log_mode` corresponds to `"console"`
            or `"both"`.

        file_path : Path or None, default=None
            Path to file for logging. If not issued, no new file logging is set up. Note
            that file logging is solely considered if `log_mode` corresponds to `"file"`
            or `"both"`.

        file_mode : str, default="w"
            Mode for file logging (e.g. `"w"` for writing (overwriting) and `"a"` for
            appending). Only effective if `file_path` is issued.

        log_mode : {None, "console", "file", "both"}, default="both"
            Mode for logging:
            - `None`, no logging is done.
            - `"console"`, only Rich console logging is considered.
            - `"file"`, only file logging is considered.
            - `"both"`, both Rich console and file logging are considered.
        """

        # Get base logger
        # NOTE: if `name` is `None`, the base root logger is returned.
        self.base_logger = logging.getLogger(name)

        # Redirect warnings of the warning package to another logger.
        # NOTE: downstream methods would make this and the base logger share the same
        # handlers.
        logging.captureWarnings(True)
        self.base_warnings_logger = logging.getLogger("py.warnings")

        # Set minimum logging level of the base logger if a level was issued
        self.level = level

        # Initialize file paths and modes for logging
        self._file_paths = [file_path] if file_path is not None else []
        self._file_modes = {file_path: file_mode} if file_path is not None else {}

        # Define Rich console
        self._set_console(console)

        # Set logging mode
        # NOTE: this will also set the Rich console and file logging handlers, depending
        # on the value of `log_mode`.
        self.log_mode = log_mode

        # Stop propagation to the base root logger to avoid double logging
        self.stop_propagation()

    @property
    def name(self) -> str:
        """
        Get the name of the base logger.

        Returns
        -------
        name : str
            The name of the base logger.
        """
        return self.base_logger.name

    @property
    def level(self) -> Literal[50, 40, 30, 20, 10, 0]:
        """
        Get the minimum [logging
        level](https://docs.python.org/3/library/logging.html#logging-levels) of the
        base logger.

        Returns
        -------
        level : {50, 40, 30, 20, 10, 0}
            The minimum logging level of the base logger.
        """
        return self.base_logger.level  # type: ignore

    @property
    def console(self) -> Console:
        """
        Get the Rich console used by the base logger.

        Returns
        -------
        console : Console or None
            The Rich console used by the base logger, if it exists.
        """
        return self._console  # type: ignore

    @property
    def log_mode(self) -> Literal["console", "file", "both"] | None:
        """
        Get the logging mode of the logger.

        Returns
        -------
        log_mode : {None, "console", "file", "both"}
            Mode for logging:
            - `None`, no logging is done.
            - `"console"`, only Rich console logging is considered.
            - `"file"`, only file logging is considered.
            - `"both"`, both Rich console and file logging are considered.
        """
        return self._log_mode  # type: ignore

    @property
    def rich_handler(self) -> RichHandler | None:
        """
        Get the Rich logging handler of the base logger, if it exists.

        Returns
        -------
        rich_handler : RichHandler or None
            The Rich logging handler of the base logger, if it exists.
        """

        rich_handler = next(
            (
                handler
                for handler in self.base_logger.handlers
                if isinstance(handler, RichHandler)
            ),
            None,
        )

        return rich_handler

    @property
    def file_handlers(self) -> dict[Path, logging.FileHandler]:
        """
        Get file logging handlers of the base logger if they exist.

        Returns
        -------
        file_handlers : dict[Path, logging.FileHandler]
            File logging handlers of the base logger if they exist, keyed by their file
            paths. If they do not exist, an empty dict is returned.
        """
        file_handlers = {
            Path(handler.baseFilename): handler
            for handler in self.base_logger.handlers
            if isinstance(handler, logging.FileHandler)
        }

        return file_handlers

    @property
    def file_paths(self) -> list[Path]:
        """
        Get the file logging paths.

        Returns
        -------
        file_paths : list[Path]
            Paths to the logging files.
        """
        return union(self._file_paths, self.get_file_handler_paths())  # type: ignore

    @property
    def file_modes(self) -> dict[Path, str]:
        """
        Get the file logging modes.

        Returns
        -------
        file_modes : dict[Path, str]
            Modes for file logging (e.g. `"w"` for writing (overwriting) and `"a"` for
            appending) keyed by the paths to the respective files (`file_paths`).
        """
        return self._file_modes | self.get_file_handler_modes()

    def __getstate__(self) -> dict:
        """
        Get dictionary storing instance's picklable parameters (all except `console`).

        Returns
        -------
        state: dict
            Dictionary with instance's picklable parameters (all except `console`).
        """
        state = self.__dict__.copy()

        # Drop unpicklable parameters
        state.pop("_console", None)

        return state

    def __setstate__(self, state: dict) -> None:
        """
        Set instance's picklable parameters from the issued `state` dictionary and
        recreate unpicklable parameters (console).

        Parameters
        ----------
        state : dict
            Dictionary with instance's picklable parameters (all except `console`).

        """
        # Update dictionary storing instance's writable parameters with the values from
        # `state`.
        self.__dict__.update(state)

        # Recreate the unpicklable console
        # NOTE: if a Rich logging handler exists, its console will be considered. If
        # there is no Rich logging handler, a new console will be considered instead.
        self._set_console(None)

    def get_file_handler_paths(self) -> list[Path]:
        """
        Get the paths of file logging handlers of the base logger if they exist.

        Returns
        -------
        file_handler_paths : list[Path]
            Paths of file logging handlers of the base logger if they exist. If they do
            not exist, an empty list is returned.
        """
        file_handler_paths = list(self.file_handlers.keys())
        return file_handler_paths

    def get_file_handler_modes(self) -> dict[Path, str]:
        """
        Get the modes of file logging handlers of the base logger if they exist, keyed
        by their file paths.

        Returns
        -------
        file_handler_modes : dict[Path, str]
            Modes of file logging handlers of the base logger if they exist, keyed by
            their file paths. If they do not exist, an empty dict is returned.
        """
        file_handler_modes = {
            Path(handler.baseFilename): handler.mode
            for handler in self.file_handlers.values()
        }

        return file_handler_modes

    @level.setter
    def level(
        self,
        level: Literal[
            50,
            40,
            30,
            20,
            10,
            0,
            "critical",
            "error",
            "warning",
            "info",
            "debug",
            "notset",
        ]
        | None,
    ) -> None:
        """
        Set the minimum [logging
        level](https://docs.python.org/3/library/logging.html#logging-levels) of the
        base logger to `level` if issued.

        Parameters
        ----------
        level : {50, 40, 30, 20, 10, 0, "critical", "error", "warning", "info", "debug",
            "notset", None}
            The minimum logging level to set in the base logger. Note that an integer,
            string or `None` can be issued. The logging levels are as follows (from the
            least to the most important):

            - `logging.NOTSET`, `0`, `"notset"`;
            - `logging.DEBUG`, `10`, `"debug"`;
            - `logging.INFO`, `20`, `"info"`;
            - `logging.WARNING`, `30`, `"warning"`;
            - `logging.ERROR`, `40`, `"error"`;
            - `logging.CRITICAL`, `50`, `"critical"`.
        """
        self.base_logger.setLevel(self.level_mapper[level])  # type: ignore

    @log_mode.setter
    def log_mode(self, log_mode: Literal["console", "file", "both"] | None) -> None:
        """
        Set the logging mode of the logger to `log_mode`. This will further set the Rich
        console and file logging handlers, depending on the argument value.

        Parameters
        ----------
        log_mode : {None, "console", "file", "both"}
            Mode for logging:
            - `None`, no logging is done.
            - `"console"`, only Rich console logging is considered.
            - `"file"`, only file logging is considered.
            - `"both"`, both Rich console and file logging are considered.
        """
        self._log_mode = log_mode

        match log_mode:
            case None:
                # Drop Rich and file logging handlers
                self.drop_all_handlers()
                self.console.quiet = True

            case "console":
                # Set Rich logging handler and drop all file logging handlers
                self.set_rich_handler(self.console)
                self.drop_all_file_handlers()
                self.console.quiet = False

            case "file":
                # Set file logging handler and drop Rich logging handler if it exists
                self.set_file_handler(
                    file_paths=self.file_paths,
                    file_modes=self.file_modes,
                )
                self.drop_all_rich_handlers()
                self.console.quiet = True

            case "both":
                # Set Rich and file logging handlers
                self.set_rich_handler(self.console)
                self.set_file_handler(
                    file_paths=self.file_paths,
                    file_modes=self.file_modes,
                )
                self.console.quiet = False

    def _set_console(self, console: Console | None) -> None:
        """
        Set the Rich console. If not issued and a Rich logging handler exists, its
        console will be used. If not issued and there is no Rich logging handler, a new
        console will be used.

        Parameters
        ----------
        console : Console or None
            The Rich console.
        """
        self._console = (
            get_terminal_console(use_global=False)
            if console is None and self.rich_handler is None
            else (
                self.rich_handler.console
                if console is None and self.rich_handler is not None
                else console
            )
        )

    def get_base_logger(self) -> logging.Logger:
        """
        Get the base logger.

        Returns
        -------
        base_logger : logging.Logger
            The base logger.
        """
        return self.base_logger

    def has_handler(self, name: str) -> bool:
        """
        Check if the base logger has a logging handler with the issued `name`.

        Parameters
        ----------
        name : str
            The name of the handler to check for.
        Returns
        -------
        has_handler : bool
            Whether the base logger has a handler with the issued name.
        """
        return any(
            getattr(handler, "name", None) == name
            for handler in self.base_logger.handlers
        )

    def drop_handler(self, name: str) -> None:
        """
        Drop the logging handler with the issued `name` from the base logger if it
        exists.

        Parameters
        ----------
        name : str
            The name of the handler to drop.
        """
        for handler in self.base_logger.handlers:
            if getattr(handler, "name", None) == name:
                self.base_logger.removeHandler(handler)
                self.base_warnings_logger.removeHandler(handler)
                handler.close()

    def drop_other_rich_handlers(self) -> None:
        """
        Drop all Rich logging handlers from the base logger except the first one.
        """
        rich_handlers = [
            handler
            for handler in self.base_logger.handlers
            if isinstance(handler, RichHandler)
        ]
        for handler in rich_handlers[1:]:
            self.base_logger.removeHandler(handler)
            self.base_warnings_logger.removeHandler(handler)
            handler.close()

    def drop_all_rich_handlers(self) -> None:
        """
        Drop all Rich logging handlers from the base logger if they exist.
        """
        for handler in self.base_logger.handlers:
            if isinstance(handler, RichHandler):
                self.base_logger.removeHandler(handler)
                self.base_warnings_logger.removeHandler(handler)
                handler.close()

    def drop_all_file_handlers(self) -> None:
        """
        Drop all file logging handlers from the base logger if they exist.
        """
        for handler in self.base_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                self.base_logger.removeHandler(handler)
                self.base_warnings_logger.removeHandler(handler)
                handler.close()

    def drop_all_handlers(self) -> None:
        """
        Drop all logging handlers from the base logger if they exist.
        """
        for handler in self.base_logger.handlers:
            self.base_logger.removeHandler(handler)
            self.base_warnings_logger.removeHandler(handler)
            handler.close()

    def add_handler(self, handler: logging.Handler) -> None:
        """
        Add the issued logging `handler` to the base logger by firstly dropping any
        existing handler with the same name (if `handler` has a name).

        Parameters
        ----------

        handler : logging.Handler
            The handler to add.
        """

        # Remove existing handler if there is one with the same name as the issued
        # handler
        if getattr(handler, "name", None) is not None:
            self.drop_handler(handler.name)  # type: ignore

        # Add logging handler
        self.base_logger.addHandler(handler)
        self.base_warnings_logger.addHandler(handler)

    def set_rich_handler(self, console: Console | None = None) -> None:
        """

        Set the issued Rich console in the base logger by removing the respective Rich
        logging handler if it exists, and creating a new one from scratch. If not
        issued, the Rich logging handler will not be changed if it exists, or a new
        console will be used and the handler created with it if it does not exist. Note
        that a Rich console handler is only considered if `log_mode` corresponds to
        `"console"` or `"both"`.

        Parameters
        ----------
        console : Console or None, default=None
            The Rich console to use in the logging handler.
        """

        # Set console
        self._set_console(console)

        # Set Rich logging handler if a console was issued or if the base logger does
        # not have any Rich logging handler. Note that Rich console logging is only
        # considered if `log_mode` corresponds to `"console"` or `"both"`.
        if (console is not None or self.rich_handler is None) and self.log_mode in [
            "console",
            "both",
        ]:
            self.drop_all_rich_handlers()
            rich_handler = create_rich_handler(self._console)  # type: ignore
            self.add_handler(rich_handler)

    def set_file_handler_single(
        self, file_path: Path | None, file_mode: str = "w"
    ) -> None:
        """
        Set a file logging handler in the base logger that writes to file at `file_path`
        considering the issued `file_mode`. Further create parent folders for the log
        file if they do not exist, and set `file_path.as_posix()` as the name of the
        handler. Note that file logging is solely considered if `log_mode` corresponds
        to `"file"` or `"both"`.

        Parameters
        ----------
        file_path : Path or None
            Path to file for logging. If not issued, no new file logging is set up. Note
            that file logging is solely considered if `log_mode` corresponds to `"file"`
            or `"both"`. Also note that if set, the file logging handler will be named
            `file_path.as_posix()`.

        file_mode : str, default="w"
            Mode for file logging (e.g. `"w"` for writing (overwriting) and `"a"` for
            appending). Only effective if `file_path` is issued.
        """
        if file_path is not None:
            # Set file logging handler
            if self.log_mode in ["file", "both"]:
                file_handler = create_file_handler(
                    file_path=file_path,
                    file_mode=file_mode,
                )
                self.add_handler(file_handler)

            # Update file paths and modes for logging
            self._file_paths = union(self._file_paths, [file_path])  # type: ignore
            self._file_modes[file_path] = file_mode

    def set_file_handler(
        self, file_paths: Path | list[Path] | None, file_modes: str | dict[Path, str]
    ) -> None:
        """
        Set file logging handlers in the base logger that writes to files at
        `file_paths` considering the issued `file_modes`. Further create parent folders
        for each `file_path` in `file_paths` and set `file_path.as_posix()` as the name
        of the respective handler. Note that file logging is solely considered if
        `log_mode` corresponds to `"file"` or `"both"`.

        Parameters
        ----------
        file_paths : Path or list[Path] or None
            Paths to files for logging. If not issued, no new file logging is set up.
            Note that file logging is solely considered if `log_mode` corresponds to
            `"file"` or `"both"`. Also note that if set, the file logging handlers will
            be named `file_path.as_posix()` for each `file_path` in `file_paths`.
        file_modes : str or dict[Path, str]
            Modes for file logging (e.g. `"w"` for writing (overwriting) and `"a"` for
            appending). In the case of multiple files, if `file_modes` is a string, it
            is applied to all files. If it is a dictionary, each value is applied to the
            corresponding file. Only effective if `file_paths` is issued.
        """
        for file_path in file_paths if isinstance(file_paths, list) else [file_paths]:
            self.set_file_handler_single(
                file_path=file_path,
                file_mode=(
                    file_modes if isinstance(file_modes, str) else file_modes[file_path]  # type: ignore
                ),
            )

    def stop_propagation(self) -> None:
        """
        Stop information propagation to the base root logger to avoid double logging.
        """
        if self.base_logger.propagate is True:
            self.base_logger.propagate = False
        if self.base_warnings_logger.propagate is True:
            self.base_warnings_logger.propagate = False

    # Wrapper methods for base logger's convenient methods
    # NOTE:
    # https://github.com/python/cpython/blob/2faceeec5c0fb06498a9654d429180ac4610c65a/Lib/logging/__init__.py#L1501
    def debug(self, msg, *args, **kwargs):
        """
        Wrapper for base logger debug method.
        """
        self.base_logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """
        Wrapper for base logger info method.
        """
        self.base_logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """
        Wrapper for base logger warning method.
        """
        self.base_logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """
        Wrapper for base logger error method.
        """
        self.base_logger.error(msg, *args, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        """
        Wrapper for base logger exception method.
        """
        self.base_logger.exception(msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """
        Wrapper for base logger critical method.
        """
        self.base_logger.critical(msg, *args, **kwargs)

    def fatal(self, msg, *args, **kwargs):
        """
        Wrapper for base logger fatal method.
        """
        self.base_logger.fatal(msg, *args, **kwargs)


def create_rich_handler(
    console: Console,
) -> RichHandler:
    """
    Create a customized logging `RichHandler` with the issued `console`. Further set the
    name of the handler to `"rich"`.

    Parameters
    ----------
    console : Console
        The Rich `Console` to use in the logging handler.

    Returns
    -------
    rich_handler : RichHandler
        The created Rich logging handler.
    """
    # Intialize rich handler
    rich_handler = RichHandler(
        console=console,
        # Whether to show current time prefixed to the core log output.
        show_time=False,
        # Whether to show path to the original log call suffixed to the core log
        # output.
        show_path=False,
        # Whether to enable markup in the core log output.
        # NOTE: Herein, markup corresponds to inline formatting tags (e.g. `"[bold
        # red]"`) inside log output that get rendered in with styles/colors.
        markup=True,
    )

    # Format the core log output to include solely the message.
    rich_handler.setFormatter(logging.Formatter("%(message)s"))

    # Set name of the handler
    rich_handler.name = "rich"

    return rich_handler


def create_file_handler(file_path: Path, file_mode: str = "w") -> logging.FileHandler:
    """
    Create a logging `FileHandler` with the issued `file_path` and `file_mode`. Further
    create parent folders for the log file if they do not exist, and set
    `file_path.as_posix()` as the name of the handler.

    Parameters
    ----------
    file_path : Path or None
        Path to file for logging.

    file_mode : str, default="w"
        Mode for file logging (e.g. `"w"` for writing (overwriting) and `"a"` for
        appending). Only effective if `file_path` is issued.

    Returns
    -------
    file_handler : logging.FileHandler
        The created file logging handler.
    """

    # Create log file output directory if it does not exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize file logging handler
    file_handler = logging.FileHandler(filename=file_path, mode=file_mode)
    # Define handler formater
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )

    # Set name of the handler
    file_handler.name = file_path.as_posix()

    return file_handler


def get_terminal_console(use_global: bool = False) -> Console:
    """
    Get Rich `Console` object with forced terminal mode (when possible) for better
    logging and display in both terminal and Jupyter notebook environments. The console
    may be either a global one or a new one depending on the issued `use_global`
    argument. In the case of the former, direct enforcing of the terminal mode cannot be
    done, but argument `is_jupyter` is set to `False` as in case of the latter.

    Parameters
    ----------
    use_global : bool, default=False
        Whether to use a global console or to create a new one.

    Returns
    -------
    console : Console
        A Rich `Console`.
    """

    # NOTE:: `get_console()` defines a global console on first call. Subsequent
    # `get_console()` calls would then use this same console.
    console = Console(force_terminal=True) if use_global is False else get_console()
    console.is_jupyter = False

    return console


def get_rich_text_from_renderable(console: Console, renderable: RenderableType) -> Text:
    """
    Capture the `console` output of the `renderable` object (without printing it to
    terminal) and return it as a `Text` object.

    For more details, see
    https://github.com/Textualize/rich/discussions/1799#discussioncomment-1994605.

    Parameters
    ----------
    console : Console
        The Rich console to use for capturing the renderable object output.

    renderable : RenderableType
        The Rich renderable object output to capture.

    Returns
    -------
    text : Text
        The captured renderable object output as a Rich `Text` object.
    """
    # Unquiet the console to capture the renderable object output if the console was
    # quieted
    quiet = console.quiet
    if quiet is True:
        console.quiet = False

    # Capture the console output of the renderable object without printing it to
    # terminal
    with console.capture() as capture:
        console.print(renderable)

    # Quiet the console if it was initially quit
    if quiet is True:
        console.quiet = True

    return Text.from_ansi(capture.get())
