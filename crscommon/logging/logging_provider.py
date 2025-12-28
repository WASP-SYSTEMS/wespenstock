"""
Logging provider.

If you're having issues, try running "python3 -m crscommon.logging" to see if
the logging provider works in (relative) isolation.
"""

import datetime
import logging
import sys
import types
from pathlib import Path

from pydantic import BaseModel

from .settings import OUTPUT_DIR


class InitializedProviderState(BaseModel):
    """
    Parameters of the LoggingProvider that are available once init_logging() has been called.
    """

    log_dir: Path


class LoggingProvider:
    """
    Logging provider for the CRS.

    Loggers used in this CRS should be created using this class.
    """

    def __init__(self) -> None:
        self.file_name_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        self.formatter = logging.Formatter("[%(asctime)s.%(msecs)03d][%(levelname)s][%(name)s] %(message)s")
        self.formatter.datefmt = "%Y-%m-%d %H:%M:%S"

        # keep track of all loggers to add file handlers on initialization
        self.created_loggers: list[logging.Logger] = []

        self._initialized: InitializedProviderState | None = None

    def new_logger(self, name: str, hook_exception: bool = False, log_to_console: bool = True) -> logging.Logger:
        """
        Create a new logger. All logs written to this logger will appear in their own log file
        inside the configured log directory and will be printed to the console if not specified otherwise.
        The timestamps in the name of all log files created by the same instance of this class are guaranteed to match.

        name: logger name
        hook_exception: log exception to log file
        log_to_console: log to console
        """

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        logger.handlers.clear()

        if log_to_console:
            # add handler writing to console
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self.formatter)

            logger.addHandler(console_handler)

        if hook_exception:
            # https://stackoverflow.com/a/60523940
            def exc_handler(exctype: type[BaseException], value: BaseException, tb: types.TracebackType | None) -> None:
                logger.critical(value, exc_info=(exctype, value, tb))
                sys.__excepthook__(exctype, value, tb)

            sys.excepthook = exc_handler

        if self._initialized is not None:
            self._init_logger(logger)
            # This used not to be supported.
            logger.warning("new_logger() has been called *after* init_logging(), are you sure this is correct?")

        self.created_loggers.append(logger)

        return logger

    def _init_logger(self, logger: logging.Logger) -> None:
        """
        Initialize a single logger.
        """
        assert self._initialized is not None
        log_file_path = self._initialized.log_dir / f"{logger.name}_{self.file_name_tag}.log"
        handler = logging.FileHandler(log_file_path)
        handler.setFormatter(self.formatter)
        logger.addHandler(handler)

    def init_logging(self, log_dir: Path | None = None) -> None:
        """
        Initialize logging for all loggers created by the same instance of this class.

        This function essentially adds file handlers to all loggers.
        """

        # Do not call this more than once.
        if self._initialized is not None:
            raise RuntimeError("LoggingProvider is already initialized, this is a bug in the calling code!")

        # create log dir
        if log_dir is None:
            log_dir = OUTPUT_DIR.get()
        log_dir.mkdir(parents=True, exist_ok=True)

        self._initialized = InitializedProviderState(log_dir=log_dir)

        for logger in self.created_loggers:
            self._init_logger(logger)


LOGGING_PROVIDER = LoggingProvider()
