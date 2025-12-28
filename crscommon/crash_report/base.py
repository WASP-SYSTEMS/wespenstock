"""Base classes for language utils."""

from abc import ABC
from abc import abstractmethod
from pathlib import Path

from pydantic import BaseModel

from crs.base.context import CrsContext
from crs.logger import CRS_LOGGER
from crscommon.editor.editor import SourceEditor
from crscommon.editor.symbol import LineInSymbolDefinition
from crscommon.editor.viewer import SymbolDescription

log = CRS_LOGGER.getChild(__name__)


class StackLevel(BaseModel):
    """Stack Level."""

    function: str
    file: Path
    line: int
    additional_info: str | None = None


class BaseStackTrace(BaseModel):
    """
    Parse stacktrace into array, where the element
    with index 0 represents stack level 0 and so on.
    This behaviour must be ensured by derived classes.
    """

    trace: list[StackLevel] = []

    def add_level(self, lvl: StackLevel) -> None:
        """Add stack level."""
        self.trace.append(lvl)

    def __getitem__(self, stack_level: int) -> StackLevel:
        return self.trace[stack_level]


class BaseCrashReport(ABC):
    """Base class for crash reports like ASAN Reports"""

    def __init__(self, report: str, ctx: CrsContext) -> None:
        self.report = self.extract_report(report)
        self.src_path_abs: Path = ctx.cp_path_abs / ctx.src_path_rel
        self.ctx = ctx

    @abstractmethod
    def get_stacktrace(self) -> BaseStackTrace:
        """Get stacktrace from the crash report."""

    @staticmethod
    @abstractmethod
    def extract_report(report: str) -> str:
        """Extract actual report from input"""

    def __repr__(self) -> str:
        return self.report

    def get_symbol_descriptions(self, editor: SourceEditor) -> list[SymbolDescription]:
        """Get symbol description for every function in stack trace."""
        # pylint: disable=W0718
        # Catching too general exception Exception

        log.info("Getting symbols from stack trace")
        try:
            stacktrace = self.get_stacktrace()
        except Exception:
            return []
        symbol_list: list[SymbolDescription] = []

        # get symbol description for every function in stack trace
        for lvl in stacktrace.trace:
            log.info(f"Getting symbol `{lvl.function}` from {lvl.file}")

            symbol = editor.get_symbol_definition(
                # set line=lvl.line - 1, because lang server uses 0-based lines
                LineInSymbolDefinition(name=lvl.function, file=lvl.file.absolute(), line=lvl.line - 1)
            )
            symbol_list.append(symbol)

        return symbol_list
