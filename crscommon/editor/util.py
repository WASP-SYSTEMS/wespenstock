"""Convenience functions."""

from crs.base.util import find_all
from crs.logger import CRS_LOGGER

from .lsp.exceptions import LspSymbolNotFound
from .lsp.lsp_types import Position
from .symbol import LineInSymbolDefinition
from .symbol import SymbolOccurrence

log = CRS_LOGGER.getChild(__name__)


def get_lines(lines: list[str], start: int, end: int) -> str:
    """
    Extract lines of code from start to end from file. Lines are zero-based.
    """

    return "\n".join(lines[start : end + 1]) + "\n"


def format_symbol_identifier(symbol_identifier: SymbolOccurrence | LineInSymbolDefinition) -> str:
    """Format symbol identifier for better logging."""

    return (
        f"{symbol_identifier.__class__.__name__}: {symbol_identifier.name}, "
        f"{symbol_identifier.file}:{symbol_identifier.line+1}"
    )


def occurrence_to_positions(occurrence: SymbolOccurrence) -> list[Position]:
    """
    Convert an occurrence to a list of positions
    (the identifier may occur more than once on the line).
    Trows LspSymbolNotFound if no positions could be determined.
    """

    file_content = occurrence.file.read_text().splitlines()

    positions: list[Position] = []

    for identifier_start in find_all(file_content[occurrence.line], occurrence.name, overlapping=True):
        positions.append(Position(line=occurrence.line, character=identifier_start + 1))

    if positions:
        return positions

    err = f"Could not find symbol in line: {file_content[occurrence.line]}; {format_symbol_identifier(occurrence)}"

    log.warning(err)
    raise LspSymbolNotFound(err)
