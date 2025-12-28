"""Simple source code viewer with file cache."""

from __future__ import annotations

from pathlib import Path

from unidiff import PatchSet

from crs.logger import CRS_LOGGER

from .language_interface import LanguageInterface
from .lsp.exceptions import LspSymbolNotFound
from .lsp.lsp_types import Diagnostics
from .lsp.lsp_types import Location
from .symbol import LineInSymbolDefinition
from .symbol import SymbolDescription
from .symbol import SymbolOccurrence
from .util import format_symbol_identifier
from .util import occurrence_to_positions

log = CRS_LOGGER.getChild(__name__)


class SourceViewer:
    """
    View source files and extract symbols.
    """

    def __init__(
        self,
        li: LanguageInterface,
        file_extensions: list[str],
    ):
        """
        file_extension is needed to create a sample file to start indexing.
        """
        self.li = li
        self.file_extensions = file_extensions

    def get_symbol_definition(self, symbol_identifier: SymbolOccurrence | LineInSymbolDefinition) -> SymbolDescription:
        """
        Get symbol definition.
        """
        log.debug(f"Trying to find symbol via {format_symbol_identifier(symbol_identifier)}")
        if isinstance(symbol_identifier, SymbolOccurrence):
            return self.li.get_symbol_definition_from_occurrence(symbol_identifier)

        return self.li.get_symbol_definition_from_line_inside(symbol_identifier)

    def get_locations_referencing_symbol(self, occurrence: SymbolOccurrence) -> list[Location]:
        """
        Get all locations referencing specific symbol.
        """
        log.debug(f"Trying to find references to {format_symbol_identifier(occurrence)}")

        occur_positions = occurrence_to_positions(occurrence)

        references: list[Location] = []

        for pos in occur_positions:
            references.extend(self.li.get_references(occurrence.file, pos))

        return references

    def get_functions_referencing_symbol(self, occurrence: SymbolOccurrence) -> list[SymbolDescription]:
        """
        Get all function definitions that reference specific symbol.
        """
        references: list[Location] = self.get_locations_referencing_symbol(occurrence)

        functions: set[SymbolDescription] = set()

        for ref in references:
            try:
                # get the function enclosing the reference
                func = self.li.get_enclosing_function(ref)
                functions.add(func)
            except LspSymbolNotFound:
                log.error(f"Failed to find enclosing function of reference {ref}")

        return list(functions)

    def get_diagnostics(self) -> dict[Path, list[Diagnostics]]:
        """
        Get all diagnostics from workspace.
        Returned dict hast the following mapping: uri -> list[Diagnostics].
        NOTE: After changes have been applied in may take some time until the diagnostics are received.
        Therefore, the parameter delay, which just sleeps delay seconds until diagnostics are returned.
        """
        return self.li.get_diagnostics()

    def get_symbols_occurring_in_diff(
        self, patch: str, src_path: Path, functions_only: bool = False
    ) -> set[SymbolDescription]:
        """
        Get symbols that occur in the diff.
        """

        changed_symbols: set[SymbolDescription] = set()

        for diff in PatchSet(patch):
            if Path(diff.target_file).suffix not in self.file_extensions:
                continue
            for symbol in self.li.all_document_symbols(src_path / diff.path, functions_only=functions_only):
                # zero based lines
                symbol_start = symbol.location.range.start.line + 1
                symbol_end = symbol.location.range.end.line + 1
                for hunk in diff:
                    hunk_start = hunk.target_start
                    hunk_end = hunk.target_start + hunk.target_length
                    # check if the intervals (hunk_start, hunk_end) and (symbol_start, symbol_end) overlap
                    if hunk_start <= symbol_end and symbol_start <= hunk_end:
                        changed_symbols.add(symbol)

        return changed_symbols
