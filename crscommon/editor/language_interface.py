"""Interface for source interaction. Must be implemented for each language."""

from pathlib import Path
from typing import Iterable
from typing import Protocol

from .lsp.lsp_types import Diagnostics
from .lsp.lsp_types import Location
from .lsp.lsp_types import Position
from .symbol import LineInSymbolDefinition
from .symbol import SymbolDescription
from .symbol import SymbolOccurrence


class LanguageInterface(Protocol):
    """Interface defining interaction with source code."""

    def open_file(self, file: Path) -> None:
        """Open a source file."""

    def close_file(self, file: Path) -> None:
        """Close a source file."""

    def file_changed(self, file: Path, content: str) -> None:
        """Call when a file was changed."""

    def get_symbol_definition_from_occurrence(self, occurrence: SymbolOccurrence) -> SymbolDescription:
        """Get symbol definition based on occurrence. Throws LspSymbolNotFound."""

    def get_symbol_definition_from_line_inside(self, line_in_symbol: LineInSymbolDefinition) -> SymbolDescription:
        """Get symbol definition from a line inside the symbol. Throws LspSymbolNotFound."""

    def get_references(self, file: Path, position: Position) -> list[Location]:
        """Get all references of a symbol."""

    def get_enclosing_function(self, location: Location) -> SymbolDescription:
        """
        Get the function enclosing a giving location. Throws LspSymbolNotFound.
        """

    def all_document_symbols(self, file: Path, functions_only: bool = False) -> Iterable[SymbolDescription]:
        """Iterate over all symbols in a given file."""

    def get_diagnostics(self) -> dict[Path, list[Diagnostics]]:
        """
        Get all diagnostics from workspace.
        """

    def symbol_names_equal(self, name_a: str, name_b: str) -> bool:
        """True if symbol names are equal, otherwise false."""

    def close(self) -> None:
        """Cleanup of language backend."""
