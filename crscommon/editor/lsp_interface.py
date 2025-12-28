"""Interface for LSP based source interaction."""

import time
from pathlib import Path
from typing import Iterable

from crs.logger import CRS_LOGGER

from .language_helper import LanguageHelper
from .language_interface import LanguageInterface
from .lsp.base_client import BaseLspClient
from .lsp.exceptions import LspSymbolNotFound
from .lsp.file_path import LspFilePath
from .lsp.lsp_types import Diagnostics
from .lsp.lsp_types import Location
from .lsp.lsp_types import Position
from .lsp.lsp_types import Range
from .symbol import LineInSymbolDefinition
from .symbol import SymbolDescription
from .symbol import SymbolOccurrence
from .util import format_symbol_identifier
from .util import occurrence_to_positions

log = CRS_LOGGER.getChild(__name__)

MAX_FORWARD_DECL_RECURSION = 4


class LspInterface(LanguageInterface):
    """Language interface for LSP based interaction."""

    def __init__(self, lsp_client: BaseLspClient, helper: LanguageHelper, file_extensions: list[str]) -> None:
        super().__init__()

        self.lsp = lsp_client
        self.helper = helper
        self.src_root = self.lsp.src_path
        self.file_extensions = file_extensions

        self.sample_file = Path(self.src_root.path / ("lsp_sample_file_to_initiate_indexing" + self.file_extensions[0]))

        # create sample file if it doesn't exist; the spaces are just there to ensure it is not empty
        self.sample_file.write_text("  \n  \n", encoding="utf-8")
        self.opened_text_docs: list[Path] = []

        # wait for indexing
        self.open_file(self.sample_file)
        time.sleep(3)  # give language server some time to get ready
        self.lsp.wait_until_work_is_done()

    def open_file(self, file: Path) -> None:
        """
        Open text document.

        This ensures that we open documents only once, since documents are
        generally not closed to improve performance.
        """
        file = file.absolute()
        if file not in self.opened_text_docs:
            self.lsp.notification_text_document_did_open(LspFilePath(file))
            self.opened_text_docs.append(file)

    def close_file(self, file: Path) -> None:
        """
        Close text document.
        """
        file = file.absolute()
        if file in self.opened_text_docs:
            self.lsp.notification_text_document_did_close(LspFilePath(file))
            self.opened_text_docs.remove(file)

    def file_changed(self, file: Path, content: str) -> None:
        self.lsp.notification_text_document_did_change(LspFilePath(file), content)

    def get_symbol_definition_from_occurrence(
        self, occurrence: SymbolOccurrence, forward_decl_recursion_count: int = 0
    ) -> SymbolDescription:
        """
        Get symbol definition based on occurrence.
        """

        err = ""
        occur_positions = occurrence_to_positions(occurrence)
        for pos in occur_positions:
            # get the document definition location (only gives us the first line and file)
            self.open_file(occurrence.file)
            definition_locations = self.lsp.request_text_document_definition(LspFilePath(occurrence.file), pos)
            if definition_locations:
                try:
                    return self._resolve_forward_declaration(
                        self.helper.get_symbol_definition(occurrence.name, definition_locations[0]),
                        forward_decl_recursion_count,
                    )
                except LspSymbolNotFound:
                    err += (
                        "Could not find symbol from location returned by textDocument/definition: "
                        f"{occurrence.name}, {LspFilePath(definition_locations[0].uri)}:"
                        f"{definition_locations[0].range.start.line + 1}\n"
                    )
            else:
                err += f"textDocument/definition returned no definitions for {format_symbol_identifier(occurrence)}\n"

        log.warning(err)
        raise LspSymbolNotFound(err)

    def get_symbol_definition_from_line_inside(self, line_in_symbol: LineInSymbolDefinition) -> SymbolDescription:
        """Get symbol definition from a line inside the symbol."""

        # Create fake location to extract symbol. ATM, the only criterion the location must fulfill
        # for the helper to work is that the line is contained within the symbol definition.
        location = Location(
            uri=LspFilePath(line_in_symbol.file).uri,
            range=Range(
                start=Position(line=line_in_symbol.line, character=0),
                end=Position(line=line_in_symbol.line, character=0),
            ),
        )

        return self.helper.get_symbol_definition(line_in_symbol.name, location)

    def get_references(self, file: Path, position: Position) -> list[Location]:
        """Get all references of a symbol."""
        self.open_file(file)
        return self.lsp.request_text_document_references(LspFilePath(file), position)

    def get_enclosing_function(self, location: Location) -> SymbolDescription:
        return self.helper.get_enclosing_function(location)

    def all_document_symbols(self, file: Path, functions_only: bool = False) -> Iterable[SymbolDescription]:
        return self.helper.all_document_symbols(LspFilePath(file), functions_only=functions_only)

    def _resolve_forward_declaration(
        self, symbol: SymbolDescription, forward_decl_recursion_count: int = 0
    ) -> SymbolDescription:
        """If a forward declaration is returned convert in to the real definition."""

        if not self.helper.is_forward_declaration(symbol):
            return symbol

        # prevent infinite recursion if no definition is found
        if forward_decl_recursion_count > MAX_FORWARD_DECL_RECURSION:
            return symbol

        forward_decl_recursion_count += 1

        # do it all again...

        occurrence: SymbolOccurrence | None = None

        # try for every occurrence in forward declaration
        for line_offset, line in enumerate(symbol.definition.splitlines()):
            if symbol.name in line:
                # create occurrence from forward declaration
                occurrence = SymbolOccurrence(
                    name=symbol.name,
                    file=LspFilePath(symbol.location.uri).path,
                    line=symbol.location.range.start.line + line_offset,
                )

                log.info(f"Resolving forward declaration for {format_symbol_identifier(occurrence)}")

                try:
                    # now we should get the actual definition
                    return self.get_symbol_definition_from_occurrence(occurrence, forward_decl_recursion_count)
                except LspSymbolNotFound:
                    pass

        log.error(f"Could not find `{symbol.name}` in its own forward declaration:\n{symbol.definition}")
        raise LspSymbolNotFound()

    def get_diagnostics(self) -> dict[Path, list[Diagnostics]]:
        """
        Get all diagnostics from workspace.

        Returned dict hast the following mapping: uri -> list[Diagnostics].
        NOTE: After changes have been applied in may take some time until the diagnostics are received.
        Therefore, the parameter delay, which just sleeps delay seconds until diagnostics are returned.
        """
        time.sleep(2.0)  # TODO: better option????
        return {k.path: v for (k, v) in self.lsp.get_diagnostics().items()}

    def symbol_names_equal(self, name_a: str, name_b: str) -> bool:
        return self.helper.symbol_names_equal(name_a, name_b)

    def close(self) -> None:
        self.lsp.close()
