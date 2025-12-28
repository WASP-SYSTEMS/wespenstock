"""Language interface implementation for python."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import jedi
from jedi.api.classes import Name
from jedi.api.errors import SyntaxError as JediSyntaxError
from pydantic import ValidationError

from crs.base.util import find_first_non_blank
from crs.logger import CRS_LOGGER

from .language_interface import LanguageInterface
from .lsp.exceptions import LspSymbolNotFound
from .lsp.file_path import LspFilePath
from .lsp.lsp_types import Diagnostics
from .lsp.lsp_types import DocumentSymbol
from .lsp.lsp_types import Location
from .lsp.lsp_types import Position
from .lsp.lsp_types import Range
from .lsp.lsp_types import SymbolKind
from .lsp.lsp_types import WorkspaceSymbol
from .symbol import LineInSymbolDefinition
from .symbol import SymbolDescription
from .symbol import SymbolOccurrence
from .util import get_lines
from .util import occurrence_to_positions

log = CRS_LOGGER.getChild(__name__)


class JediInterface(LanguageInterface):
    """Language interface for Jedi based interaction with python."""

    def __init__(self, path: Path):
        self.path = path
        self.opened_files: set[Path] = set()
        self.project = jedi.Project(path=self.path.absolute())

    def script(self, path: Path) -> jedi.Script:
        """Returns a jedi Script of the path."""
        self.open_file(path)
        return jedi.Script(path=path, project=self.project)

    def open_file(self, file: Path) -> None:
        self.opened_files.add(file.absolute())

    def close_file(self, file: Path) -> None:
        pass

    def file_changed(self, file: Path, content: str) -> None:
        pass

    def get_symbol_definition_from_occurrence(self, occurrence: SymbolOccurrence) -> SymbolDescription:
        occur_positions = occurrence_to_positions(occurrence)
        script = self.script(occurrence.file)
        for pos in occur_positions:
            cursor_symbols = script.goto(line=pos.line + 1, column=pos.character)
            if not cursor_symbols:
                continue
            symbols: list[JediSymbol] = []
            for symbol in cursor_symbols:
                while symbol != (sym_goto := symbol.goto()[0]):
                    symbol = sym_goto
                symbols.append(JediSymbol(symbol))
            return symbols[0].to_symbol_description()

        raise LspSymbolNotFound()

    def get_symbol_definition_from_line_inside(self, line_in_symbol: LineInSymbolDefinition) -> SymbolDescription:
        line = line_in_symbol.file.read_text().splitlines()[line_in_symbol.line]
        col_start = find_first_non_blank(line)
        script = self.script(line_in_symbol.file)
        context = script.get_context(line=line_in_symbol.line + 1, column=col_start)

        if not context:
            raise LspSymbolNotFound()

        return JediSymbol(context).to_symbol_description()

    def get_references(self, file: Path, position: Position) -> list[Location]:
        script = self.script(file)
        cursor_symbols = script.get_references(line=position.line + 1, column=position.character)
        return [
            Location(
                uri=LspFilePath(s.module_path).uri,
                range=Range(
                    start=Position(
                        line=s.line - 1,
                        character=s.column,
                    ),
                    end=Position(
                        line=s.line - 1,
                        character=s.column,
                    ),
                ),
            )
            for s in cursor_symbols
        ]

    def get_enclosing_function(self, location: Location) -> SymbolDescription:
        script = self.script(LspFilePath(location.uri).path)
        context = script.get_context(line=location.range.start.line + 1, column=location.range.start.character)

        if not context:
            raise LspSymbolNotFound()

        return JediSymbol(context).to_symbol_description()

    def all_document_symbols(self, file: Path, functions_only: bool = False) -> Iterable[SymbolDescription]:
        if functions_only:
            raise NotImplementedError("TODO")  # :)

        script = self.script(file)
        # sometimes jedi throws weird exceptions
        try:
            return (JediSymbol(x).to_symbol_description() for x in script.get_names(all_scopes=True))
        except Exception:  # pylint: disable=W0718
            return (JediSymbol(x).to_symbol_description() for x in script.get_names())

    def get_diagnostics(self) -> dict[Path, list[Diagnostics]]:
        jedi_errors: dict[Path, list[JediSyntaxError]] = {}
        for path in self.opened_files:
            script = self.script(path)
            jedi_errors[path] = script.get_syntax_errors()

        def to_diagnostic(jedi_error: JediSyntaxError) -> Diagnostics:
            """Convert Jedi Syntax Error to LSP Diagnostics."""
            return Diagnostics(
                range=Range(
                    start=Position(
                        line=jedi_error.line - 1,
                        character=jedi_error.column,
                    ),
                    end=Position(
                        line=jedi_error.until_line - 1,
                        character=jedi_error.until_column,
                    ),
                ),
                message="A Syntax Error occurred",
                severity=1,  # Error
            )

        return {key: list(map(to_diagnostic, values)) for key, values in jedi_errors.items()}

    def symbol_names_equal(self, name_a: str, name_b: str) -> bool:
        if len(name_a) > len(name_b):
            return name_a.endswith(name_b) and name_a[-len(name_b) - 1] == "."
        if len(name_b) > len(name_a):
            return name_b.endswith(name_a) and name_b[-len(name_a) - 1] == "."
        return name_a == name_b

    def close(self) -> None:
        pass


class JediSymbol:
    """Jedi Symbol class with methods to convert to LSP types."""

    def __init__(self, symbol: Name):
        self.symbol = symbol

    # pylint: disable=R0911
    def kind(self) -> SymbolKind | None:
        """Create LSP kind."""
        match self.symbol.type:
            case "module":
                return SymbolKind.Module
            case "class":
                return SymbolKind.Class
            case "instance":
                return SymbolKind.Object
            case "function":
                return SymbolKind.Function
            case "param":
                return SymbolKind.TypeParameter
            case "path":
                return SymbolKind.File
            case "keyword":
                return None
            case "property":
                return SymbolKind.Property
            case "statement":
                return None
        return None

    def to_workspace_symbol(self) -> WorkspaceSymbol | None:
        """Convert Jedi symbol to WorkspaceSymbol."""
        container_name = None
        if parent := self.symbol.parent():
            container_name = parent.name
        if not (symbol_kind := self.kind()):
            return None
        try:
            return WorkspaceSymbol(
                name=self.symbol.name,
                kind=symbol_kind,
                containerName=container_name,
                location=self.to_location(),
            )
        except ValidationError:
            return None

    def to_document_symbol(self) -> DocumentSymbol | None:
        """Convert Jedi symbol to DocumentSymbol."""
        if not (symbol_kind := self.kind()):
            return None
        try:
            return DocumentSymbol(
                name=self.symbol.name,
                kind=symbol_kind,
                detail=self.symbol.description,
                range=self.to_location().range,
                selectionRange=self.to_location().range,
            )
        except ValidationError:
            return None

    def to_location(self) -> Location:
        """Create LSP Location."""
        lines = self.symbol.module_path.read_text().splitlines()

        if self.symbol.type == "module":
            start_line, start_col = 1, 0
            if lines:
                end_line, end_col = len(lines), len(lines[-1])
            else:
                end_line, end_col = 1, 0
        else:
            start_line, start_col = self.symbol.get_definition_start_position()
            end_line, end_col = self.symbol.get_definition_end_position()
            # TODO: multi-line decorators
            while start_line - 1 > 0 and lines[start_line - 1].strip().startswith("@"):
                start_line -= 1

        # lines are one-based, columns zero-based
        return Location(
            uri=LspFilePath(self.symbol.module_path).uri,
            range=Range(
                start=Position(
                    line=start_line - 1,
                    character=start_col,
                ),
                end=Position(
                    line=end_line - 1,
                    character=end_col,
                ),
            ),
        )

    def to_symbol_description(self) -> SymbolDescription:
        """Create SymbolDescription"""

        location = self.to_location()

        return SymbolDescription(
            name=self.symbol.name,
            definition=get_lines(
                self.symbol.module_path.read_text().splitlines(), location.range.start.line, location.range.end.line
            ),
            location=location,
        )
