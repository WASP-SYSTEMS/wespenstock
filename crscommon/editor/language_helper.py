"""Interface for language specific helper class."""

from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any
from typing import Iterable

from clang.cindex import CompilationDatabase
from clang.cindex import Cursor
from clang.cindex import CursorKind
from clang.cindex import Index
from clang.cindex import TranslationUnit
from clang.cindex import TranslationUnitLoadError

from crs.logger import CRS_LOGGER

from .lsp.exceptions import LspSymbolNotFound
from .lsp.file_path import LspFilePath
from .lsp.lsp_types import Location
from .lsp.lsp_types import Position
from .lsp.lsp_types import Range
from .symbol import SymbolDescription
from .util import get_lines

log = CRS_LOGGER.getChild(__name__)


class LanguageHelper(ABC):
    """
    Interface for language specific helper class.
    """

    def __init__(self, data: Any) -> None:
        super().__init__()
        self.data = data

    @abstractmethod
    def symbol_names_equal(self, name_a: str, name_b: str) -> bool:
        """True if symbol names are equal, otherwise false."""

    @abstractmethod
    def get_symbol_definition(self, name: str, location: Location) -> SymbolDescription:
        """
        Get symbol description of a symbol definition.

        name: Name of the symbol
        location: Location of the symbol definition returned by `textDocument/definition`

        The LSP request `textDocument/definition` returns the location of
        a symbol definition, but the location only includes the range of the
        symbol identifier and not the whole range of the definition.
        """

    @abstractmethod
    def get_enclosing_function(self, location: Location) -> SymbolDescription:
        """
        Get the function enclosing a giving definition. Throws LspSymbolNotFound.
        """

    @abstractmethod
    def is_forward_declaration(self, symbol: SymbolDescription) -> bool:
        """True if symbol is a forward declaration."""

    @abstractmethod
    def all_document_symbols(self, file_path: LspFilePath, functions_only: bool) -> Iterable[SymbolDescription]:
        """Iterate over all symbols in a given file."""


class CLanguageHelper(LanguageHelper):
    """Helper functions for C."""

    def __init__(self, data: Path):
        super().__init__(data)
        self.compilation_db_path = data
        self.comp_db = CompilationDatabase.fromDirectory(self.compilation_db_path.as_posix())

    def symbol_names_equal(self, name_a: str, name_b: str) -> bool:

        return name_a == name_b

    def get_symbol_definition(self, name: str, location: Location) -> SymbolDescription:

        file_path = LspFilePath(location.uri)
        tu = self._index_file(file_path, detailed=True)

        node = self._find_symbol(name, location, tu.cursor)
        if node is None:
            raise LspSymbolNotFound()

        return self._node_to_symbol_description(node, file_path.path.read_text(encoding="utf-8").splitlines())

    def get_enclosing_function(self, location: Location) -> SymbolDescription:
        file_path = LspFilePath(location.uri)
        tu = self._index_file(file_path, detailed=True)

        node = self._find_enclosing_function(location, tu.cursor)
        if node is None:
            raise LspSymbolNotFound()

        return self._node_to_symbol_description(node, file_path.path.read_text(encoding="utf-8").splitlines())

    def is_forward_declaration(self, symbol: SymbolDescription) -> bool:

        tu = self._index_file(LspFilePath(symbol.location.uri), detailed=True)

        node = self._find_symbol(symbol.name, symbol.location, tu.cursor)
        if node is None:
            raise LspSymbolNotFound()

        if node.kind == CursorKind.MACRO_DEFINITION:
            return False

        for child in node.get_children():
            # We have the full definition (not the forward declaration) if a direct child is a
            # compound statement (e.g the function body) or a field (e.g. struct)
            if child.kind in (CursorKind.COMPOUND_STMT, CursorKind.FIELD_DECL):
                return False

        return True

    def all_document_symbols(self, file_path: LspFilePath, functions_only: bool) -> Iterable[SymbolDescription]:

        tu = self._index_file(file_path, detailed=True)

        file_buf = file_path.path.read_text(encoding="utf-8").splitlines()

        for node in self._all_document_symbols(tu.cursor):
            if functions_only and not (node.kind == CursorKind.FUNCTION_DECL and node.is_definition()):
                continue

            yield self._node_to_symbol_description(node, file_buf)

    def _all_document_symbols(self, node: Cursor) -> Iterable[Cursor]:

        if not node.get_children():
            return

        # HACK: The type stubs are lying. :(
        assert hasattr(node.extent.start.file, "name")
        assert isinstance(node.extent.start.file.name, str)

        for child in node.get_children():
            if not hasattr(child.extent.start.file, "name"):
                continue

            # ensure only symbols from this file are returned
            if node.extent.start.file.name == child.extent.start.file.name:
                if child.spelling.strip() != "":
                    yield child
                yield from self._all_document_symbols(child)

    def _index_file(self, file_path: LspFilePath, detailed: bool = False) -> TranslationUnit:

        options = 0
        if detailed:
            # includes macros
            options = TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD

        index = Index.create()
        cmd = next(iter(self.comp_db.getCompileCommands(file_path.path.as_posix())))

        args: list[str] = []

        all_args = list(cmd.arguments)

        for i, arg in enumerate(all_args):
            if arg.startswith("-I"):
                # include params are can be give in two ways:
                #   1. '-I' '/path/inc'
                #   2. '-I/path/inc'
                if arg[2:].strip() == "":
                    # case 1
                    path = Path(cmd.directory) / all_args[i + 1]
                else:
                    # case 2
                    path = Path(cmd.directory) / arg[2:]

                path = path.resolve().absolute()
                if path.exists():
                    args.append(f"-I{path.as_posix()}")
            elif arg.startswith("-D"):
                args.append(arg)

        try:
            log.info(f"Parsing {file_path} with args {args}")
            tu = index.parse(file_path.path.as_posix(), args=args, options=options)
        except TranslationUnitLoadError:
            log.warning(f"Failed to parse {file_path} without PARSE_INCOMPLETE flag; trying with the flag")
            try:
                tu = index.parse(
                    file_path.path.as_posix(), args=args, options=options | TranslationUnit.PARSE_INCOMPLETE
                )
            except TranslationUnitLoadError:
                log.error(f"Failed to parse {file_path}")
                raise

        return tu

    def _find_symbol(self, name: str, location: Location, node: Cursor) -> Cursor | None:
        """Efficiently find a symbol in a file."""

        if name == node.spelling:
            return node

        # HACK: The type stubs are lying. :(
        assert hasattr(node.extent.start.file, "name")
        assert isinstance(node.extent.start.file.name, str)

        for child in node.get_children():
            if not hasattr(child.extent.start.file, "name"):
                continue

            if (
                node.extent.start.file.name == child.extent.start.file.name
                # lines are NOT zero-based in the clang API
                and child.extent.start.line <= location.range.start.line + 1 <= child.extent.end.line
                and (result := self._find_symbol(name, location, child)) is not None
            ):
                return result

        return None

    def _find_enclosing_function(
        self, location: Location, node: Cursor, last_enclosing_function: Cursor | None = None
    ) -> Cursor | None:
        """Find the function enclosing a location."""

        # HACK: The type stubs are lying. :(
        assert hasattr(node.extent.start.file, "name")
        assert isinstance(node.extent.start.file.name, str)

        if node.is_definition() or node.kind == CursorKind.MACRO_DEFINITION:
            last_enclosing_function = node

        for child in node.get_children():
            if not hasattr(child.extent.start.file, "name"):
                continue

            if (
                node.extent.start.file.name == child.extent.start.file.name
                # lines are NOT zero-based in the clang API
                and child.extent.start.line <= location.range.start.line + 1 <= child.extent.end.line
                and (result := self._find_enclosing_function(location, child, last_enclosing_function)) is not None
            ):
                return result

        return last_enclosing_function

    def _node_to_symbol_description(self, node: Cursor, file_buf: list[str]) -> SymbolDescription:

        # HACK: The type stubs are lying. :(
        assert hasattr(node.extent.start.file, "name")
        assert isinstance(node.extent.start.file.name, str)

        sym_range = Range(
            start=Position(line=node.extent.start.line - 1, character=node.extent.start.column - 1),
            end=Position(line=node.extent.end.line - 1, character=node.extent.start.column - 1),
        )

        return SymbolDescription(
            name=node.spelling,
            location=Location(uri=LspFilePath(Path(node.extent.start.file.name)).uri, range=sym_range),
            definition=get_lines(file_buf, sym_range.start.line, sym_range.end.line),
        )
