"""Simple source code editor."""

from pathlib import Path

from git import Repo

from crs.logger import CRS_LOGGER

from .lsp.file_path import LspFilePath
from .lsp.lsp_types import Location
from .viewer import LineInSymbolDefinition
from .viewer import SourceViewer
from .viewer import SymbolOccurrence
from .viewer import format_symbol_identifier

log = CRS_LOGGER.getChild(__name__)


class SourceEditor(SourceViewer):
    """
    Safely edit files with language server synchronization.
    """

    def insert_above_line(self, file_path: Path, text: str, line: int) -> None:
        """
        Insert text above line in document. Lines are zero-based.
        """
        self.replace_lines(file_path, text, line, line)

    def replace_lines(self, file_path: Path, text: str, start: int, end: int) -> None:
        """
        Replace lines with new text. Lines are zero-based.
        """

        self.li.open_file(file_path)

        buf: list[str]

        with open(file_path, encoding="utf-8") as f:
            buf = f.readlines()

        if start != end:
            del buf[start : end + 1]  # remove lines

        buf.insert(start, text)  # insert new content

        new_content = "".join(buf)

        file_path.write_text(new_content, encoding="utf-8")

        self.li.file_changed(file_path, new_content)

    def replace_symbol_definition(
        self, symbol_identifier: SymbolOccurrence | LineInSymbolDefinition | Location, text: str
    ) -> None:
        """
        Replace existing symbol with updated version.
        """

        if not isinstance(symbol_identifier, Location):
            log.debug(f"Trying to find symbol via {format_symbol_identifier(symbol_identifier)}")
            if isinstance(symbol_identifier, SymbolOccurrence):
                location = self.li.get_symbol_definition_from_occurrence(symbol_identifier).location
            else:
                location = self.li.get_symbol_definition_from_line_inside(symbol_identifier).location
        else:
            location = symbol_identifier

        if text[-1:] != "\n":
            text += "\n"

        self.replace_lines(
            LspFilePath(location.uri).path,
            text,
            location.range.start.line,
            location.range.end.line,
        )

    def restore_file(self, file_path: Path, repo: Repo) -> None:
        """
        Restore a file from last commit.
        """

        repo.git.restore(file_path.absolute())
        old_content = file_path.read_text()
        self.li.file_changed(file_path, old_content)
