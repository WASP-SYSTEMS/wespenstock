"""Cache for symbols form source code."""

from pathlib import Path

from crs.base.util import dedupe
from crs.logger import CRS_LOGGER
from crscommon.editor.lsp.file_path import LspFilePath
from crscommon.editor.viewer import SourceViewer
from crscommon.editor.viewer import SymbolDescription
from crscommon.editor.viewer import SymbolOccurrence

log = CRS_LOGGER.getChild(__name__)


class SymbolHistory:
    """History of all symbols the llm has seen so far."""

    def __init__(self, viewer: SourceViewer) -> None:

        self.buffer: list[SymbolDescription | Path] = []
        self.viewer = viewer

    def update(self, new_entry: SymbolDescription | Path, add: bool = True) -> None:
        """Update symbol definition. By default unknown definitions are added as new ones."""

        # replace entry if it already exists
        for i, curr_entry in enumerate(self.buffer):
            if (
                isinstance(new_entry, SymbolDescription)
                and isinstance(curr_entry, SymbolDescription)
                and self.viewer.li.symbol_names_equal(curr_entry.name, new_entry.name)
            ):
                self.buffer[i] = new_entry

                # append symbol to list and remove old definition
                self.buffer.append(new_entry)
                del self.buffer[i]

                log.info(
                    f"Updated symbol `{new_entry.name}`: {LspFilePath(new_entry.location.uri).path.name}:"
                    f"{new_entry.location.range.start.line}, {new_entry.location.range.end.line}"
                )
                return
            if isinstance(new_entry, Path) and isinstance(curr_entry, Path) and curr_entry == new_entry:
                log.info(f"Found path {new_entry.as_posix()}")
                return

        if not add:
            return

        # no entry found, append new one
        self.buffer.append(new_entry)

        if isinstance(new_entry, SymbolDescription):
            log.info(
                f"Added symbol `{new_entry.name}`: {LspFilePath(new_entry.location.uri).path.name}: "
                f"{new_entry.location.range.start.line}, {new_entry.location.range.end.line}"
            )
        else:
            log.info(f"Added path {new_entry.as_posix()}")

    def find_references(self, symbol_name: str, max_count: int | None = None) -> list[SymbolOccurrence]:
        """
        Get reference of symbol from history.
        Returns references to symbols based on simple text search.
        References in recently added symbols are returned first in
        reversed order in which they were added.
        """

        references: list[SymbolOccurrence] = []

        # iterate through buffer backwards to look at the latest symbols first
        for history_entry in reversed(self.buffer):

            if isinstance(history_entry, SymbolDescription):
                lines = history_entry.definition.splitlines()
                base_line = history_entry.location.range.start.line
                file_path = LspFilePath(history_entry.location.uri).path
            else:
                lines = history_entry.read_text().splitlines()
                base_line = 0
                file_path = history_entry

            # iterate over every line in symbol to find symbol occurrence of symbol_name
            for line_offset, line in enumerate(lines):
                if symbol_name in line:  # TODO: better identification if symbol
                    references.append(
                        SymbolOccurrence(
                            name=symbol_name,
                            file=file_path,
                            line=base_line + line_offset,
                        )
                    )

                if max_count is not None and len(references) >= max_count:
                    break
            else:
                continue
            break  # break only if inner loop did break

        return references

    def export_viewed_files(self, root: Path) -> list[Path]:
        """Return a list of files covered by this history relative to the given root."""
        root = root.resolve()

        result: list[Path] = []
        for symbol in self.buffer:
            if isinstance(symbol, SymbolDescription):
                file = LspFilePath(symbol.location.uri).path
            else:
                assert isinstance(symbol, Path)
                file = symbol

            try:
                result.append(file.resolve().relative_to(root))
            except ValueError:
                log.warning(f"File {file} not added to viewed files because it is not relative to the challenge root")

        return dedupe(result)
