"""Tools for patcher."""

import atexit
from pathlib import Path
from typing import ClassVar
from typing import Iterable

from git import Repo
from pydantic import BaseModel
from pydantic import Field

from crs.agents.tools.named_base_tool import NamedBaseTool
from crs.base.context import CrsContext
from crs.logger import CRS_LOGGER
from crscommon.crash_report.base import BaseCrashReport
from crscommon.editor.editor import Location
from crscommon.editor.editor import SourceEditor
from crscommon.editor.lsp.exceptions import LspSymbolNotFound
from crscommon.editor.lsp.file_path import LspFilePath
from crscommon.editor.symbol_history import SymbolHistory
from crscommon.editor.util import format_symbol_identifier
from crscommon.editor.viewer import SymbolDescription
from crscommon.lang_config.config import LangConfig

log = CRS_LOGGER.getChild(__name__)

MAX_REFERENCES = 10


class LSPContext:
    """
    Manages data needed by the LSP stack (e.g. symbol history).
    """

    def __init__(self, ctx: CrsContext):

        self.crs_ctx = ctx
        self.lang_config = LangConfig.get(ctx)
        self.symbol_history = SymbolHistory(self.lang_config.editor)

        atexit.register(self.lang_config.editor.li.close)

    def add_known_symbols(self) -> None:
        """
        Add files and symbols known from the harnesses and files viewed by previous agents.
        """

        # add all harnesses to history and open it in lang server, because they are not in the src path
        for _, harness in self.crs_ctx.proj_yaml_model.harnesses.items():
            harness_src = self.crs_ctx.cp_path_abs / harness.source
            self.symbol_history.update(harness_src)
            self.lang_config.editor.li.open_file(harness_src)

        # add viewed files to history
        for file in self.crs_ctx.viewed_files:
            self.symbol_history.update(self.crs_ctx.cp_path_abs / file)

    def add_changed_symbols(self, repository: Repo) -> str:
        """
        Add all symbols which were changed in the last commit to the symbol history.
        Returns the git diff of the last commit.
        """

        vuln_commit = repository.commit(self.crs_ctx.commit_hash)
        vuln_parent = vuln_commit.parents[0]
        diff = repository.git.diff(vuln_parent.hexsha, vuln_commit.hexsha)

        # add changed files to symbol history
        for file_path in vuln_commit.stats.files.keys():
            file = self.crs_ctx.cp_path_abs / self.crs_ctx.src_path_rel / file_path
            if file.exists() and file.suffix in self.lang_config.editor.file_extensions:
                self.symbol_history.update(file)

        # add git diff symbols to history
        for symbol in self.lang_config.editor.get_symbols_occurring_in_diff(
            diff, self.crs_ctx.cp_path_abs / self.crs_ctx.src_path_rel
        ):
            self.symbol_history.update(symbol)

        return diff

    def add_crash_report(self, report: BaseCrashReport) -> None:
        """Add symbols from crash report."""
        for line_in_symbol in report.get_symbol_descriptions(self.lang_config.editor):
            self.symbol_history.update(line_in_symbol)

    def add_files(
        self,
        files: Iterable[Path],
    ) -> None:
        """
        Add files to symbol history.
        """
        for p in files:
            self.symbol_history.update(p)


UPDATE_SYMBOL_DEFINITION_DESCRIPTION = """
UpdateSymbolDefinition replaces the definition of a given symbol (e.g., function, class, variable) in source code with a new definition.

Example:
```
{
  "symbol_name": "add_numbers",
  "symbol_definition": "def add_numbers(a, b):\n    return a + b + 1",
  "ignore_errors": false
}
```
This will replace the existing `add_numbers` function with the new definition.

Note that it only updates the definition itself and does not automatically refactor any other
instances where the symbol's name, type, or signature is used.
Manual adjustment of those instances may be necessary.
"""


class UpdateSymbolDefinition(NamedBaseTool):
    """Updates symbol with new definition"""

    class Input(BaseModel):
        """Input."""

        symbol_name: str = Field(
            description=(
                "The exact name of the symbol (function, class, variable, etc.) whose definition should be replaced. "
                "This should match exactly as it appears in the code."
            )
        )
        symbol_definition: str = Field(
            description=(
                "The full source code string that should replace the current definition of the symbol."
                "This replaces the current definition in the code."
            )
        )
        ignore_errors: bool = Field(
            default=False,
            description=(
                "A boolean flag that, when set to true, applies the changes even if linters report errors. "
                "By default, it's false, requiring errors to be addressed before changes are applied."
            ),
        )

    NAME: ClassVar[str] = "UpdateSymbolDefinition"
    name: str = NAME
    description: str = UPDATE_SYMBOL_DEFINITION_DESCRIPTION
    args_schema: type[BaseModel] = Input

    # pylint: disable=no-member
    editor: SourceEditor = Field(exclude=True)
    symbol_history: SymbolHistory = Field(exclude=True)
    repository: Repo = Field(exclude=True)

    def get_errors(self, file_path: Path) -> str | None:
        """Get errors from language server."""
        # get diagnostics
        diagnostics = self.editor.get_diagnostics()

        # construct feedback for llm
        feedback: str = ""

        # get only errors from current file
        try:
            errors = diagnostics[file_path]
            errors = list(filter(lambda error: error.severity == 1, errors))
            if len(errors) > 0:
                for err in errors:
                    feedback += f"- `{err.message}`"

            if feedback != "":
                return f"The language server reported the following errors:\n {feedback}\n\n"
        except KeyError:
            pass

        return None

    def update_symbol_history(self, file_path: Path, symbols_before_change: list[SymbolDescription]) -> None:
        """Update the symbol history."""

        def is_new_symbol(symbol: SymbolDescription) -> bool:
            """Check if symbol is new."""
            return not any(self.editor.li.symbol_names_equal(s.name, symbol.name) for s in symbols_before_change)

        for s in self.editor.li.all_document_symbols(file_path):
            self.symbol_history.update(s, add=is_new_symbol(s))

    # pylint: disable=arguments-differ
    def _run(self, symbol_name: str, symbol_definition: str, ignore_errors: bool = False) -> str:
        log.info(f"Replacing symbol `{symbol_name}`")

        # get all applicable references
        references = self.symbol_history.find_references(symbol_name)

        if not references:
            log.warning(f"No references found for symbol `{symbol_name}` in symbol history.")

        # find first reference where symbol can be successfully retrieved
        for ref in references:
            try:
                # get uri of symbol location
                symbol_2b_replaced = self.editor.get_symbol_definition(ref)
                file_path = LspFilePath(symbol_2b_replaced.location.uri).path
                # backup symbols before change to later check if new symbol was added,
                # so we can add that to the history
                symbols_before_change = list(
                    self.editor.li.all_document_symbols(LspFilePath(symbol_2b_replaced.location.uri).path)
                )

                self.editor.replace_symbol_definition(symbol_2b_replaced.location, symbol_definition)

                if error_msg := self.get_errors(file_path):
                    log.warning("Symbol was successfully updated, but errors were reported.")
                    if not ignore_errors:
                        error_msg += "The changes you just made were therefore not applied to source code.\n"
                        # rollback changes
                        self.editor.restore_file(file_path, self.repository)
                    else:
                        error_msg += "The changes were applied despite the error message.\n"

                    return error_msg

                self.update_symbol_history(file_path, symbols_before_change)

                # no errors, commit changes
                self.repository.index.add([file_path.absolute().as_posix()])
                self.repository.index.commit("patch")

                log.info(f"Symbol `{symbol_name}` successfully updated")

                return (
                    "The symbol definition was successfully updated with the given definition. "
                    "\nThe syntax seems correct."
                )
            except LspSymbolNotFound:
                pass

        return f"The symbol `{symbol_name}` you wanted to update could not be found."


class GetSymbolDefinition(NamedBaseTool):
    """Get the definition of a symbol."""

    class Input(BaseModel):
        """Input."""

        symbol_name: str = Field(
            description=(
                "The exact name of the symbol (function, class, variable, etc.) to search for in the codebase. "
                "This parameter is used to locate the specific symbol within the "
                "codebase and return its full definition."
            )
        )

    NAME: ClassVar[str] = "GetSymbolDefinition"
    name: str = NAME
    description: str = (
        "Retrieves and returns the complete definition of a specified symbol within a codebase, "
        "including its type, location, scope, and documentation if available. "
        "Ideal for quickly understanding symbol roles and usages."
    )
    args_schema: type[BaseModel] = Input

    # pylint: disable=no-member
    editor: SourceEditor = Field(exclude=True)
    symbol_history: SymbolHistory = Field(exclude=True)

    def get_definition(self, symbol_name: str) -> SymbolDescription | None:
        """Return the symbol description if found, and None otherwise."""

        log.info(f"Searching symbol `{symbol_name}`")

        # get all applicable references
        references = self.symbol_history.find_references(symbol_name)

        # find first reference where symbol can be successfully retrieved
        for ref in references:
            try:
                symbol = self.editor.get_symbol_definition(ref)
                self.symbol_history.update(symbol)

                log.info(f"Found symbol `{symbol_name}`")

                return symbol
            except LspSymbolNotFound:
                pass

        return None

    # pylint: disable=arguments-differ
    def _run(self, symbol_name: str) -> str:
        """Run the tool and get a LLM-suitable answer"""
        symbol = self.get_definition(symbol_name)
        if symbol is not None:
            return f"```\n{symbol.definition}\n```\n"
        return f"The symbol `{symbol_name}` could not be found."


class FindReferences(NamedBaseTool):
    """Find references of a symbol."""

    class Input(BaseModel):
        """Input."""

        symbol_name: str = Field(
            description=(
                "Specifies the exact name of the symbol you want to search for in the source code. "
                "This parameter is case-sensitive and should match the symbol's name as used in "
                "the code to accurately retrieve all functions that reference it."
            )
        )

    NAME: ClassVar[str] = "FindReferences"
    name: str = NAME
    description: str = (
        "Allows you to identify all functions in the source code that utilize a specific symbol. "
        "To use this tool, you input the target symbol, and it returns a list of function names "
        "where the symbol appears. This is particularly useful for understanding symbol usage "
        "and dependencies in the codebase."
    )
    args_schema: type[BaseModel] = Input

    # pylint: disable=no-member
    editor: SourceEditor = Field(exclude=True)
    symbol_history: SymbolHistory = Field(exclude=True)

    # pylint: disable=arguments-differ
    def _run(self, symbol_name: str) -> str:
        log.info(f"Finding references to symbol `{symbol_name}`")

        # References are retrieved in 2 steps:
        # 1. The symbol history is searched using simple text search to return locations,
        #    were the requested symbol occurs (or is referenced). These are just textual references.
        # 2. For each location we then ask the language server to return all actual,
        #    semantic references in the code base (references for the first location for which we
        #    successfully retrieve references are returned).

        # Fetch only 10 textual references (step 1)
        text_search_references = self.symbol_history.find_references(symbol_name, max_count=10)

        if not text_search_references:
            log.warning(f"No text search references found for symbol `{symbol_name}`")
            return f"No references found for symbol `{symbol_name}`."

        # step 2
        for ref in text_search_references:
            try:
                referencing_locations = self.editor.get_locations_referencing_symbol(ref)
                break
            except LspSymbolNotFound:
                log.warning(f"No referencing locations found for {format_symbol_identifier(ref)}")
                continue

        log.debug(f"Editor found {len(referencing_locations)} referencing locations for `{symbol_name}`")
        # map a function to a set of referencing locations
        referencing_symbols: dict[SymbolDescription, set[Location]] = {}

        ref_count = 0
        for location in referencing_locations:
            try:
                function = self.editor.li.get_enclosing_function(location)
            except LspSymbolNotFound:
                log.warning(f"Could not find enclosing function for {location}")
                continue

            log.debug(f"Found enclosing function `{function.name}`")

            if function not in referencing_symbols:
                log.debug(f"Add symbol `{function.name}` to symbol history")
                referencing_symbols[function] = set()
                self.symbol_history.update(function)
            referencing_symbols[function].add(location)

            ref_count += 1
            if ref_count >= MAX_REFERENCES:
                break

        if not referencing_locations:
            return f"No references found for symbol `{symbol_name}`."

        return_msg = ""
        for function, locations in referencing_symbols.items():
            return_msg += f"\nReferences found in {function.name}:\n"

            uri = list(locations)[0].uri
            if locations:
                file_content = LspFilePath(uri).path.read_text("utf-8").splitlines()

                for location in locations:
                    assert location.uri == uri
                    start_line, end_line = location.range.start.line, location.range.end.line

                    # grep 4 lines before and after occurrence (just like vscode)
                    body = "\n".join(file_content[max(start_line - 4, 0) : end_line + 5])
                    return_msg += f"```\n{body}\n```\n"

        return return_msg
