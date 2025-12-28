"""Clangd LSP client."""

import shutil
from pathlib import Path

from .compilation_db_preparer import CompilationDbPreparer
from .logger import LSP_LOGGER
from .stdio_base import BaseStdioLspClient

log = LSP_LOGGER.getChild(__name__)


class ClangdClient(BaseStdioLspClient):
    """Clangd client."""

    def __init__(self, src_path: Path, compilation_db_preparer: CompilationDbPreparer | None) -> None:
        server_cmd: str = (shutil.which("clangd") or "clangd") + " --log=verbose"

        if compilation_db_preparer:
            compilation_db_preparer.prepare_db(src_path)

        CompilationDbPreparer._check_for_compilation_db(src_path)

        super().__init__(src_path, server_cmd)
