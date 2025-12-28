"""Exceptions for the LSP code."""


class LspRequestFailed(Exception):
    """LSP request failed exception."""


class LspProtocolError(Exception):
    """Exception raised when the language server returns unparseable garbage."""


class LspSymbolNotFound(Exception):
    """Symbol not found exception."""


class CompilationDbError(Exception):
    """Any error associated with the file compile_commands.json produced by bear."""
