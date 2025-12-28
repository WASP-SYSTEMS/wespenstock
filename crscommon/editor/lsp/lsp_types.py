"""
See https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/.
"""

from __future__ import annotations

import typing
from enum import IntEnum
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import FileUrl


class SymbolKind(IntEnum):
    """
    A symbol kind.

    https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#symbolKind
    """

    # pylint: disable=invalid-name
    File = 1
    Module = 2
    Namespace = 3
    Package = 4
    Class = 5
    Method = 6
    Property = 7
    Field = 8
    Constructor = 9
    Enum = 10
    Interface = 11
    Function = 12
    Variable = 13
    Constant = 14
    String = 15
    Number = 16
    Boolean = 17
    Array = 18
    Object = 19
    Key = 20
    Null = 21
    EnumMember = 22
    Struct = 23
    Event = 24
    Operator = 25
    TypeParameter = 26


class InitializeResult(BaseModel):
    """InitializeResult"""

    capabilities: dict  # Far too complex to define exact type. Maybe later...

    serverInfo: dict | None = None


class TextDocumentItem(BaseModel):
    """A Text Document Item."""

    uri: FileUrl
    """The text document's URI."""

    languageId: str
    """The text document's language identifier."""

    version: int = 0
    """
    The version number of this document
    (it will increase after each change, including undo/redo)
    """

    text: str
    """The content of the opened text document."""


class TextDocumentIdentifier(BaseModel):
    """A Text Document Identifier."""

    uri: FileUrl
    """The text document's URI."""


class Position(BaseModel):
    """Position."""

    model_config = ConfigDict(frozen=True)

    line: int
    """Line position in a document (zero-based)"""

    character: int
    """Character offset on a line in a document (zero-based)."""


class Range(BaseModel):
    """Range."""

    model_config = ConfigDict(frozen=True)

    start: Position
    end: Position


class Location(BaseModel):
    """Location."""

    model_config = ConfigDict(frozen=True)

    uri: FileUrl
    range: Range


class DocumentSymbol(BaseModel):
    """Symbol Information."""

    name: str
    """The name of this symbol."""

    detail: str | None = None
    """More detail for this symbol, e.g the signature of a function."""

    kind: SymbolKind
    """The kind of this symbol."""

    tags: list[int] | None = None
    """Tags for this completion item"""

    deprecated: bool | None = None
    """Indicates if this symbol is deprecated. @deprecated, use tags."""

    range: Range
    """The range enclosing this symbol not including leading/trailing whitespace but everything else like comments."""

    selectionRange: Range
    """The range that should be selected and revealed when this symbol is being picked."""

    children: list[DocumentSymbol] | None = None
    """Children of this symbol, e.g. properties of a class."""


class WorkspaceSymbol(BaseModel):
    """Symbol Information."""

    name: str
    """The name of this symbol."""

    kind: SymbolKind
    """The kind of this symbol."""

    tags: list[int] | None = None
    """Tags for this completion item"""

    containerName: str | None = None
    """The name of the symbol containing this symbol."""

    location: Location
    """The location of this symbol."""

    data: typing.Any | None = None


class TextDocumentContentChangeEvent(BaseModel):
    """TextDocumentContentChangeEvent."""

    text: str
    """The new text of the whole document."""


class CodeDescription(BaseModel):
    """Structure to capture a description for an error code."""

    href: str
    """An URI to open with more information about the diagnostic error."""


class DiagnosticRelatedInformation(BaseModel):
    """
    Represents a related message and source code location for a diagnostic.
    This should be used to point to code locations that cause or are related to
    a diagnostics, e.g. when duplicating a symbol in a scope.
    """

    location: Location
    """The location of this related diagnostic information."""

    message: str
    """The message of this related diagnostic information."""


class Diagnostics(BaseModel):
    """Diagnostics"""

    range: Range
    """The range at which the message applies."""

    severity: Literal[1, 2, 3, 4] | None = None
    """
    The diagnostic's severity. Can be omitted. If omitted it is up to the
    client to interpret diagnostics as error, warning, info or hint.
    """

    code: int | str | None = None
    """The diagnostic's code"""

    codeDescription: CodeDescription | None = None
    """An optional property to describe the error code."""

    source: str | None = None
    """A human-readable string describing the source of this diagnostic, e.g. 'typescript' or 'super lint'."""

    message: str
    """The message"""

    tags: Literal[1, 2] | None = None
    """Additional metadata about the diagnostic."""

    relatedInformation: list[DiagnosticRelatedInformation] | None = None
    """
    An array of related diagnostic information, e.g. when symbol-names within
    a scope collide all definitions can be marked via this property
    """

    data: typing.Any | None = None
    """
    A data entry field that is preserved between a `textDocument/publishDiagnostics` notification and
    `textDocument/codeAction` request
    """


class WorkspaceFolder(BaseModel):
    """WorkspaceFolder"""

    uri: FileUrl
    """The associated URI for this workspace folder."""

    name: str
    """
    The name of the workspace folder. Used to refer to this
    workspace folder in the user interface.
    """


class ReferenceContext(BaseModel):
    """ReferenceContext"""

    includeDeclaration: bool
    """Include the declaration of the current symbol."""


### Parameter definitions ###


class InitializeParams(BaseModel):
    """InitializeParams"""

    processId: int | None = None
    """The process Id of the parent process that started the server."""

    rootUri: FileUrl | None = None
    """The rootUri of the workspace."""

    capabilities: dict
    """The capabilities provided by the client (editor or tool)."""

    workspaceFolders: list[WorkspaceFolder] | None = None


class LogMessageParams(BaseModel):
    """LogMessageParams"""

    type: int
    """The message type."""

    message: str
    """The actual message."""


class WorkDoneProgressParams(BaseModel):
    """WorkDoneProgressParams"""

    workDoneToken: int | str | None = None
    """An optional token that a server can use to report work done progress."""


class PartialResultParams(BaseModel):
    """WorkDoneProgressParams"""

    partialResultToken: int | str | None = None
    """An optional token that a server can use to report partial results (e.g. streaming) to the client."""


class DidOpenTextDocumentParams(BaseModel):
    """DidOpenTextDocumentParams."""

    textDocument: TextDocumentItem
    """The document that was opened."""


class DidCloseTextDocumentParams(BaseModel):
    """DidCloseTextDocumentParams."""

    textDocument: TextDocumentIdentifier
    """The document that was closed."""


class DidChangeTextDocumentParams(BaseModel):
    """DidChangeTextDocumentParams."""

    textDocument: TextDocumentIdentifier
    """The document that was closed."""

    contentChanges: list[TextDocumentContentChangeEvent]
    """
    The actual content changes.
    """


class TextDocumentPositionParams(BaseModel):
    """TextDocumentPositionParams"""

    textDocument: TextDocumentIdentifier
    """The text document."""

    position: Position
    """The position inside the text document."""


class DefinitionParams(TextDocumentPositionParams, WorkDoneProgressParams, PartialResultParams):
    """DefinitionParams"""


class ReferenceParams(TextDocumentPositionParams, WorkDoneProgressParams, PartialResultParams):
    """DocumentSymbolParams."""

    context: ReferenceContext


class WorkspaceSymbolParams(WorkDoneProgressParams):
    """WorkspaceSymbolParams"""

    query: str
    """
    A query string to filter symbols by.
    Clients may send an empty string here to request all symbols."""


class DocumentSymbolParams(WorkDoneProgressParams):
    """DocumentSymbolParams."""

    textDocument: TextDocumentIdentifier
    """The document that was closed."""


class PublishDiagnosticParams(BaseModel):
    """DocumentDiagnosticParams"""

    uri: FileUrl
    """TThe URI for which diagnostic information is reported."""

    version: int | None = None
    """Optional the version number of the document the diagnostics are published for."""

    diagnostics: list[Diagnostics]
    """An array of diagnostic information items."""


class WorkDoneProgressCreateParams(BaseModel):
    """ProgressParams"""

    token: int | str
    """The progress token provided by the client or server."""


class ProgressParams(BaseModel):
    """ProgressParams"""

    token: int | str
    """The progress token provided by the client or server."""

    value: dict


SupportedMethods = Literal[
    "initialize",
    "initialized",
    "textDocument/didOpen",
    "textDocument/didClose",
    "workspace/symbol",
    "textDocument/documentSymbol",
    "textDocument/didChange",
    "textDocument/publishDiagnostics",
    "textDocument/definition",
    "textDocument/references",
    "window/workDoneProgress/create",
    "window/logMessage",
    "$/progress",
]

SupportedParams = (
    DidOpenTextDocumentParams
    | DidCloseTextDocumentParams
    | DidChangeTextDocumentParams
    | WorkspaceSymbolParams
    | PublishDiagnosticParams
    | ProgressParams
    | WorkDoneProgressCreateParams
    | DocumentSymbolParams
    | DefinitionParams
    | ReferenceParams
    | LogMessageParams
    | InitializeParams
)

SupportedResults = Location | list[Location] | list[WorkspaceSymbol] | list[DocumentSymbol] | InitializeResult


class ResponseError(BaseModel):
    """Response Error."""

    code: int
    message: str
    data: dict | None = None


class RequestMessage(BaseModel):
    """Response Message."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: int | str
    method: SupportedMethods
    params: SupportedParams


class NotificationMessage(BaseModel):
    """Notification Message."""

    jsonrpc: Literal["2.0"] = "2.0"
    method: SupportedMethods
    params: SupportedParams


class ResponseMessage(BaseModel):
    """Response Message."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: int | str
    result: SupportedResults | None = None
    error: ResponseError | None = None
