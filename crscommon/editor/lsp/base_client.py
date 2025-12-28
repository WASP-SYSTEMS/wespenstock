"""Base classes for the language server."""

from __future__ import annotations

import json
import random
import time
from abc import ABC
from abc import abstractmethod
from enum import Enum
from functools import wraps
from pathlib import Path
from queue import Empty
from queue import Queue
from threading import Lock
from threading import Thread
from typing import Any
from typing import Callable
from typing import Concatenate
from typing import ParamSpec
from typing import TypeVar
from typing import cast
from typing import get_args

from pydantic import ValidationError

from .exceptions import LspRequestFailed
from .file_path import LspFilePath
from .logger import LSP_LOGGER
from .lsp_types import DefinitionParams
from .lsp_types import Diagnostics
from .lsp_types import DidChangeTextDocumentParams
from .lsp_types import DidCloseTextDocumentParams
from .lsp_types import DidOpenTextDocumentParams
from .lsp_types import InitializeParams
from .lsp_types import Location
from .lsp_types import LogMessageParams
from .lsp_types import NotificationMessage
from .lsp_types import Position
from .lsp_types import ProgressParams
from .lsp_types import PublishDiagnosticParams
from .lsp_types import ReferenceContext
from .lsp_types import ReferenceParams
from .lsp_types import RequestMessage
from .lsp_types import ResponseMessage
from .lsp_types import SupportedMethods
from .lsp_types import SupportedParams
from .lsp_types import SupportedResults
from .lsp_types import SymbolKind
from .lsp_types import TextDocumentContentChangeEvent
from .lsp_types import TextDocumentIdentifier
from .lsp_types import TextDocumentItem
from .lsp_types import WorkspaceFolder

log = LSP_LOGGER.getChild(__name__)

P = ParamSpec("P")
R = TypeVar("R")


class FsmState(Enum):
    """Fmt State."""

    IDLE = 1
    WORK_DONE_PROGRESS_UPDATE = 2
    WORK_DONE_PROGRESS_CREATE = 3


# pylint: disable=R0904
# too many public methods: i dont care


class BaseLspClient(ABC):
    # pylint: disable=R0902
    # too many instance attributes
    """
    Base class for LSP client.
    """

    max_request_retries = 3

    capabilities: dict = {
        "window": {"workDoneProgress": True},
        "workspace": {
            "symbol": {
                "symbolKind": {
                    "valueSet": get_args(SymbolKind),
                },
                "resolveSupport": False,
            },
        },
        "textDocumentSync": {
            "openClose": True,
            "change": 1,  # changes are synced full (full files are send)
        },
        "textDocument": {
            "publishDiagnostics": {
                "relatedInformation": True,
                "codeDescriptionSupport": True,
            },
            "documentSymbol": {
                "symbolKind": {
                    "valueSet": get_args(SymbolKind),
                },
                "hierarchicalDocumentSymbolSupport": True,
            },
        },
    }

    @staticmethod
    def retry(f: Callable[Concatenate[BaseLspClient, P], R]) -> Callable[Concatenate[BaseLspClient, P], R]:
        """Retry lsp request when result is empty, because work might be in progress"""

        @wraps(f)
        def apply_retries(self: BaseLspClient, *args: P.args, **kwargs: P.kwargs) -> R:
            request_retries = 0
            result: R | None = None
            while request_retries < BaseLspClient.max_request_retries:
                result = f(self, *args, **kwargs)
                if result:
                    return result

                log.debug(f"Retrying {f.__name__}...")
                time.sleep(0.5)
                self.wait_until_work_is_done()
                request_retries += 1
            assert result is not None
            return result

        return apply_retries

    def __init__(self, src_path: Path) -> None:
        self.id = random.randint(1, 20)
        self.src_path = LspFilePath(src_path.absolute())

        self.reader_thread = Thread(target=self.reader_task)
        self.reader_thread.daemon = True

        self.result_queue: Queue[SupportedResults | None] = Queue()  # stores results received from server

        self.diagnostics: dict[LspFilePath, list[Diagnostics]] = {}  # stores diagnostics received from server
        self.diagnostics_mutex = Lock()

        self.work_in_progress = False  # indicates whether work is in progress on the server side
        self.work_in_progress_queue: Queue[bool] = Queue()  # used to wait for work done progress
        self.work_in_progress_mutex = Lock()

        self.server_capabilities: dict[str, Any] = {}

        self.initialized = False

        # may be set in derived class
        if not hasattr(self, "log_messages"):
            self.log_messages = False  # log all communication

    def send_response(self, msg: ResponseMessage) -> None:
        """Sends a response message."""
        self.write(msg.model_dump_json(exclude_none=True))

    def send_request(self, msg: RequestMessage) -> SupportedResults | None:
        """
        Send request to server
        """
        return self.send(msg)

    def send(self, msg: RequestMessage | NotificationMessage) -> SupportedResults | None:
        """
        Send json rpc to language server and get response.
        """
        json_req = msg.model_dump_json(exclude_none=True)

        if self.log_messages:
            log.debug("Client -> Server " + json_req)

        self.write(json_req)

        # Automatically making requests to the language server is far to fast.
        # Therefore, the delay
        time.sleep(0.3)  # TODO: better option????

        # check if request was send and wait for response
        if isinstance(msg, RequestMessage):
            try:
                return self.result_queue.get(timeout=4 * 60)
            except Empty:
                log.error("Timeout while waiting for LSP result.")
        return None  # notification was sent -> no response

    # pylint: disable= R0915:
    # Too many statements, it is a state machine
    def _protocol_fsm(self) -> None:
        # pylint: disable=R0912
        # too many branches
        """
        Finite state machine for protocol management.
        """

        state = FsmState.IDLE  # default state

        rsp: ResponseMessage
        req: RequestMessage
        notif: NotificationMessage

        while True:

            log.debug(f"STATE: {state}")

            match state:
                case FsmState.IDLE:
                    # read incoming data
                    data = self.read()
                    try:
                        msg = json.loads(data)
                    except ValueError:
                        log.error("Failed to decode json")
                        time.sleep(1)  # wait one second to prevent logs from cluttering
                        continue

                    if self.log_messages:
                        log.debug("Server -> Client " + json.dumps(msg))

                    if "error" in msg or "result" in msg:
                        # check if a response was received
                        try:
                            rsp = ResponseMessage.model_validate_json(data)
                            # error received from server
                            if rsp.error:
                                log.error(f"LSP Server error: {rsp.error}")
                                self.result_queue.put(None)
                            # response received
                            else:
                                self.result_queue.put(rsp.result)  # return response

                            continue
                        except ValidationError as e:
                            log.warning(f"Response not recognized: {json.dumps(json.loads(data))}\n{str(e)[:100]}")
                    elif "method" in msg and "id" in msg:
                        # check if a request was received
                        try:
                            req = RequestMessage.model_validate_json(data)
                            # new work done progress loop
                            if req.method == "window/workDoneProgress/create":
                                state = FsmState.WORK_DONE_PROGRESS_CREATE
                            continue
                        except ValidationError as e:
                            log.warning(f"Request not recognized: {json.dumps(json.loads(data))}\n{str(e)[:100]}")
                    else:
                        # check if a notification was received
                        try:
                            notif = NotificationMessage.model_validate_json(data)
                            # progress update
                            if notif.method == "$/progress":
                                state = FsmState.WORK_DONE_PROGRESS_UPDATE
                            # diagnostics received, can be ignored
                            elif notif.method == "textDocument/publishDiagnostics":
                                state = FsmState.IDLE
                                diagnostics_params: PublishDiagnosticParams = cast(
                                    PublishDiagnosticParams, notif.params
                                )
                                log.debug(f"Received diagnostics {diagnostics_params.diagnostics}")
                                with self.diagnostics_mutex:
                                    self.diagnostics[LspFilePath(diagnostics_params.uri)] = (
                                        diagnostics_params.diagnostics
                                    )
                            elif notif.method == "window/logMessage":
                                log_params: LogMessageParams = cast(LogMessageParams, notif.params)
                                log.debug(f"Server: {log_params.message}")
                            continue
                        except ValidationError as e:
                            log.warning(f"Notification not recognized: {json.dumps(json.loads(data))}\n{str(e)[:100]}")

                case FsmState.WORK_DONE_PROGRESS_CREATE:
                    self.send_response(self.make_response(None, req.id))
                    self._work_in_progress_created()
                    log.debug("Work in progress created")
                    state = FsmState.IDLE

                case FsmState.WORK_DONE_PROGRESS_UPDATE:
                    # work in progress started
                    if isinstance(notif.params, ProgressParams):
                        if "kind" in notif.params.value and notif.params.value["kind"] == "begin":
                            log.debug(f"Work in progress ({notif.params.token}): {notif.params.value['title']}")
                        # work in progress ended
                        elif "kind" in notif.params.value and notif.params.value["kind"] == "end":
                            log.debug(f"Work in progress ({notif.params.token}): Done")
                            self._work_in_progress_finished()
                    state = FsmState.IDLE

    def _work_in_progress_created(self) -> None:
        with self.work_in_progress_mutex:
            self.work_in_progress = True
            self.work_in_progress_queue = Queue()
            log.debug("_work_in_progress_created")

    def _work_in_progress_finished(self) -> None:
        with self.work_in_progress_mutex:
            self.work_in_progress = False
            self.work_in_progress_queue.put(True)
            log.debug("_work_in_progress_finished")

    def wait_until_work_is_done(self) -> None:
        """
        Waits until work done progress process has finished.
        """
        if self.work_in_progress:
            log.debug("wait_until_work_is_done")
            self.work_in_progress_queue.get(timeout=8 * 60)

    @staticmethod
    def make_notification(method: SupportedMethods, params: SupportedParams) -> NotificationMessage:
        """
        Make lsp notification.
        """
        return NotificationMessage(method=method, params=params)

    def make_request(self, method: SupportedMethods, params: SupportedParams) -> RequestMessage:
        """
        Make lsp notification.
        """
        return RequestMessage(id=self.id, method=method, params=params)

    @staticmethod
    def make_response(result: SupportedResults | None, msg_id: int | str) -> ResponseMessage:
        """
        Make lsp notification.
        """
        return ResponseMessage(id=msg_id, result=result)

    def init(self) -> dict:
        """
        Initialize lsp.
        """

        if self.initialized:
            return self.server_capabilities

        params = InitializeParams(
            processId=None,
            rootUri=self.src_path.uri,
            capabilities=self.capabilities,
            workspaceFolders=[
                WorkspaceFolder(
                    uri=self.src_path.uri,
                    name=f"workspace-name-{self.id}",
                )
            ],
        )

        req = self.make_request("initialize", params)
        result = self.send(req)
        self.server_capabilities = result.capabilities if result else {}  # type: ignore

        self.initialized = True

        return cast(dict, result)

    def notification_text_document_did_open(self, text_doc: LspFilePath, lang_id: str = "") -> None:
        """
        Open document. lang_id is automatically resolved if empty.
        """

        if not lang_id:
            lang_id = self.resolve_lang_id(text_doc)

        params = DidOpenTextDocumentParams(
            textDocument=TextDocumentItem(
                uri=text_doc.uri,
                languageId=lang_id,
                text=text_doc.path.read_text(),
            )
        )

        notification = self.make_notification("textDocument/didOpen", params)
        self.send(notification)

    def notification_text_document_did_close(self, text_doc: LspFilePath) -> None:
        """
        Close document.
        """

        params = DidCloseTextDocumentParams(
            textDocument=TextDocumentIdentifier(
                uri=text_doc.uri,
            )
        )

        notification = self.make_notification("textDocument/didClose", params)
        self.send(notification)

    def notification_text_document_did_change(self, text_doc: LspFilePath, text: str) -> None:
        """
        Changes applied to document. text must be the whole new document.
        """

        params = DidChangeTextDocumentParams(
            textDocument=TextDocumentIdentifier(uri=text_doc.uri),
            contentChanges=[TextDocumentContentChangeEvent(text=text)],
        )

        notification = self.make_notification("textDocument/didChange", params)
        self.send(notification)

    @retry
    def request_text_document_definition(self, text_doc: LspFilePath, position: Position) -> list[Location]:
        """
        Make textDocument/definition request.
        """
        params = DefinitionParams(textDocument=TextDocumentIdentifier(uri=text_doc.uri), position=position)

        req = self.make_request("textDocument/definition", params)
        result = self.send_request(req)
        if result is None:
            raise LspRequestFailed("textDocument/definition failed.")
        if isinstance(result, list):
            return cast(list[Location], result)
        return [cast(Location, result)]

    @retry
    def request_text_document_references(self, text_doc: LspFilePath, position: Position) -> list[Location]:
        """
        Make textDocument/references request.
        """
        params = ReferenceParams(
            textDocument=TextDocumentIdentifier(uri=text_doc.uri),
            position=position,
            context=ReferenceContext(includeDeclaration=False),
        )

        req = self.make_request("textDocument/references", params)
        result = self.send_request(req)
        if result is None:
            raise LspRequestFailed("textDocument/references failed.")
        if isinstance(result, list):
            return cast(list[Location], result)
        return [cast(Location, result)]

    def get_diagnostics(self) -> dict[LspFilePath, list[Diagnostics]]:
        """
        Get all diagnostics from workspace.
        Returned dict hast the following mapping: uri -> list[Diagnostics].
        NOTE: After changes have been applied in may take some time until the diagnostics are received.
        """
        with self.diagnostics_mutex:
            return self.diagnostics

    @staticmethod
    def resolve_lang_id(text_doc: LspFilePath) -> str:
        """
        Get language id based on file extension.
        """

        suffix = text_doc.path.suffix[1:]

        # https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocumentItem
        # for lang_id.
        if suffix in ("cpp", "cc"):
            return "cpp"
        if suffix == "c":
            return "c"
        if suffix == "py":
            return "python"

        return ""

    def start_reader_thread(self) -> None:
        """
        Should be called after read/write functions have been initialized.
        """
        # start reader thread as daemon
        self.reader_thread.start()

    def reader_task(self) -> None:
        """
        Worker thread reading data from language server.
        """
        self._protocol_fsm()

    def write(self, req: str) -> int:
        """
        Enclose the given request into LSP framing and send it to the language server.
        """
        req_bytes = req.encode()
        header = f"Content-Length: {len(req_bytes)}\r\n\r\n".encode()
        return self.write_raw(header + req_bytes)

    @abstractmethod
    def read(self) -> str:
        """
        Read function for underlying communication with language server.
        """

    @abstractmethod
    def write_raw(self, req: bytes) -> int:
        """
        Write function for underlying communication with language server.
        """

    @abstractmethod
    def close(self) -> None:
        """Close the server."""
