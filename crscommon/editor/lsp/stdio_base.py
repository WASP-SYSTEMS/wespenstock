"""Base class for stdio based LSP clients."""

import io
import logging
import re
import subprocess
from pathlib import Path
from typing import BinaryIO
from typing import cast

from .base_client import BaseLspClient
from .exceptions import LspProtocolError
from .logger import LSP_LOGGER

HEADER_LINE = re.compile(rb"\A([A-Za-z0-9-]+): (.*)\r\n\Z")


class BaseStdioLspClient(BaseLspClient):
    """Base class for stdio based LSP clients."""

    def __init__(self, src_path: Path, server_cmd: str) -> None:
        super().__init__(src_path)

        stderr_fd: int | io.TextIOWrapper = subprocess.DEVNULL
        # log output only in DEBUG logs
        if LSP_LOGGER.level == logging.DEBUG:
            # get file handler from logger
            handler = next((h for h in LSP_LOGGER.handlers if isinstance(h, logging.FileHandler)), None)
            if handler is not None:
                stderr_fd = handler.stream

        # pylint: disable=R1732
        # consider using "with". No I won't.
        self.server_process: subprocess.Popen = subprocess.Popen(
            server_cmd.split(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_fd,  # write language server logs to log file
            cwd=src_path,
        )

        self.start_reader_thread()
        # initialize automatically
        self.init()

    def close(self) -> None:
        """Close server."""

        self.server_process.kill()

    def _read_header(self, stream: BinaryIO) -> int | None:
        """Read an LSP message header and return the Content-Length."""

        content_length: int | None = None
        first = True

        while True:
            line = stream.readline()
            if not line and first:
                # EOF
                return None
            first = False

            # Very common case: End of headers.
            if line == b"\r\n":
                break

            # The other common case: Just a Content-Length header.
            if line.startswith(b"Content-Length:") and line.endswith(b"\r\n"):
                if content_length is not None:
                    raise LspProtocolError("Duplicate Content-Length header")
                content_length = int(line[15:])
                continue

            # General case.
            m = HEADER_LINE.match(line)
            if not m:
                raise LspProtocolError(f"Invalid header line received: {line!r}")
            raw_name, raw_value = m.group(1, 2)
            try:
                name, value = raw_name.decode("ascii"), raw_value.decode("ascii")
            except UnicodeDecodeError as exc:
                raise LspProtocolError("Malformed LSP header name or value") from exc

            if name.lower() == "content-length":
                if content_length is not None:
                    raise LspProtocolError("Duplicate Content-Length header")
                content_length = int(value)
            # We don't care about the other headers (or actually header (we don't care)).

        if content_length is None:
            raise LspProtocolError("LSP message missing Content-Length header")
        return content_length

    def read(self) -> str:
        if self.server_process.stdout:
            # get content length
            content_len = self._read_header(cast(BinaryIO, self.server_process.stdout))
            if content_len is not None:
                # read response
                return self.server_process.stdout.read(content_len)
        return ""

    def read_all(self) -> str:
        """Read all remaining input."""
        if self.server_process.stdout:
            return self.server_process.stdout.read()
        return ""

    def write_raw(self, req: bytes) -> int:
        if self.server_process.stdin:
            n = self.server_process.stdin.write(req)
            self.server_process.stdin.flush()
            return n
        return -1
