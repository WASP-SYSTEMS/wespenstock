"""
Execute a command as subprocess and log the out put live.
"""

from __future__ import annotations

import io
import os
import selectors
import subprocess
import time
from pathlib import Path
from typing import IO
from typing import Generator
from typing import Literal
from typing import cast

from crs.logger import CRS_LOGGER

log = CRS_LOGGER.getChild(__name__)


class CmdExecutor:
    """Class handling the command execution and logging."""

    class SelectorStreamManager:
        """Manage multiple selectors for streams. Currently only supports selectors.EVENT_READ."""

        def __init__(self, streams: list[IO[str]]) -> None:
            self.sel = selectors.DefaultSelector()

            # used to buffer lines until line is complete
            self.line_buffers: dict[int, list[str]] = {}

            for stream in streams:
                self.sel.register(stream, selectors.EVENT_READ)
                self.line_buffers[stream.fileno()] = []
                # make io for stream non-blocking
                os.set_blocking(stream.fileno(), False)

        def select_lines(self, timeout: int | None = None) -> Generator[tuple[IO[str], str]]:
            """
            Collect input from the streams and yield any complete lines.

            Yields:
                stream, line
            """

            for selector, _ in self.sel.select(timeout=timeout):
                assert isinstance(selector.fileobj, io.TextIOWrapper)
                assert selector.fileobj.fileno() in self.line_buffers
                stream = selector.fileobj

                ready = stream.read()
                if not ready:
                    # EOF received, unregister stream
                    self.sel.unregister(selector.fileobj)
                    continue

                parts = ready.split("\n")
                line_buffer = self.line_buffers[stream.fileno()]

                # If there is a buffered partial line, the first entry of
                # parts contributes to it; if parts has more than one entry,
                # the first one also completes the buffered line.
                if line_buffer:
                    line_buffer.append(parts.pop(0))
                    if not parts:
                        continue
                    yield stream, "".join(line_buffer)
                    line_buffer.clear()

                # If the first entry of parts completed a line, it has been
                # popped above; all remaining entries of parts (if any) but
                # the last one are complete lines.
                for line in parts[:-1]:
                    yield stream, line

                # If there was at least one \n in ready, the final entry of
                # parts is an incomplete line. As a special case, if the final
                # entry is empty, there is nothing to be buffered.
                if parts and parts[-1]:
                    line_buffer.append(parts[-1])

        def active_streams(self) -> list[IO[str]]:
            """Get a list of streams which selectors are still active."""
            return [
                cast(IO[str], key.fileobj)
                for key in self.sel.get_map().values()
                if isinstance(key.fileobj, io.TextIOWrapper)
            ]

        def pop_buffered_lines(self) -> Generator[tuple[int, str]]:
            """
            Get buffered lines and clear buffer.

            Yields:
                fd of stream, line
            """
            for fd, line_buffer in self.line_buffers.items():
                if line_buffer:
                    yield fd, "".join(line_buffer)
                    line_buffer.clear()

    def __init__(self) -> None:
        self.return_code: int | None = None
        self.pid: int | None

    def exec(
        self, args: list[str], cwd: Path | None = None, timeout: int | None = None
    ) -> Generator[tuple[Literal["stdout", "stderr"], str]]:
        """
        Execute a command and yield the stream, line and return code.

        args: command to be executed
        cwd: working directory
        timeout: timeout in seconds

        Yields:
            stream: stdout/stdin
            line: line outputted from stream
        """

        # pylint: disable=too-many-branches

        log.info("Executing %s", str(args))

        with subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="ignore",  # ignore decoding errors
            cwd=cwd.as_posix() if cwd else None,
        ) as process:
            assert process.stdout is not None and process.stderr is not None

            self.pid = process.pid

            log.info("Started process pid=%d", self.pid)

            selector = self.SelectorStreamManager([process.stdout, process.stderr])

            stream_name: dict[int, Literal["stdout", "stderr"]] = {
                process.stdout.fileno(): "stdout",
                process.stderr.fileno(): "stderr",
            }

            start_time = time.time()

            while True:
                # check timeout
                if timeout and time.time() - start_time > timeout:
                    # yield unfinished lines
                    for fd, line in selector.pop_buffered_lines():
                        yield stream_name[fd], line

                    err = f"Command {args} exceeded timeout after {timeout} seconds"
                    log.warning(err)
                    # TODO: Yes, docker containers continue to exist.
                    # Do we have a way of knowing the cid in oss-fuzz?
                    log.info("Killing process pid=%d", self.pid)
                    process.kill()
                    raise TimeoutError(err)

                for stream, line in selector.select_lines(timeout=1):
                    yield stream_name[stream.fileno()], line

                if not selector.active_streams():
                    # yield unfinished lines
                    for fd, line in selector.pop_buffered_lines():
                        yield stream_name[fd], line
                    # break if no selectors are registered anymore
                    break

            self.return_code = process.wait()
            log_msg = f"Process pid={self.pid} existed with code {self.return_code}"
            if self.return_code == 0:
                log.info(log_msg)
            else:
                log.warning(log_msg)
