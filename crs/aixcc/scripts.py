"Wrapper around the CP's run.sh script."

import re
import shlex
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from crs.logger import CRS_LOGGER
from crscommon.cmd_executor.cmd_executor import CmdExecutor
from crscommon.logging.logging_provider import LOGGING_PROVIDER

from .settings import RUNSH_POV_TIMEOUT

log = CRS_LOGGER.getChild(__name__)

RUNSH_LOGGER = LOGGING_PROVIDER.new_logger("runsh")


class RunResult(BaseModel):
    """Result of run.sh"""

    stdout: str
    stderr: str
    return_code: int | None  # None means timeout

    # set via scan_for_sanitizer()
    sanitizer: str | None = None
    crash_output: str | None = None

    def scan_for_sanitizer(self) -> bool:
        """Scan for sanitizer. If true, fields sanitizer and crash_output are set."""

        for output in (self.stdout, self.stderr):
            match = re.search(r"==\d+==ERROR:\s*([^:]+):", output)

            if match:
                self.sanitizer = match.group(1).strip()
                self.crash_output = output
                return True

            if "Traceback (most recent call last):" in output:
                # TODO: regex to extract exception?
                self.sanitizer = "Python exception"
                self.crash_output = output
                return True

        return False


def invoke_run_sh(params: list[str], challenge_root: Path) -> RunResult:  # pylint: disable=R0912,R0915
    "Invoke the CP's run.sh with the given arguments."

    RUNSH_LOGGER.info("Executing: `%s`", shlex.join(["./run.sh", "-v", "-x", *params]))

    timeout: int | None = None
    if params and params[0] == "run_pov":
        timeout = RUNSH_POV_TIMEOUT.get()

    line_buffers: dict[Literal["stdout", "stderr"], list[str]] = {"stdout": [], "stderr": []}

    executor = CmdExecutor()

    try:
        for stream, line in executor.exec(["./run.sh", "-v", "-x", *params], cwd=challenge_root, timeout=timeout):
            RUNSH_LOGGER.info("%s: %s", stream, line)
            line_buffers[stream].append(line)
    except TimeoutError:
        RUNSH_LOGGER.warning("run.sh timed out after %d seconds", timeout)

    if executor.return_code == 0:
        RUNSH_LOGGER.info("Finished executing run.sh")
    elif executor.return_code is not None:
        RUNSH_LOGGER.warning("run.sh exited with code %d", executor.return_code)

    return RunResult(
        stdout="\n".join(line_buffers["stdout"]),
        stderr="\n".join(line_buffers["stderr"]),
        return_code=executor.return_code,
    )
