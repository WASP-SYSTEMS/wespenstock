"""Definition of c specific types."""

from __future__ import annotations

import re
from pathlib import Path

from crs.base.util import docker_to_local_path

from .base import BaseCrashReport
from .base import BaseStackTrace
from .base import StackLevel

CPP_PATTERN = re.compile(r"::(\w+)\(.*?\) (.+?):(\d+)")
C_PATTERN = re.compile(r"in\s+(\w+)\s+(.+?):(\d+)")


class CStackTrace(BaseStackTrace):
    """Stacktrace for c language"""


class AsanReport(BaseCrashReport):
    """ASAN report class"""

    def _parse_asan_report(self, report: str) -> CStackTrace:
        """Parse stacktrace from ASAN report."""
        stacktrace = CStackTrace()

        for line in report.splitlines():
            line = line.strip()
            if re.search(r"^#[0-9]*", line):

                # parse stack trace for c++ and c functions
                if match := re.search(CPP_PATTERN, line):
                    func = match.group(1)
                    file = match.group(2)
                    line = match.group(3)
                else:
                    if match := re.search(C_PATTERN, line):
                        func = match.group(1)
                        file = match.group(2)
                        line = match.group(3)
                    else:
                        continue

                try:
                    file = docker_to_local_path(Path(file), self.src_path_abs)
                    if not file:
                        continue

                    lvl = StackLevel(file=file, function=func, line=int(line))
                    stacktrace.add_level(lvl)
                except IndexError:  # for some functions no line numbers are found
                    pass

        return stacktrace

    @staticmethod
    def extract_report(report: str) -> str:
        # extract the addr sanitizer report
        match = re.search(r"(?s)==(.*)==ABORTING", report)
        if not match:
            raise RuntimeError("No crash report found")

        return match.group(0)

    def get_stacktrace(self) -> CStackTrace:
        return self._parse_asan_report(self.report)
