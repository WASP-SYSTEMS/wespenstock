"""Python Stacktraces."""

import re
from pathlib import Path

from crscommon.crash_report.base import BaseCrashReport
from crscommon.crash_report.base import BaseStackTrace
from crscommon.crash_report.base import StackLevel


class PythonReport(BaseCrashReport):
    """Python Crash Report."""

    def parse_line(self, line: str) -> StackLevel | None:
        """Parse line in format: File <filename>, line <lineno>, in <symbol_name>"""
        pattern = r"File (.*?), line (\d+), in (.*)"
        if m := re.match(pattern, line.strip()):
            filename = m.group(1)
            lineno = int(m.group(2))
            name = m.group(3)
            return StackLevel(
                function=name,
                line=lineno,
                file=Path(filename),
            )
        return None

    def get_stacktrace(self) -> BaseStackTrace:
        """Parse stacktrace."""
        stacktrace = BaseStackTrace()
        for line in self.report.splitlines():
            if trace := self.parse_line(line):
                stacktrace.add_level(trace)
        return stacktrace

    @staticmethod
    def extract_report(report: str) -> str:
        """We can just return report directly."""
        return report
