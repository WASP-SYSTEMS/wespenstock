"""Miscellaneous utilities."""

import subprocess
from pathlib import Path
from typing import Generator
from typing import Iterable
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T")

BASE_POLL_INTERVAL = 1.0  # seconds
POLL_INTERVAL_GROW = 1.1  # times
MAX_POLL_INTERVAL = 10.0  # seconds


def to_instance_or_none(value: T | type[T] | object, expected_type: type[T]) -> T | None:
    """
    Coerce value to an instance of expected_type or return None.

    If value is an instance of expected_type already, returns value unchanged.
    If value is a subclass of expected_type, instantiates it with no arguments.
    Otherwise, returns None.

    The behavior corresponds to that of the raise statement, except that the result is
    checked against the expected_type.
    """
    if isinstance(value, expected_type):
        return value
    if isinstance(value, type) and issubclass(value, expected_type):
        return value()
    return None


def dedupe(items: Iterable[T]) -> list[T]:
    """Return a copy of items with duplicates removed."""
    result: list[T] = []
    seen: set[T] = set()

    for item in items:
        if item in seen:
            continue
        result.append(item)
        seen.add(item)

    return result


def format_text_list(items: Iterable[str], joiner: str | None = None, if_empty: str = "(none)") -> str:
    """
    Build an English string expressing a list of the given items.

    joiner can be None for a pure comma-separated list, or a word like "and" or "or",
    it will only be used between the last two elements.
    If items is empty, if_empty is returned instead.
    """
    if not items:
        return if_empty

    fmt_items: list[str] = list(items)
    if joiner is not None and len(fmt_items) > 1:
        fmt_items[-1] = f"{joiner} {fmt_items[-1]}"
    return ", ".join(fmt_items)


class GrepMatch(BaseModel):
    """Grep match."""

    file: Path
    line: int


def recursive_grep(pattern: str, cwd: Path) -> list[GrepMatch]:
    """Run grep recursively in directory"""

    output = subprocess.run(
        ["grep", "-rn", pattern, "."],
        check=False,
        capture_output=True,
        text=True,
        errors="ignore",
        cwd=cwd.absolute(),
    )

    matches: list[GrepMatch] = []

    # parse output
    if output.stdout:
        for match in output.stdout.splitlines():
            try:
                file, line = match.split(":")[:2]
            except ValueError:
                pass

            matches.append(GrepMatch(file=Path(file), line=int(line)))

    return matches


def find_all(string: str, substr: str, overlapping: bool = False) -> Generator[int, None, None]:
    """Find all occurrences of substr in string."""
    start = 0
    offset = 1 if overlapping else len(substr)
    while True:
        start = string.find(substr, start)
        if start < 0:
            return
        yield start
        start += offset


def find_first_non_blank(line: str) -> int | None:
    """Return the (zero-based) index of the first character which is not whitespace."""

    for i, c in enumerate(line):
        if not c.isspace():
            return i

    return None


def docker_to_local_path(file: Path, base_path: Path) -> Path | None:
    """
    Projects are executed in docker so the paths point to the volume mounted in the
    docker container. This method adjusts a path to point to the correct local
    project according to base_path.
    None is returned if the path does not exist.
    """

    if file.is_absolute():
        file = Path(*file.parts[1:])  # remove /

    # Cut parts of the path from the beginning until the path exists in the CP.
    # Thereby we also find paths which are nested differently form the local folder
    # structure.
    for i in range(0, len(file.parts)):
        possible_path = (base_path / Path(*file.parts[i:])).resolve()
        if possible_path.exists():
            return possible_path

    return None
