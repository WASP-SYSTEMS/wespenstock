"""File path module for easy uri handling."""

import os
from pathlib import Path
from urllib.parse import unquote

from pydantic import FileUrl


class LspFilePath(os.PathLike):
    """
    File path class for easy switching between path and uri
    representation.
    """

    def __init__(self, path: Path | FileUrl):
        if isinstance(path, Path):
            self.path = path.absolute()
        else:
            self.path = Path(unquote(path.path or "")).absolute()

    @property
    def uri(self) -> FileUrl:
        """Get path as uri"""
        return FileUrl(self.path.as_uri())

    def __repr__(self) -> str:
        return f"LspFilePath({self.path.as_posix()!r})"

    def __str__(self) -> str:
        return str(self.path)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LspFilePath) and self.path == other.path

    def __hash__(self) -> int:
        return hash(self.path)

    def __fspath__(self) -> str:
        """Return the file system path representation of the object."""
        return self.path.as_posix()
