"""Generates compilation DB. Needed for clangd."""

import json
from pathlib import Path

from .exceptions import CompilationDbError
from .logger import LSP_LOGGER

log = LSP_LOGGER.getChild(__name__)


class CompilationDbPreparer:
    """Prepare a compilation DB based of a already generated one."""

    def __init__(self, cp_path: Path) -> None:
        self.cp_path = cp_path

    def prepare_db(self, target_location: Path) -> None:
        """
        Prepare the compilation DB. The file compile_commands.json is expected to be present at project-path/work.
        It is important the file was generated using bear inside an oss-fuzz container.
        """

        log.info(f"Preparing compile_commands.json in {target_location.as_posix()}")

        self._check_for_compilation_db(self.cp_path / "work")

        db_source_path = self.cp_path / "work" / "compile_commands.json"

        with db_source_path.open() as fp:
            db = json.load(fp)
        # for every entry make the path relative to self.cp_path
        for entry in db:
            entry["directory"] = (self.cp_path / Path(*Path(entry["directory"]).parts[1:])).as_posix()

        new_db_path = target_location / "compile_commands.json"
        with new_db_path.open("w", encoding="utf-8") as fp:
            json.dump(db, fp, indent=2)

        log.info(f"Prepared {new_db_path.as_posix()}")

    @staticmethod
    def _check_for_compilation_db(path: Path) -> None:
        """Check whether a compile_commands.json exists in the source directory"""

        compilation_db_file = path / "compile_commands.json"

        if not compilation_db_file.exists():
            raise CompilationDbError(f"Could not find compile_commands.json at {path.as_posix()}")

        # load DB to make sure it is not empty
        try:
            with compilation_db_file.open() as fp:
                compilation_db = json.load(fp)
            if not compilation_db:
                raise CompilationDbError(f"{compilation_db_file.as_posix()} is empty")
        except json.JSONDecodeError as e:
            raise CompilationDbError(f"{compilation_db_file.as_posix()} cannot be parsed") from e

        log.info(f"Found compilation DB at {compilation_db_file}")
