"""Test setup."""

import argparse
import subprocess
from pathlib import Path

from git import Repo
from typing_extensions import override

from crscommon.editor.lsp.compilation_db_preparer import CompilationDbPreparer
from crscommon.editor.lsp.exceptions import CompilationDbError
from crscommon.editor.lsp.logger import LSP_LOGGER
from oss_fuzz_integration.prepare_project import transform_project

log = LSP_LOGGER.getChild(__name__)

ONIGURUMA_TEST_COMMIT = "ee93c10"
LIGHTTPD_TEST_COMMIT = "985914f"
DOTENV_TEST_COMMIT = "01f8997"

TEST_C_SRC_PATH = Path("test/data") / "oniguruma"
TEST_CALL_TREE_CP = Path("test/data/cp-lighttpd-HEAD")
TEST_CALL_TREE_SRC = TEST_CALL_TREE_CP / "src" / "lighttpd1.4"
TEST_PYTHON_SRC_PATH = Path("test/data") / "python-dotenv"

OSS_FUZZ_PATH = Path("oss_fuzz_integration/oss-fuzz")


def clone_oss_fuzz() -> None:
    """Clone oss fuzz."""

    if not OSS_FUZZ_PATH.exists():
        Repo.clone_from(
            "https://github.com/google/oss-fuzz.git",
            OSS_FUZZ_PATH,
        )


def _prepare_c_env_with_make(repo_url: str, dst_path: Path) -> None:
    """Prepare a C env with make."""

    if not dst_path.exists():
        log.info(f"Cloning {repo_url}")
        Repo.clone_from(repo_url, dst_path)

    try:
        CompilationDbPreparer._check_for_compilation_db(dst_path)  # pylint: disable=protected-access
    except CompilationDbError:
        log.info("Preparing compilation DB")
        subprocess.run("./autogen.sh", shell=True, cwd=dst_path, check=False)
        subprocess.run("./configure", shell=True, cwd=dst_path, check=False)
        subprocess.run("make clean", shell=True, cwd=dst_path, check=False)
        subprocess.run("bear -- make", shell=True, cwd=dst_path, check=True)


def prepare_c_env() -> None:
    """
    Prepare test environment for c.
    """

    _prepare_c_env_with_make("https://github.com/kkos/oniguruma", TEST_C_SRC_PATH)


def prepare_env_call_tree() -> None:
    """
    Prepare test environment for c for call tree tests.
    """

    args = argparse.Namespace()
    args.commit_hash = None
    args.double_build_mode = False
    args.success_list_path = None
    args.failed_list_path = None
    args.no_oss_fuzz_checkout = False
    args.single_cp = False
    args.unique_tag = "test-call-tree-docker-container"
    args.success_list_path = Path("/dev/null")

    if not TEST_CALL_TREE_CP.exists():
        transform_project(
            "lighttpd",
            None,
            Path("test/data"),
            OSS_FUZZ_PATH,
            Repo(OSS_FUZZ_PATH),
            Path("oss_fuzz_integration"),
            args,
        )

    CompilationDbPreparer(TEST_CALL_TREE_CP).prepare_db(TEST_CALL_TREE_SRC)


def prepare_python_env() -> None:
    """
    Prepare test environment for python.
    """

    if not TEST_PYTHON_SRC_PATH.exists():
        log.info("Cloning https://github.com/theskumar/python-dotenv.git")
        Repo.clone_from("https://github.com/theskumar/python-dotenv.git", TEST_PYTHON_SRC_PATH)


def reset_c_env() -> None:
    """Reset the environment to its initial state."""
    Repo(TEST_C_SRC_PATH).git.checkout(ONIGURUMA_TEST_COMMIT, force=True)


def reset_env_call_tree() -> None:
    """Reset the environment to its initial state."""
    Repo(TEST_CALL_TREE_SRC).git.checkout(LIGHTTPD_TEST_COMMIT, force=True)


def reset_python_env() -> None:
    """Reset the environment to its initial state."""
    Repo(TEST_PYTHON_SRC_PATH).git.checkout(DOTENV_TEST_COMMIT, force=True)


class CompilationDbPreparerForTests(CompilationDbPreparer):
    """Mock CompilationDbGenerator."""

    def __init__(self) -> None:
        # cp path not used for tests, since prepare_db is overridden
        super().__init__(Path())

    @override
    def prepare_db(self, target_location: Path) -> None:
        """Creates the compilation database for make based project."""
        self._check_for_compilation_db(target_location)
