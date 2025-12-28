"""Pyest config."""

import time
from pathlib import Path

import pytest
from _pytest.fixtures import FixtureRequest

from crscommon.editor.editor import SourceEditor
from crscommon.editor.jedi_interface import JediInterface
from crscommon.editor.language_helper import CLanguageHelper
from crscommon.editor.lsp.base_client import BaseLspClient
from crscommon.editor.lsp.clangd_client import ClangdClient
from crscommon.editor.lsp_interface import LspInterface
from crscommon.logging.logging_provider import LOGGING_PROVIDER

from .setup_test_env import TEST_C_SRC_PATH
from .setup_test_env import TEST_PYTHON_SRC_PATH
from .setup_test_env import CompilationDbPreparerForTests
from .setup_test_env import clone_oss_fuzz
from .setup_test_env import prepare_c_env
from .setup_test_env import prepare_env_call_tree
from .setup_test_env import prepare_python_env
from .setup_test_env import reset_c_env
from .setup_test_env import reset_env_call_tree
from .setup_test_env import reset_python_env

# This is a helper variable used to make sure, that ClangdClient is instantiated
# after the env is prepared and logging is initialized. Otherwise, there would be no LSP
# logs from the client.
CLANGD: ClangdClient


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment() -> None:
    """Test env setup."""

    LOGGING_PROVIDER.init_logging(Path("test_log"))

    clone_oss_fuzz()

    prepare_c_env()
    reset_c_env()

    prepare_env_call_tree()
    reset_env_call_tree()

    prepare_python_env()
    reset_python_env()

    # init LSP clients
    global CLANGD  # pylint: disable=global-statement
    CLANGD = ClangdClient(TEST_C_SRC_PATH, CompilationDbPreparerForTests())


# pylint: disable=redefined-outer-name
# because of fixtures


# Execute fixture and dependent tests for each LSP client.
@pytest.fixture(scope="module", params=["clangd"])
def lsp_client(request: FixtureRequest) -> BaseLspClient:
    """Fixture used to select the lsp client in pytest.mark.parametrize"""

    time.sleep(2)

    client: BaseLspClient

    # select client based on param
    if request.param == "clangd":
        client = CLANGD
    else:
        raise RuntimeError("Invalid LSP client specified")

    client.wait_until_work_is_done()

    return client


@pytest.fixture(scope="module")
def c_editor(lsp_client: BaseLspClient) -> SourceEditor:
    """Editor fixture for c."""

    file_extensions = [".c", ".cc", ".cpp", ".h"]
    return SourceEditor(LspInterface(lsp_client, CLanguageHelper(TEST_C_SRC_PATH), file_extensions), file_extensions)


@pytest.fixture(scope="module")
def py_editor() -> SourceEditor:
    """Editor fixture for python."""

    file_extensions = [".py"]
    return SourceEditor(JediInterface(TEST_PYTHON_SRC_PATH), file_extensions)
