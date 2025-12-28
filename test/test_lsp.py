"""Tests for language server bindings."""

import os
import time
from pathlib import Path

import pytest
from pydantic import FileUrl

from crscommon.editor.lsp.base_client import BaseLspClient
from crscommon.editor.lsp.file_path import LspFilePath
from crscommon.editor.lsp.lsp_types import Position

from .setup_test_env import TEST_C_SRC_PATH
from .setup_test_env import reset_c_env

TEST_C_TEXT_DOC = LspFilePath(TEST_C_SRC_PATH / "src/regexec.c")


@pytest.fixture(scope="module", autouse=True)
def setup() -> None:
    """Setup module."""
    reset_c_env()


def test_lsp_file_path() -> None:
    """
    Testing lsp file path
    """

    path = LspFilePath(TEST_C_SRC_PATH)
    assert path.path.as_posix() == os.path.abspath(TEST_C_SRC_PATH)
    assert path.uri == FileUrl(f"file://{os.path.abspath(TEST_C_SRC_PATH)}")
    path = LspFilePath(FileUrl(f"file://{os.path.abspath(TEST_C_SRC_PATH)}"))
    assert path.path.as_posix() == os.path.abspath(TEST_C_SRC_PATH)
    assert path.uri == FileUrl(f"file://{os.path.abspath(TEST_C_SRC_PATH)}")


def test_percent_encoding() -> None:
    """
    Regression test for that one annoying percent encoding bug
    """

    uri = FileUrl("file:///usr/include/c%2B%2B/v1/algorithm")
    path = Path("/usr/include/c++/v1/algorithm")

    assert LspFilePath(uri).path == path
    assert LspFilePath(path).uri == uri


@pytest.mark.parametrize("text_doc, pos", [(TEST_C_TEXT_DOC, (2636, 29))])
def test_request_text_document_definition(
    lsp_client: BaseLspClient, text_doc: LspFilePath, pos: tuple[int, int]
) -> None:
    """Testing textDocument/definition request"""

    lsp_client.notification_text_document_did_open(text_doc)
    symbol_location = lsp_client.request_text_document_definition(
        text_doc,
        Position(
            line=pos[0],
            character=pos[1],
        ),
    )

    assert symbol_location[0].range.start.line


@pytest.mark.parametrize("text_doc, pos", [(TEST_C_TEXT_DOC, (862, 13))])
def test_request_text_document_references(
    lsp_client: BaseLspClient, text_doc: LspFilePath, pos: tuple[int, int]
) -> None:
    """Testing textDocument/definition request"""

    lsp_client.notification_text_document_did_open(text_doc)
    symbol_location = lsp_client.request_text_document_references(
        text_doc,
        Position(
            line=pos[0],
            character=pos[1],
        ),
    )

    assert symbol_location[0].range.start.line


@pytest.mark.parametrize(
    "text_doc, replacement",
    [
        (
            TEST_C_TEXT_DOC,
            ("start     = onig_get_start_by_callout_args(args);", "start     = onig_get_start_by_callout_args(args)"),
        ),
    ],
)
def test_text_document_did_change_and_diagnostics(
    lsp_client: BaseLspClient, text_doc: LspFilePath, replacement: tuple[str, str]
) -> None:
    """Testing textDocument/publishDiagnostics"""

    lsp_client.notification_text_document_did_open(text_doc)

    changed_text_doc = TEST_C_TEXT_DOC.path.read_text("utf-8").replace(*replacement)
    TEST_C_TEXT_DOC.path.write_text(changed_text_doc, encoding="utf-8")

    lsp_client.notification_text_document_did_change(text_doc, changed_text_doc)

    time.sleep(2)

    diagnostics = lsp_client.get_diagnostics()

    reset_c_env()

    error_found = False

    for d in diagnostics[text_doc]:
        if d.severity == 1:
            error_found = True

    assert error_found is True
