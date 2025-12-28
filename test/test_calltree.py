"""Call tree tests."""

from pathlib import Path

import pytest

from crs.agents.pov_builder.coverage.calltree import CallTree

from .setup_test_env import TEST_CALL_TREE_CP
from .setup_test_env import TEST_CALL_TREE_SRC

# pylint: disable=redefined-outer-name
# because of fixtures

CALL_TREE_OUT_PATH = Path("test/data/calltree")


@pytest.fixture(scope="module")
def call_tree() -> CallTree:
    """Call tree object."""

    return CallTree.from_project("lighttpd", TEST_CALL_TREE_CP, TEST_CALL_TREE_SRC, CALL_TREE_OUT_PATH, 0)


@pytest.mark.parametrize(
    "paths",
    [
        [
            ["LLVMFuzzerTestOneInput", "buffer_urldecode_path", "buffer_clen"],
            [
                "LLVMFuzzerTestOneInput",
                "run_burl_normalize",
                "burl_normalize",
                "burl_normalize_qs20_to_plus",
                "burl_normalize_qs20_to_plus_fix",
                "buffer_clen",
            ],
            [
                "LLVMFuzzerTestOneInput",
                "run_burl_normalize",
                "burl_normalize",
                "burl_normalize_qs20_to_plus",
                "buffer_clen",
            ],
            [
                "LLVMFuzzerTestOneInput",
                "run_burl_normalize",
                "burl_normalize",
                "burl_normalize_path",
                "buffer_clen",
            ],
            [
                "LLVMFuzzerTestOneInput",
                "run_burl_normalize",
                "burl_normalize",
                "burl_normalize_path",
                "buffer_clen",
            ],
            [
                "LLVMFuzzerTestOneInput",
                "run_burl_normalize",
                "burl_normalize",
                "burl_normalize_path",
                "buffer_clen",
            ],
            [
                "LLVMFuzzerTestOneInput",
                "run_burl_normalize",
                "burl_normalize",
                "burl_normalize_2F_to_slash",
                "burl_normalize_2F_to_slash_fix",
                "buffer_clen",
            ],
            [
                "LLVMFuzzerTestOneInput",
                "run_burl_normalize",
                "burl_normalize",
                "burl_normalize_2F_to_slash",
                "buffer_clen",
            ],
            [
                "LLVMFuzzerTestOneInput",
                "run_burl_normalize",
                "burl_normalize",
                "burl_contains_ctrls",
                "buffer_clen",
            ],
            [
                "LLVMFuzzerTestOneInput",
                "run_burl_normalize",
                "burl_normalize",
                "burl_normalize_basic_unreserved",
                "burl_normalize_basic_unreserved_fix",
                "buffer_clen",
            ],
            [
                "LLVMFuzzerTestOneInput",
                "run_burl_normalize",
                "burl_normalize",
                "burl_normalize_basic_unreserved",
                "buffer_clen",
            ],
            [
                "LLVMFuzzerTestOneInput",
                "run_burl_normalize",
                "burl_normalize",
                "burl_normalize_basic_required",
                "burl_normalize_basic_required_fix",
                "buffer_clen",
            ],
            [
                "LLVMFuzzerTestOneInput",
                "run_burl_normalize",
                "burl_normalize",
                "burl_normalize_basic_required",
                "buffer_clen",
            ],
        ]
    ],
)
def test_get_all_paths(call_tree: CallTree, paths: list[list[str]]) -> None:
    """Test get all paths."""

    assert call_tree.get_all_paths("LLVMFuzzerTestOneInput", "buffer_clen") == paths


@pytest.mark.parametrize(
    "tree",
    [
        """LLVMFuzzerTestOneInput
  buffer_init
    ck_assert_failed
      ck_bt_stderr
  buffer_init
    ck_assert_failed
      ck_bt_stderr
  run_burl_normalize
    buffer_copy_string_len
      buffer_alloc_replace
        buffer_realloc
          ck_assert_failed
            ck_bt_stderr
          ck_assert_failed
            ck_bt_stderr
    burl_normalize
      burl_normalize_basic_required
        buffer_clen
        buffer_truncate
        burl_normalize_basic_required_fix
          buffer_clen
          buffer_string_prepare_copy
            buffer_alloc_replace
              buffer_realloc
                ck_assert_failed
                  ck_bt_stderr
                ck_assert_failed
                  ck_bt_stderr
          buffer_copy_string_len
            buffer_alloc_replace
              buffer_realloc
                ck_assert_failed
                  ck_bt_stderr
                ck_assert_failed
                  ck_bt_stderr
      burl_normalize_basic_unreserved
        buffer_clen
        burl_is_unreserved
          light_isalnum
            light_isdigit
            light_isalpha
        buffer_truncate
        burl_normalize_basic_unreserved_fix
          buffer_clen
          buffer_string_prepare_copy
            buffer_alloc_replace
              buffer_realloc
                ck_assert_failed
                  ck_bt_stderr
                ck_assert_failed
                  ck_bt_stderr
          burl_is_unreserved
            light_isalnum
              light_isdigit
              light_isalpha
          buffer_copy_string_len
            buffer_alloc_replace
              buffer_realloc
                ck_assert_failed
                  ck_bt_stderr
                ck_assert_failed
                  ck_bt_stderr
      burl_scan_qmark
      burl_contains_ctrls
        buffer_clen
      burl_normalize_2F_to_slash
        buffer_clen
        burl_normalize_2F_to_slash_fix
          buffer_clen
          buffer_truncate
      burl_normalize_path
        buffer_clen
        buffer_copy_string_len
          buffer_alloc_replace
            buffer_realloc
              ck_assert_failed
                ck_bt_stderr
              ck_assert_failed
                ck_bt_stderr
        buffer_truncate
        buffer_path_simplify
          buffer_is_blank
          buffer_blank
            buffer_truncate
            buffer_extend
              buffer_string_prepare_append_resize
                buffer_string_prepare_copy
                  buffer_alloc_replace
                    buffer_realloc
                      ck_assert_failed
                        ck_bt_stderr
                      ck_assert_failed
                        ck_bt_stderr
                ck_assert_failed
                  ck_bt_stderr
                buffer_realloc
                  ck_assert_failed
                    ck_bt_stderr
                  ck_assert_failed
                    ck_bt_stderr
        buffer_clen
        buffer_clen
        buffer_append_string_len
          buffer_extend
            buffer_string_prepare_append_resize
              buffer_string_prepare_copy
                buffer_alloc_replace
                  buffer_realloc
                    ck_assert_failed
                      ck_bt_stderr
                    ck_assert_failed
                      ck_bt_stderr
              ck_assert_failed
                ck_bt_stderr
              buffer_realloc
                ck_assert_failed
                  ck_bt_stderr
                ck_assert_failed
                  ck_bt_stderr
      burl_normalize_qs20_to_plus
        buffer_clen
        burl_normalize_qs20_to_plus_fix
          buffer_clen
          buffer_truncate
  buffer_urldecode_path
    buffer_clen
    hex2int
    hex2int
  buffer_free
  buffer_free"""
    ],
)
def test_get_indented_call_tree(call_tree: CallTree, tree: str) -> None:
    """Test generation of indented call tree."""

    t = call_tree.get_indented_call_tree("LLVMFuzzerTestOneInput")

    str_tree_list = []

    for n in t:
        str_tree_list.append(f"{' ' * n[0] * 2}{n[1]}")

    assert "\n".join(str_tree_list) == tree
