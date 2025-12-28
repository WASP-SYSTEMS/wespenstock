"""Test lsp tools."""

from test.setup_test_env import TEST_C_SRC_PATH

import pytest

from crs.agents.tools.lsp_tools import FindReferences
from crs.agents.tools.lsp_tools import GetSymbolDefinition
from crscommon.editor.editor import SourceEditor
from crscommon.editor.symbol import LineInSymbolDefinition
from crscommon.editor.symbol_history import SymbolHistory

# pylint: disable=redefined-outer-name
# because of fixtures


@pytest.fixture(scope="module")
def symbol_history(c_editor: SourceEditor) -> SymbolHistory:
    """Symbol history"""

    history = SymbolHistory(c_editor)

    history.update(
        c_editor.get_symbol_definition(
            LineInSymbolDefinition(
                name="euctw_mbc_to_code",
                line=119,
                file=TEST_C_SRC_PATH / "src/euc_tw.c",
            )
        )
    )

    history.update(
        c_editor.get_symbol_definition(
            LineInSymbolDefinition(
                name="regset_search_body_position_lead",
                line=4552,
                file=TEST_C_SRC_PATH / "src/regexec.c",
            )
        )
    )

    return history


@pytest.mark.parametrize(
    "expected",
    [
        ("euctw_mbc_to_code", "onigenc_mbn_mbc_to_code(ONIG_ENCODING_EUC_TW, p, end);"),
        ("onigenc_mbn_mbc_to_code", "n = (OnigCodePoint )(*p++);"),
        ("enclen", "ONIGENC_MBC_ENC_LEN(enc,p)"),
        ("ONIGENC_MBC_ENC_LEN", "(enc)->mbc_enc_len(p)"),
        ("forward_search", "if (reg->dist_min != 0) {"),
        ("slow_search", "while (s < end) {"),
        ("sunday_quick_search_step_forward", " enc = reg->enc;"),
        ("ONIGENC_IS_SINGLEBYTE", "(ONIGENC_MBC_MAXLEN(enc) == 1)"),
    ],
)
def test_search_symbol_tool(c_editor: SourceEditor, symbol_history: SymbolHistory, expected: tuple[str, str]) -> None:
    """Test search symbol tool"""

    tool = GetSymbolDefinition(editor=c_editor, symbol_history=symbol_history)

    symbol, substr = expected

    assert substr in tool._run(symbol)  # pylint: disable=protected-access


@pytest.mark.parametrize(
    "expected",
    [
        (
            "euctw_mbc_to_code",
            """
References found in OnigEncodingEUC_TW:
```
  "EUC-TW",   /* name */
  4,          /* max enc length */
  1,          /* min enc length */
  onigenc_is_mbc_newline_0x0a,
  euctw_mbc_to_code,
  euctw_code_to_mbclen,
  euctw_code_to_mbc,
  euctw_mbc_case_fold,
  onigenc_ascii_apply_all_case_fold,
```
""",
        ),
        (
            "regset_search_body_position_lead",
            """
References found in onig_regset_search_with_param:
```
      MATCH_ARG_INIT(msas[i], set->rs[i].reg, option, set->rs[i].region,
                     orig_start, mps[i]);
    }

    r = regset_search_body_position_lead(set, str, end, start, range,
                                         orig_range, option, msas, rmatch_pos);
  }
  else {
    r = regset_search_body_regex_lead(set, str, end, start, orig_range,
```
""",
        ),
    ],
)
def test_find_references_tool_output_format(
    c_editor: SourceEditor, symbol_history: SymbolHistory, expected: tuple[str, str]
) -> None:
    """Test find references tool output format."""

    tool = FindReferences(editor=c_editor, symbol_history=symbol_history)

    symbol, expected_output = expected

    output = tool._run(symbol)  # pylint: disable=protected-access

    assert output == expected_output


@pytest.mark.parametrize(
    "expected",
    [
        (
            "forward_search",
            6,  # forward_search is referenced 6 times
            [  # forward_search is used in these two functions
                "regset_search_body_position_lead",
                "search_in_range",
            ],
        ),
    ],
)
def test_find_references_tool_completeness(
    c_editor: SourceEditor, symbol_history: SymbolHistory, expected: tuple[str, int, list[str]]
) -> None:
    """Test that find references tool finds all references to a symbol."""
    tool = FindReferences(editor=c_editor, symbol_history=symbol_history)

    symbol, num_references, functions = expected

    output = tool._run(symbol)  # pylint: disable=protected-access

    assert output.count(symbol) == num_references
    for line in output.splitlines():
        if line.startswith("References found"):
            assert any(line.endswith(f"{name}:") for name in functions)
