"""C helper tests."""

import pytest

from crscommon.editor.language_helper import CLanguageHelper
from crscommon.editor.lsp.file_path import LspFilePath
from crscommon.editor.lsp.lsp_types import Location
from crscommon.editor.lsp.lsp_types import Position
from crscommon.editor.lsp.lsp_types import Range
from crscommon.editor.symbol import SymbolDescription

from .setup_test_env import TEST_C_SRC_PATH
from .setup_test_env import reset_c_env

# pylint: disable=redefined-outer-name
# because of fixtures


@pytest.fixture(scope="module", autouse=True)
def setup() -> None:
    """Setup module."""
    reset_c_env()


@pytest.fixture
def helper() -> CLanguageHelper:
    """Language helper instance."""
    return CLanguageHelper(TEST_C_SRC_PATH)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("symbol_name", "symbol_name", True),
        ("symbol_name", "symbolname", False),
    ],
)
def test_symbol_names_equal(helper: CLanguageHelper, a: str, b: str, expected: bool) -> None:
    """test_symbol_names_equal"""

    assert helper.symbol_names_equal(a, b) is expected


@pytest.mark.parametrize(
    "symbol, expected",
    [
        (
            SymbolDescription(
                name="onig_posix_regcomp",
                location=Location(
                    uri=LspFilePath(TEST_C_SRC_PATH / "src/onigposix.h").uri,
                    range=Range(start=Position(line=162, character=19), end=Position(line=162, character=37)),
                ),
                definition="",
            ),
            True,
        ),
        (
            SymbolDescription(
                name="onig_positive_int_multiply",
                location=Location(
                    uri=LspFilePath(TEST_C_SRC_PATH / "src/regint.h").uri,
                    range=Range(start=Position(line=960, character=11), end=Position(line=960, character=37)),
                ),
                definition="",
            ),
            True,
        ),
        (
            SymbolDescription(
                name="onig_positive_int_multiply",
                location=Location(
                    uri=LspFilePath(TEST_C_SRC_PATH / "src/regcomp.c").uri,
                    range=Range(start=Position(line=344, character=0), end=Position(line=344, character=26)),
                ),
                definition="",
            ),
            False,
        ),
    ],
)
def test_is_forward_declaration(helper: CLanguageHelper, symbol: SymbolDescription, expected: bool) -> None:
    """Test forward declaration detection"""

    assert helper.is_forward_declaration(symbol) is expected


@pytest.mark.parametrize(
    "text_doc, expected_symbol_names",
    [
        (
            LspFilePath(TEST_C_SRC_PATH / "src/iso8859_1.c"),
            [
                "regenc.h",
                "LARGE_S",
                "SMALL_S",
                "ENC_IS_ISO_8859_1_CTYPE",
                "LARGE_S",
                "SMALL_S",
                "LARGE_S",
                "LARGE_S",
                "SMALL_S",
                "CASE_FOLD_IS_NOT_ASCII_ONLY",
                "SMALL_S",
                "SMALL_S",
                "LARGE_S",
                "CASE_FOLD_IS_NOT_ASCII_ONLY",
                "CASE_FOLD_IS_NOT_ASCII_ONLY",
                "UChar",
                "UChar",
                "ARG_UNUSED",
                "UChar",
                "UChar",
                "INTERNAL_ONIGENC_CASE_FOLD_MULTI_CHAR",
                "CASE_FOLD_IS_NOT_ASCII_ONLY",
                "ONIGENC_IS_ASCII_CODE",
                "ONIGENC_ISO_8859_1_TO_LOWER_CASE",
                "ENC_IS_ISO_8859_1_CTYPE",
                "FALSE",
                "NULL",
                "NULL",
                "ENC_FLAG_ASCII_COMPATIBLE",
                "ENC_FLAG_SKIP_OFFSET_1",
                "EncISO_8859_1_CtypeTable",
                "CaseFoldMap",
                "OnigPairCaseFoldCodes",
                "apply_all_case_fold",
                "flag",
                "OnigCaseFoldType",
                "f",
                "OnigApplyAllCaseFoldFunc",
                "arg",
                "onigenc_apply_all_case_fold_with_map",
                "onigenc_apply_all_case_fold_with_map",
                "onigenc_apply_all_case_fold_with_map",
                "CaseFoldMap",
                "OnigPairCaseFoldCodes",
                "CaseFoldMap",
                "CaseFoldMap",
                "flag",
                "flag",
                "f",
                "f",
                "arg",
                "arg",
                "get_case_fold_codes_by_str",
                "flag",
                "OnigCaseFoldType",
                "p",
                "OnigUChar",
                "end",
                "OnigUChar",
                "items",
                "OnigCaseFoldCodeItem",
                "sa",
                "OnigUChar",
                "i",
                "j",
                "n",
                "p",
                "p",
                "p",
                "p",
                "p",
                "p",
                "end",
                "end",
                "p",
                "p",
                "p",
                "p",
                "p",
                "p",
                "flag",
                "ss_combination",
                "byte_len",
                "items",
                "items",
                "code_len",
                "items",
                "items",
                "code",
                "code",
                "items",
                "items",
                "OnigCodePoint",
                "n",
                "i",
                "i",
                "i",
                "i",
                "j",
                "j",
                "j",
                "j",
                "sa",
                "sa",
                "i",
                "i",
                "p",
                "p",
                "sa",
                "sa",
                "j",
                "j",
                "p",
                "p",
                "byte_len",
                "items",
                "items",
                "n",
                "n",
                "code_len",
                "items",
                "items",
                "n",
                "n",
                "code",
                "code",
                "items",
                "items",
                "n",
                "n",
                "OnigCodePoint",
                "sa",
                "sa",
                "i",
                "i",
                "code",
                "code",
                "items",
                "items",
                "n",
                "n",
                "OnigCodePoint",
                "sa",
                "sa",
                "j",
                "j",
                "n",
                "byte_len",
                "items",
                "items",
                "code_len",
                "items",
                "items",
                "code",
                "code",
                "items",
                "items",
                "OnigCodePoint",
                "p",
                "p",
                "p",
                "p",
                "p",
                "p",
                "p",
                "p",
                "end",
                "end",
                "p",
                "p",
                "p",
                "p",
                "p",
                "p",
                "flag",
                "ss_combination",
                "byte_len",
                "items",
                "items",
                "code_len",
                "items",
                "items",
                "code",
                "code",
                "items",
                "items",
                "OnigCodePoint",
                "p",
                "p",
                "flag",
                "p",
                "p",
                "p",
                "p",
                "byte_len",
                "items",
                "items",
                "code_len",
                "items",
                "items",
                "code",
                "code",
                "items",
                "items",
                "OnigCodePoint",
                "p",
                "p",
                "p",
                "p",
                "p",
                "p",
                "p",
                "p",
                "byte_len",
                "items",
                "items",
                "code_len",
                "items",
                "items",
                "code",
                "code",
                "items",
                "items",
                "OnigCodePoint",
                "code",
                "code",
                "items",
                "items",
                "OnigCodePoint",
                "byte_len",
                "items",
                "items",
                "code_len",
                "items",
                "items",
                "code",
                "code",
                "items",
                "items",
                "OnigCodePoint",
                "code",
                "code",
                "items",
                "items",
                "OnigCodePoint",
                "byte_len",
                "items",
                "items",
                "code_len",
                "items",
                "items",
                "code",
                "code",
                "items",
                "items",
                "OnigCodePoint",
                "code",
                "code",
                "items",
                "items",
                "OnigCodePoint",
                "byte_len",
                "items",
                "items",
                "code_len",
                "items",
                "items",
                "code",
                "code",
                "items",
                "items",
                "OnigCodePoint",
                "code",
                "code",
                "items",
                "items",
                "OnigCodePoint",
                "p",
                "p",
                "byte_len",
                "items",
                "items",
                "code_len",
                "items",
                "items",
                "code",
                "code",
                "items",
                "items",
                "OnigCodePoint",
                "p",
                "p",
                "p",
                "p",
                "p",
                "p",
                "byte_len",
                "items",
                "items",
                "code_len",
                "items",
                "items",
                "code",
                "code",
                "items",
                "items",
                "OnigCodePoint",
                "p",
                "p",
                "p",
                "p",
                "p",
                "p",
                "p",
                "p",
                "byte_len",
                "items",
                "items",
                "code_len",
                "items",
                "items",
                "code",
                "code",
                "items",
                "items",
                "OnigCodePoint",
                "p",
                "p",
                "mbc_case_fold",
                "flag",
                "OnigCaseFoldType",
                "pp",
                "OnigUChar",
                "end",
                "OnigUChar",
                "lower",
                "OnigUChar",
                "p",
                "OnigUChar",
                "pp",
                "pp",
                "p",
                "p",
                "flag",
                "flag",
                "lower",
                "lower",
                "lower",
                "pp",
                "pp",
                "flag",
                "p",
                "p",
                "lower",
                "lower",
                "OnigEncISO_8859_1_ToLowerCaseTable",
                "OnigEncISO_8859_1_ToLowerCaseTable",
                "p",
                "p",
                "lower",
                "lower",
                "p",
                "p",
                "pp",
                "pp",
                "is_code_ctype",
                "code",
                "OnigCodePoint",
                "ctype",
                "code",
                "code",
                "EncISO_8859_1_CtypeTable",
                "EncISO_8859_1_CtypeTable",
                "code",
                "code",
                "ctype",
                "OnigEncodingISO_8859_1",
                "OnigEncodingType",
                "onigenc_single_byte_mbc_enc_len",
                "onigenc_single_byte_mbc_enc_len",
                '"ISO-8859-1"',
                "onigenc_is_mbc_newline_0x0a",
                "onigenc_is_mbc_newline_0x0a",
                "onigenc_single_byte_mbc_to_code",
                "onigenc_single_byte_mbc_to_code",
                "onigenc_single_byte_code_to_mbclen",
                "onigenc_single_byte_code_to_mbclen",
                "onigenc_single_byte_code_to_mbc",
                "onigenc_single_byte_code_to_mbc",
                "mbc_case_fold",
                "mbc_case_fold",
                "apply_all_case_fold",
                "apply_all_case_fold",
                "get_case_fold_codes_by_str",
                "get_case_fold_codes_by_str",
                "onigenc_minimum_property_name_to_ctype",
                "onigenc_minimum_property_name_to_ctype",
                "is_code_ctype",
                "is_code_ctype",
                "onigenc_not_support_get_ctype_code_range",
                "onigenc_not_support_get_ctype_code_range",
                "onigenc_single_byte_left_adjust_char_head",
                "onigenc_single_byte_left_adjust_char_head",
                "onigenc_always_true_is_allowed_reverse_match",
                "onigenc_always_true_is_allowed_reverse_match",
                "onigenc_always_true_is_valid_mbc_string",
                "onigenc_always_true_is_valid_mbc_string",
            ],
        ),
    ],
)
def test_all_document_symbols(helper: CLanguageHelper, text_doc: LspFilePath, expected_symbol_names: list[str]) -> None:
    """Test test_all_document_symbols"""

    symbols = helper.all_document_symbols(text_doc, functions_only=False)
    symbol_names = [x.name for x in symbols]

    assert symbol_names == expected_symbol_names


@pytest.mark.parametrize(
    "name, location, expected_symbol",
    [
        (
            "STACK_RETURN",  # macro
            Location(
                uri=LspFilePath(TEST_C_SRC_PATH / "src/regexec.c").uri,
                range=Range(start=Position(line=2417, character=8), end=Position(line=2417, character=20)),
            ),
            """#define STACK_RETURN(addr)  do {\\
  int level = 0;\\
  StackType* k = stk;\\
  while (1) {\\
    k--;\\
    STACK_BASE_CHECK(k, "STACK_RETURN"); \\
    if (k->type == STK_CALL_FRAME) {\\
      if (level == 0) {\\
        (addr) = k->u.call_frame.ret_addr;\\
        break;\\
      }\\
      else level--;\\
    }\\
    else if (k->type == STK_RETURN)\\
      level++;\\
  }\\
} while(0)
""",
        ),
        (
            "onig_region_free",  # function
            Location(
                uri=LspFilePath(TEST_C_SRC_PATH / "src/regexec.c").uri,
                range=Range(start=Position(line=994, character=0), end=Position(line=994, character=16)),
            ),
            """extern void
onig_region_free(OnigRegion* r, int free_self)
{
  if (r != 0) {
    if (r->allocated > 0) {
      if (r->beg) xfree(r->beg);
      if (r->end) xfree(r->end);
      r->allocated = 0;
    }
#ifdef USE_CAPTURE_HISTORY
    history_root_free(r);
#endif
    if (free_self) xfree(r);
  }
}
""",
        ),
        (
            "MatchArg",  # struct
            Location(
                uri=LspFilePath(TEST_C_SRC_PATH / "src/regexec.c").uri,
                range=Range(start=Position(line=182, character=2), end=Position(line=182, character=10)),
            ),
            r"""typedef struct {
  void* stack_p;
  int   stack_n;
  OnigOptionType options;
  OnigRegion*    region;
  int            ptr_num;
  const UChar*   start;   /* search start position (for \G: BEGIN_POSITION) */
  unsigned int   match_stack_limit;
#ifdef USE_RETRY_LIMIT
  unsigned long  retry_limit_in_match;
  unsigned long  retry_limit_in_search;
  unsigned long  retry_limit_in_search_counter;
#endif
  OnigMatchParam* mp;
#ifdef USE_FIND_LONGEST_SEARCH_ALL_OF_RANGE
  int    best_len;      /* for ONIG_OPTION_FIND_LONGEST */
  UChar* best_s;
#endif
#ifdef USE_CALL
  unsigned long  subexp_call_in_search_counter;
#endif
#ifdef USE_SKIP_SEARCH
  UChar* skip_search;
#endif
} MatchArg;
""",
        ),
    ],
)
def test_get_symbol_definition(helper: CLanguageHelper, name: str, location: Location, expected_symbol: str) -> None:
    """Test test_all_document_symbols"""

    symbol = helper.get_symbol_definition(name, location)

    assert symbol.definition == expected_symbol


@pytest.mark.parametrize(
    "location, expected_symbol",
    [
        (
            Location(
                uri=LspFilePath(TEST_C_SRC_PATH / "src/regexec.c").uri,
                range=Range(start=Position(line=989, character=2), end=Position(line=989, character=17)),
            ),
            """extern OnigRegion*
onig_region_new(void)
{
  OnigRegion* r;

  r = (OnigRegion* )xmalloc(sizeof(OnigRegion));
  CHECK_NULL_RETURN(r);
  onig_region_init(r);
  return r;
}
""",
        ),
        (
            Location(
                uri=LspFilePath(TEST_C_SRC_PATH / "src/regexec.c").uri,
                range=Range(start=Position(line=2162, character=4), end=Position(line=2162, character=19)),
            ),
            r"""#define POP_TIL_BODY(aname, til_type) do {\
  while (1) {\
    stk--;\
    STACK_BASE_CHECK(stk, (aname));\
    if ((stk->type & STK_MASK_POP_HANDLED_TIL) != 0) {\
      if (stk->type == (til_type)) break;\
      else {\
        if (stk->type == STK_MEM_START) {\
          mem_start_stk[stk->zid] = stk->u.mem.prev_start;\
          mem_end_stk[stk->zid]   = stk->u.mem.prev_end;\
        }\
        else if (stk->type == STK_MEM_END) {\
          mem_start_stk[stk->zid] = stk->u.mem.prev_start;\
          mem_end_stk[stk->zid]   = stk->u.mem.prev_end;\
        }\
        POP_REPEAT_INC \
        POP_EMPTY_CHECK_START \
        POP_CALL \
        /* Don't call callout here because negation of total success by (?!..) (?<!..) */\
      }\
    }\
  }\
} while(0)
""",
        ),
    ],
)
def test_get_enclosing_function(helper: CLanguageHelper, location: Location, expected_symbol: str) -> None:
    """Test test_all_document_symbols"""

    symbol = helper.get_enclosing_function(location)

    assert symbol.definition == expected_symbol
