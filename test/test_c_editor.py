"""
Editor tests for c.
"""

from pathlib import Path

import pytest
from git import Repo

from crscommon.editor.editor import SourceEditor
from crscommon.editor.symbol import LineInSymbolDefinition
from crscommon.editor.symbol import SymbolOccurrence
from crscommon.editor.util import get_lines

from .setup_test_env import TEST_C_SRC_PATH
from .setup_test_env import reset_c_env

# pylint: disable=redefined-outer-name
# because of fixtures


@pytest.fixture(autouse=True)
def reset_env() -> None:
    """Reset env before every test."""
    reset_c_env()


@pytest.mark.parametrize(
    "file, lines, expected",
    [
        (
            TEST_C_SRC_PATH / "src/regexec.c",
            (40, 44),
            """#ifdef USE_CRNL_AS_LINE_TERMINATOR
#define ONIGENC_IS_MBC_CRNL(enc,p,end) \\
  (ONIGENC_MBC_TO_CODE(enc,p,end) == 13 && \\
   ONIGENC_IS_MBC_NEWLINE(enc,(p+enclen(enc,p)),end))
#endif
""",
        )
    ],
)
def test_get_lines(file: Path, lines: tuple[int, int], expected: str) -> None:
    """Test get lines"""
    assert get_lines(file.read_text().splitlines(), lines[0], lines[1]) == expected


@pytest.mark.parametrize(
    "occurrence, expected_symbol",
    [
        (
            SymbolOccurrence(
                name="MATCH_AT_ERROR_RETURN",  # macro
                file=TEST_C_SRC_PATH / "src/regexec.c",
                line=3172,
            ),
            """#define MATCH_AT_ERROR_RETURN(err_code) do {\\
  best_len = err_code; goto match_at_end;\\
} while(0)
""",
        ),
        (
            SymbolOccurrence(
                name="onig_get_used_stack_size_in_callout",  # function
                file=TEST_C_SRC_PATH / "sample/callout.c",
                line=44,
            ),
            """extern int
onig_get_used_stack_size_in_callout(OnigCalloutArgs* a, int* used_num, int* used_bytes)
{
  int n;

  n = (int )(a->stk - a->stk_base);

  if (used_num != 0)
    *used_num = n;

  if (used_bytes != 0)
    *used_bytes = n * sizeof(StackType);

  return ONIG_NORMAL;
}
""",
        ),
        (
            SymbolOccurrence(
                name="onig_get_used_stack_size_in_callout",  # with forward declartion resolving
                file=TEST_C_SRC_PATH / "src/regexec.c",
                line=6379,
            ),
            """extern int
onig_get_used_stack_size_in_callout(OnigCalloutArgs* a, int* used_num, int* used_bytes)
{
  int n;

  n = (int )(a->stk - a->stk_base);

  if (used_num != 0)
    *used_num = n;

  if (used_bytes != 0)
    *used_bytes = n * sizeof(StackType);

  return ONIG_NORMAL;
}
""",
        ),
        (
            LineInSymbolDefinition(
                name="onig_get_used_stack_size_in_callout",  # function from LineInSymbolDefinition
                file=TEST_C_SRC_PATH / "src/regexec.c",
                line=6385,
            ),
            """extern int
onig_get_used_stack_size_in_callout(OnigCalloutArgs* a, int* used_num, int* used_bytes)
{
  int n;

  n = (int )(a->stk - a->stk_base);

  if (used_num != 0)
    *used_num = n;

  if (used_bytes != 0)
    *used_bytes = n * sizeof(StackType);

  return ONIG_NORMAL;
}
""",
        ),
        (
            SymbolOccurrence(
                name="OnigSyntaxType",  # struct
                file=TEST_C_SRC_PATH / "sample/sql.c",
                line=9,
            ),
            """typedef struct {
  unsigned int   op;
  unsigned int   op2;
  unsigned int   behavior;
  OnigOptionType options;   /* default option */
  OnigMetaCharTableType meta_char_table;
} OnigSyntaxType;
""",
        ),
    ],
)
def test_get_symbol_definition(
    c_editor: SourceEditor, occurrence: SymbolOccurrence | LineInSymbolDefinition, expected_symbol: str
) -> None:
    """Test symbol retrieval"""
    assert c_editor.get_symbol_definition(occurrence).definition == expected_symbol


@pytest.mark.parametrize(
    "occurrence, function_names",
    [
        (
            SymbolOccurrence(
                name="xfree",  # struct
                file=TEST_C_SRC_PATH / "src/regexec.c",
                line=1502,
            ),
            [
                "onig_new_deluxe",
                "exec_end_call_list",
                "make_callout_func_list",
                "bbuf_free",
                "onig_regset_search",
                "onig_free_match_param_content",
                "onig_regset_new",
                "regset_search_body_position_lead",
                "parse_and_tune",
                "onig_new",
                "i_free_name_entry",
                "callout_name_entry",
                "clear_optimize_info",
                "onig_free_body",
                "onig_compile",
                "onig_regset_free",
                "unset_addr_list_end",
                "onig_free_match_param",
                "i_free_callout_tag_entry",
                "free_regex_ext",
                "history_tree_free",
                "onig_detect_can_be_slow_pattern",
                "node_free_body",
                "ops_free",
                "name_add",
                "onig_node_free",
                "onig_node_str_clear",
                "i_free_callout_name_entry",
                "prs_callout_of_contents",
                "onig_st_insert_strend",
                "onig_region_free",
                "onig_free",
                "onig_regset_search_with_param",
                "new_code_range",
                "onig_unicode_define_user_property",
                "free_callout_func_list",
                "onig_free_reg_callout_list",
                "clear_callout_args",
                "st_insert_callout_name_table",
                "ops_make_string_pool",
            ],
        )
    ],
)
def test_get_functions_referencing_symbol(
    c_editor: SourceEditor, occurrence: SymbolOccurrence, function_names: list[str]
) -> None:
    """Test enclosing function extraction"""

    functions = c_editor.get_functions_referencing_symbol(occurrence)
    assert {f.name for f in functions} == set(function_names)


@pytest.mark.parametrize(
    "diff, expected",
    [
        (
            """diff --git a/src/regparse.c b/src/regparse.c
index 24bcbaa..9acdd6e 100644
--- a/src/regparse.c
+++ b/src/regparse.c
@@ -793,8 +793,13 @@ onig_print_names(FILE* fp, regex_t* reg)
 #endif /* ONIG_DEBUG */

 static int
-i_free_name_entry(UChar* key, NameEntry* e, void* arg ARG_UNUSED)
+i_free_name_entry(st_data_t akey, st_data_t ae, st_data_t arg ARG_UNUSED)
 {
+  UChar* key;
+  NameEntry* e;
+
+  key = (UChar* )akey;
+  e = (NameEntry* )ae;
   xfree(e->name);
   if (IS_NOT_NULL(e->back_refs)) xfree(e->back_refs);
   xfree(key);
@@ -850,8 +855,14 @@ typedef struct {
 } INamesArg;

 static int
-i_names(UChar* key ARG_UNUSED, NameEntry* e, INamesArg* arg)
+i_names(st_data_t key ARG_UNUSED, st_data_t ae, st_data_t aarg)
 {
+  NameEntry* e;
+  INamesArg* arg;
+
+  e = (NameEntry* )ae;
+  arg = (INamesArg* )aarg;
+
   int r = (*(arg->func))(e->name,
                          e->name + e->name_len,
                          e->back_num,
@@ -883,9 +894,14 @@ onig_foreach_name(regex_t* reg,
 }

 static int
-i_renumber_name(UChar* key ARG_UNUSED, NameEntry* e, GroupNumMap* map)
+i_renumber_name(st_data_t key ARG_UNUSED, st_data_t ae, st_data_t amap)
 {
   int i;
+  NameEntry* e;
+  GroupNumMap* map;
+
+  e = (NameEntry* )ae;
+  map = (GroupNumMap* )amap;

   if (e->back_num > 1) {
     for (i = 0; i < e->back_num; i++) {
@@ -1374,9 +1390,14 @@ static int CalloutNameIDCounter;
 #ifdef USE_ST_LIBRARY

 static int
-i_free_callout_name_entry(st_callout_name_key* key, CalloutNameEntry* e,
-                          void* arg ARG_UNUSED)
+i_free_callout_name_entry(st_data_t akey, st_data_t ae, st_data_t arg ARG_UNUSED)
 {
+  st_callout_name_key* key;
+  CalloutNameEntry* e;
+
+  key = (st_callout_name_key* )akey;
+  e = (CalloutNameEntry* )ae;
+
   if (IS_NOT_NULL(e)) {
     xfree(e->name);
   }
@@ -1870,10 +1891,14 @@ typedef intptr_t   CalloutTagVal;
 #define CALLOUT_TAG_LIST_FLAG_TAG_EXIST     (1<<0)

 static int
-i_callout_callout_list_set(UChar* key, CalloutTagVal e, void* arg)
+i_callout_callout_list_set(st_data_t key ARG_UNUSED, st_data_t ae, st_data_t arg)
 {
   int num;
-  RegexExt* ext = (RegexExt* )arg;
+  CalloutTagVal e;
+  RegexExt* ext;
+
+  e   = (CalloutTagVal )ae;
+  ext = (RegexExt* )arg;

   num = (int )e - 1;
   ext->callout_list[num].flag |= CALLOUT_TAG_LIST_FLAG_TAG_EXIST;
@@ -1926,8 +1951,11 @@ onig_callout_tag_is_exist_at_callout_num(regex_t* reg, int callout_num)
 }

 static int
-i_free_callout_tag_entry(UChar* key, CalloutTagVal e, void* arg ARG_UNUSED)
+i_free_callout_tag_entry(st_data_t akey, st_data_t e ARG_UNUSED, st_data_t arg ARG_UNUSED)
 {
+  UChar* key;
+
+  key = (UChar* )akey;
   xfree(key);
   return ST_DELETE;
 }
diff --git a/src/st.h b/src/st.h
index 5efee8b..70798dc 100644
--- a/src/st.h
+++ b/src/st.h
@@ -34,13 +34,6 @@ enum st_retval {ST_CONTINUE, ST_STOP, ST_DELETE, ST_CHECK};
 #ifndef _
 # define _(args) args
 #endif
-#ifndef ANYARGS
-# ifdef __cplusplus
-#   define ANYARGS ...
-# else
-#   define ANYARGS
-# endif
-#endif

 st_table *st_init_table _((struct st_hash_type *));
 st_table *st_init_table_with_size _((struct st_hash_type *, int));
@@ -52,7 +45,7 @@ int st_delete _((st_table *, st_data_t *, st_data_t *));
 int st_delete_safe _((st_table *, st_data_t *, st_data_t *, st_data_t));
 int st_insert _((st_table *, st_data_t, st_data_t));
 int st_lookup _((st_table *, st_data_t, st_data_t *));
-int st_foreach _((st_table *, int (*)(ANYARGS), st_data_t));
+int st_foreach _((st_table *, int (*)(st_data_t, st_data_t, st_data_t), st_data_t));
 void st_add_direct _((st_table *, st_data_t, st_data_t));
 void st_free_table _((st_table *));
 void st_cleanup_safe _((st_table *, st_data_t));
""",
            [
                "i_free_name_entry",
                "key",
                "NameEntry",
                "xfree",
                "IS_NOT_NULL",
                "INamesArg",
                "i_names",
                "onig_foreach_name",
                "i_renumber_name",
                "st_data_t",
                "i_free_callout_name_entry",
                "st_callout_name_key",
                "i_callout_callout_list_set",
                "RegexExt",
                "CALLOUT_TAG_LIST_FLAG_TAG_EXIST",
                "onig_callout_tag_is_exist_at_callout_num",
                "i_free_callout_tag_entry",
                "ST_DELETE",
                "st_table",
                "st_init_table",
                "st_init_table_with_size",
                "st_delete_safe",
                "st_insert",
                "st_lookup",
                "st_foreach",
                "st_add_direct",
                "st_free_table",
                "st_cleanup_safe",
            ],
        )
    ],
)
def test_get_symbols_occurring_in_diff(c_editor: SourceEditor, diff: str, expected: list[str]) -> None:
    """Test get_changed_files_from_diff"""

    # checkout commit from which diff is from
    Repo(TEST_C_SRC_PATH).git.checkout("5f1408dee", force=True)

    changed_symbol_names = [x.name for x in c_editor.get_symbols_occurring_in_diff(diff, TEST_C_SRC_PATH)]

    for name in expected:
        assert name in changed_symbol_names


@pytest.mark.parametrize(
    "file, text, start, end",
    [
        (
            TEST_C_SRC_PATH / "src/ascii.c",
            """ * These lines
 * are a replace-
 * ment
 * .
""",
            19,
            22,
        ),
    ],
)
def test_replace_lines(c_editor: SourceEditor, file: Path, text: str, start: int, end: int) -> None:
    """Test replace_lines"""

    c_editor.replace_lines(file, text, start, end)

    assert get_lines(file.read_text().splitlines(), start, end) == text


@pytest.mark.parametrize(
    "file, text, line",
    [
        (TEST_C_SRC_PATH / "src/ascii.c", "/* Inserted comment */\n", 29),
    ],
)
def test_insert_above_line(c_editor: SourceEditor, file: Path, text: str, line: int) -> None:
    """Test insert lines"""

    c_editor.insert_above_line(file, text, line)

    assert get_lines(file.read_text().splitlines(), line, line) == text


@pytest.mark.parametrize(
    "occurrence, new_definition",
    [
        (
            SymbolOccurrence(
                name="onig_get_used_stack_size_in_callout",  # function
                file=TEST_C_SRC_PATH / "sample/callout.c",
                line=44,
            ),
            """extern int
onig_get_used_stack_size_in_callout(OnigCalloutArgs* a, int* used_num, int* used_bytes)
{
  return 0;
}
""",
        )
    ],
)
def test_replace_symbol_definition(c_editor: SourceEditor, occurrence: SymbolOccurrence, new_definition: str) -> None:
    """Test replace symbol definition"""

    c_editor.replace_symbol_definition(occurrence, new_definition)

    d = c_editor.get_symbol_definition(occurrence)

    assert d.definition == new_definition


def test_unicode_handling(c_editor: SourceEditor) -> None:
    """Test unicode handling"""

    evil_encoded = "  // Между прочим, наша реализация LSP в состоянии работать с нетривиальным Юникодом!\n"

    c_editor.insert_above_line(TEST_C_SRC_PATH / "src/regexec.c", evil_encoded, 6383)

    occurrence = SymbolOccurrence(
        name="onig_get_used_stack_size_in_callout",  # function
        file=TEST_C_SRC_PATH / "sample/callout.c",
        line=44,
    )

    symbol = c_editor.get_symbol_definition(occurrence)

    assert evil_encoded in symbol.definition
