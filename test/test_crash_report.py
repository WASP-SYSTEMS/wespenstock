"""Test crash report parsing."""

from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel

from crs.base.context import CrsContext
from crscommon.crash_report.base import BaseCrashReport
from crscommon.crash_report.base import StackLevel
from crscommon.crash_report.c import AsanReport
from crscommon.editor.editor import SourceEditor

from .setup_test_env import TEST_C_SRC_PATH


class DummyCtx(BaseModel):
    """Dummy context for tests"""

    cp_path_abs: Path
    src_path_rel: Path


# pylint: disable=line-too-long


@pytest.mark.parametrize(
    "report, exptected_stack_levels",
    [
        (
            AsanReport(
                """==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x5030000380f1 at pc 0x5793e842795e bp 0x7ffcfef85ce0 sp 0x7ffcfef85cd8
Indirect leak of 40 byte(s) in 1 object(s) allocated from:
    #0 0x7d5d5f0fd891 in malloc /src/debug/gcc/gcc/libsanitizer/asan/asan_malloc_linux.cpp:69
    #1 0x7d5d5ef24779 in onig_region_resize /src/oniguruma/src/regexec.c:927
    #2 0x7d5d5ef249d1 in onig_region_resize_clear /src/oniguruma/src/regexec.c:952
    #3 0x7d5d5ef453d1 in onig_match_with_param /src/oniguruma/src/regexec.c:5191
    #4 0x55f4a5e14ca9 in match /src/oniguruma/sample/callback_each_match.c:121
    #5 0x55f4a5e150bb in main /src/oniguruma/sample/callback_each_match.c:162
    #6 0x7d5d5ecbae07  (/usr/lib/libc.so.6+0x25e07) (BuildId: 98b3d8e0b8c534c769cb871c438b4f8f3a8e4bf3)
    #7 0x7d5d5ecbaecb in __libc_start_main (/usr/lib/libc.so.6+0x25ecb) (BuildId: 98b3d8e0b8c534c769cb871c438b4f8f3a8e4bf3)
    #8 0x55f4a5e14204 in _start (/src/oniguruma/sample/.libs/callback_each_match+0x2204) (BuildId: 47cdfe2098aa6a45e108e54ec5d4641f7b019fb2)
==39==ABORTING
""",
                cast(CrsContext, DummyCtx(cp_path_abs=Path(), src_path_rel=TEST_C_SRC_PATH)),
            ),
            [
                StackLevel(
                    file=Path("test/data/oniguruma/src/regexec.c").absolute(), function="onig_region_resize", line=927
                ),
                StackLevel(
                    file=Path("test/data/oniguruma/src/regexec.c").absolute(),
                    function="onig_region_resize_clear",
                    line=952,
                ),
                StackLevel(
                    file=Path("test/data/oniguruma/src/regexec.c").absolute(),
                    function="onig_match_with_param",
                    line=5191,
                ),
                StackLevel(
                    file=Path("test/data/oniguruma/sample/callback_each_match.c").absolute(), function="match", line=121
                ),
                StackLevel(
                    file=Path("test/data/oniguruma/sample/callback_each_match.c").absolute(), function="main", line=162
                ),
            ],
        ),
    ],
)
def test_crash_report(
    c_editor: SourceEditor, report: BaseCrashReport, exptected_stack_levels: list[StackLevel]
) -> None:
    """Test asan report parser"""

    assert report.get_stacktrace().trace == exptected_stack_levels

    functions = report.get_symbol_descriptions(c_editor)

    print([lvl.function for lvl in exptected_stack_levels])

    assert [lvl.function for lvl in exptected_stack_levels] == [f.name for f in functions]
