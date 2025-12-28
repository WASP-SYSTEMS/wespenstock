"""Language specific configuration."""

from __future__ import annotations

from dataclasses import dataclass

from crs.base.context import CrsContext
from crscommon.crash_report.base import BaseCrashReport
from crscommon.crash_report.c import AsanReport
from crscommon.crash_report.python import PythonReport
from crscommon.editor.editor import SourceEditor
from crscommon.editor.jedi_interface import JediInterface
from crscommon.editor.language_helper import CLanguageHelper
from crscommon.editor.lsp.clangd_client import ClangdClient
from crscommon.editor.lsp.compilation_db_preparer import CompilationDbPreparer
from crscommon.editor.lsp_interface import LspInterface


@dataclass
class LangConfig:
    """Language config for specific language."""

    crash_report_t: type[BaseCrashReport]
    editor: SourceEditor

    @staticmethod
    def get(ctx: CrsContext) -> LangConfig:
        """Get language configuration"""

        match ctx.proj_yaml_model.language:
            case "c":
                file_extensions = [".c", ".cc", ".cpp", ".h"]
                clangd = ClangdClient(ctx.cp_path_abs / ctx.src_path_rel, CompilationDbPreparer(ctx.cp_path_abs))
                editor = SourceEditor(
                    LspInterface(clangd, CLanguageHelper(ctx.cp_path_abs / ctx.src_path_rel), file_extensions),
                    file_extensions,
                )

                return LangConfig(crash_report_t=AsanReport, editor=editor)

            case "python":
                editor = SourceEditor(JediInterface(ctx.cp_path_abs / ctx.src_path_rel), [".py"])
                return LangConfig(crash_report_t=PythonReport, editor=editor)

        raise NotImplementedError(f"Language {ctx.proj_yaml_model.language} not supported!")
