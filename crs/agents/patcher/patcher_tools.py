"""Tools for patcher."""

from pathlib import Path

from pydantic import BaseModel

from crs.agents.constants import DISPOSITION_PATCH_SUCCESS
from crs.agents.tools.named_base_tool import NamedBaseTool
from crs.base.context import CrsContext
from crs.logger import CRS_LOGGER
from crscommon.editor.editor import SourceEditor
from crscommon.editor.symbol_history import SymbolHistory
from crscommon.lang_config.config import LangConfig

log = CRS_LOGGER.getChild(__name__)


def create_verify_tool(
    blob: Path,
    symbol_history: SymbolHistory,
    editor: SourceEditor,
    lang_config: LangConfig,
    ctx: CrsContext,
) -> NamedBaseTool:
    """Factor for patch verifier tool."""

    class VerifyPatch(NamedBaseTool):
        """Runs pov and tests and verifies if patch works."""

        NAME = "VerifyPatch"

        class Input(BaseModel):
            """Input"""

        name: str = "VerifyPatch"
        description: str = "Execute the program to verify that the vulnerability was fixed."
        args_schema: type[BaseModel] = Input

        # pylint: disable=arguments-differ
        def _run(self) -> str | dict[str, str]:

            log.info("Verifying patch...")

            # first rebuild the docker (important that we use local docker image!)
            output = ctx.comp_env.build()
            if output.return_code != 0:
                log.error("Compilation error")
                # TODO: is it always stderr?? detect where the error is
                return f"Compilation error:\n```\n{output.stderr}\n```\n"

            # execute pov
            harness = ctx.proj_yaml_model.harnesses[ctx.harness_id].name
            output = ctx.comp_env.run_pov(blob, harness)

            # check if pov crashed
            if output.scan_for_sanitizer():
                assert output.crash_output is not None
                # add (new) symbols from crash report to history
                parsed_report = lang_config.crash_report_t(output.crash_output, ctx)
                for line_in_symbol in parsed_report.get_symbol_descriptions(editor):
                    symbol_history.update(line_in_symbol)
            else:
                # run tests if program did not crash
                test_out = ctx.comp_env.run_tests()
                # 0 indicates all tests passed. 1 indicates an internal error,
                # which has nothing to do with out patch
                if test_out.return_code == 0:
                    log.info("Success: Bug is fixed and tests pass")
                    return {
                        "content": "The program did not crash and all tests passed. Hooray!",
                        "disposition": DISPOSITION_PATCH_SUCCESS,
                    }

                if test_out.return_code is None or 1 < test_out.return_code < 125:
                    log.warning(f"Tests failed (return code {test_out.return_code})")
                    return (
                        "The program did not crash, but"
                        f"the tests failed:\n ```json\n{test_out.model_dump_json()}\n```\n"
                    )

            log.info("Something went not quite right")
            # something else happened
            if output.return_code is None:
                answer = "Program run timed out. Details:"
            elif output.return_code == 0:
                answer = "Program exited successfully. Details:"
            else:
                answer = f"Program exited with code {output.return_code}. Details:"
            answer += f"\n```json\n{output.model_dump_json()}\n```\n"
            return answer

    return VerifyPatch()
