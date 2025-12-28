"""Generate reproducer tool.'"""

import base64
import datetime
import uuid
from pathlib import Path
from typing import Annotated
from typing import ClassVar

from langgraph.prebuilt.tool_node import InjectedState
from pydantic import BaseModel
from pydantic import Field

from crs.agents.constants import DISPOSITION_TASK_DONE
from crs.agents.tools.named_base_tool import NamedBaseTool
from crs.aixcc.scripts import RunResult
from crs.aixcc.scripts import invoke_run_sh
from crs.base.base_state import BaseChatState
from crs.base.context import CrsContext
from crs.base.context import ReproducerDescription
from crs.base.settings import POV_MAX_LEN_BYTES
from crs.base.settings import POV_REPR_CUT_ON_MAX_LEN_TO
from crs.base.settings import POV_REPR_MAX_LEN
from crs.logger import CRS_LOGGER
from crscommon.logging.settings import OUTPUT_DIR
from crscommon.sandbox import run_in_sandbox

log = CRS_LOGGER.getChild(__name__)
# pylint: disable=line-too-long


def reproducer_timestamp() -> str:
    """Get formatted current timestamp for reproducer."""
    return datetime.datetime.now().replace(microsecond=0).isoformat()


class GenerateReproducer(NamedBaseTool):
    """
    Generate a PoV blob trough python code given by the llm using exec().
    Execute it and validating a crash.

    The result of the tool execution is available in tool.pov_description
    if the generated input was successful it will be a ReproducerDescription else it will stay None.
    We are aware that this is a little hacky, compared to just returning the value, but yeah. Framework things :)

    If something went wrong the LLM will be informed about that and tool.pov_description stays None
    """

    # pylint: disable=missing-class-docstring
    class Args(BaseModel):
        python_code: str = Field(
            description=(
                "Expects a snippet of Python code that generates a reproducer in the form of bytes or a string. "
                "The output of this code is crucial for validation of the crash and must be assigned to a variable "
                "named `reproducer` in the last line. This allows for the automated evaluation of the "
                "reproducer against the provided harness to validate the crash."
            )
        )
        state: Annotated[BaseChatState, InjectedState] = Field(
            description="State of the conversation (to be injected by runtime code)"
        )

    # (...) attributes annotated with typing.ClassVar will be automatically excluded from the model. ~pydantic docs
    NAME: ClassVar[str] = "generate_crashing_reproducer"
    """same as .name but as a class variable, can be used to access the tool name without having an instance of it."""

    name: str = NAME
    description: str = (
        "This tool is the main interface for the creation of Python code snippets that output a reproducer "
        "in the form of a sequence of bytes or a string."
        "The generated reproducer is automatically tested against the provided harness for potential crashes. "
        "The tool requires Python code, in which the final line assigns the generated "
        "reproducer to a variable named `reproducer`.\n\n"
        "Usage Instructions:\n"
        "- Provide valid Python code as input to the tool.\n"
        "- Ensure the code produces a reproducer of the desired type (bytes or string).\n"
        "- Assign the reproducer to the variable `reproducer` in the last line of the code.\n"
        "- Do NOT attempt to run the reproducer yourself. The tool will handle this for you.\n"
    )

    args_schema: type[BaseModel] = Args

    # needed as implicit return from the tool, set after tool is run
    pov_description: ReproducerDescription | None = Field(default=None, exclude=True)

    # things we need to run the tool that we don't wanna serialize

    ctx: CrsContext = Field(exclude=True)
    harness_name: str = Field(exclude=True)
    reproducer_file_work: Path | None = Field(default=None, exclude=True)
    state: BaseChatState | None = Field(default=None, exclude=True)

    # pylint: disable=no-member
    # pylint: disable=arguments-differ
    def _run(self, python_code: str, state: BaseChatState) -> str | dict[str, str]:
        self.state = state
        # exec the code of the llm
        try:
            pov = self._llm_sandbox(python_code)
        # pylint: disable=broad-exception-caught
        except Exception as e:
            return f"There was an exception executing the python code: {type(e).__name__}: {e}"

        pov_length = len(pov) if isinstance(pov, bytes) else len(pov.encode("utf-8"))
        if pov_length > POV_MAX_LEN_BYTES.get():
            log.info(
                f"The reproducer is too long and therefore discarded. Length: {pov_length} B, limit: {POV_MAX_LEN_BYTES} B, "
                f"Reproducer: {pov[:1024]!r}"
            )
            return f"The reproducer is too long and therefore discarded. Length: {pov_length} bytes, maximum allowed: {POV_MAX_LEN_BYTES} bytes"

        # write the blob to work (which is mounted in a docker container where the reproducer is validated)
        try:
            self.reproducer_file_work = self.ctx.cp_path_abs / "work" / f"pov-{uuid.uuid4()}.blob"
            self._write_reproducer(pov, self.reproducer_file_work)
        # pylint: disable=broad-exception-caught
        except Exception as e:
            err_msg = f"There was an exception writing the reproducer: {type(e).__name__}: {e}"
            log.warning(err_msg)
            return err_msg

        run_result = invoke_run_sh(
            ["run_pov", self.reproducer_file_work.absolute().as_posix(), self.harness_name], self.ctx.cp_path_abs
        )

        # did we trigger a sanitizer?
        if run_result.scan_for_sanitizer():
            return self.handle_sanitizer_triggered(run_result, pov)

        return self.handle_no_sanitizer_triggered(run_result, pov)

    def handle_sanitizer_triggered(
        self,
        run_result: RunResult,
        pov: str | bytes,
    ) -> dict[str, str]:
        """Handle case where a sanitizer was triggered."""

        assert run_result.sanitizer is not None
        assert run_result.crash_output is not None

        # map harness name back to id (we've got only the sanitizer id until now)
        harness_id = self.ctx.proj_yaml_model.harness_id_by_name(self.harness_name)

        # prepare for returning
        reproducer_file_local = OUTPUT_DIR.get() / f"pov_success_{reproducer_timestamp()}"

        log.info(
            f"The reproducer triggered successfully. The reproducer file can be found at: "
            f"'{reproducer_file_local.as_posix()}'"
        )
        # the file will still keep the name pov_... bc for legacy reasons. all our eval scripts rely on that.
        self._write_reproducer(pov, reproducer_file_local)

        self.pov_description = ReproducerDescription(
            blob=base64.b64encode(pov if isinstance(pov, bytes) else pov.encode("utf-8")),
            commit_hash=self.ctx.commit_hash,
            crash_report=run_result.crash_output,
            sanitizer_name=run_result.sanitizer,
            harness_id=harness_id,
            path=reproducer_file_local,
        )

        return {
            "content": "The vulnerability was triggered successfully. Hooray!",
            "disposition": DISPOSITION_TASK_DONE,
        }

    def handle_no_sanitizer_triggered(
        self,
        run_result: RunResult,
        pov: str | bytes,
    ) -> str:
        """Handle case where no sanitizier was triggered."""
        # ensure that we don't dump too massive PoVs into the LLM
        msg = self.format_failed_pov_feedback(pov)
        if run_result.return_code is None:
            msg += "\nThe program timed out. Maybe some details are off and size isn't the issue?"

        reproducer_file_local = OUTPUT_DIR.get() / f"pov_failed_{reproducer_timestamp()}"
        self._write_reproducer(pov, reproducer_file_local)

        log.info(f"New prompt: {msg}")
        return msg

    @staticmethod
    def format_failed_pov_feedback(
        pov: str | bytes, failed_reason: str = "The reproducer failed to trigger the vulnerability."
    ) -> str:
        """Given a failed PoV blob, format or abbreviate it into a response to the LLM."""

        pov_repr = repr(pov)
        if len(pov_repr) <= POV_REPR_MAX_LEN.get() or len(pov_repr) <= POV_REPR_CUT_ON_MAX_LEN_TO.get():
            return f"{failed_reason} The __repr__ of the reproducer: {pov_repr}"

        prefix_len, suffix_len = divmod(POV_REPR_CUT_ON_MAX_LEN_TO.get(), 2)
        suffix_len += prefix_len
        assert len(pov_repr) >= prefix_len + suffix_len
        # What the code below really needs is suffix_len > 0, but ridiculously short abbreviations are not good, either.
        assert prefix_len >= 2 and suffix_len >= 2

        pov_repr = f"{pov_repr[:prefix_len]}...({len(pov_repr) - prefix_len - suffix_len} more chars)...{pov_repr[-suffix_len:]}"

        log.debug(f"Failed reproducer was too long; truncated before passing to the LLM: {pov_repr}")
        return (
            f"{failed_reason} The __repr__ of the reproducer is too long to be displayed in full. "
            f"Maybe some details are off and size isn't the issue? Here's a summary of the generated reproducer: {pov_repr}"
        )

    @staticmethod
    def _llm_sandbox(code: str) -> bytes | str:
        """The sandbox for the llm code so the normal function scope will not be trashed by the llm code"""
        try:
            result: object = run_in_sandbox(code, "reproducer", str | bytes)
            assert isinstance(result, (str, bytes))
            return result
        except TimeoutError as exc:
            raise TimeoutError(
                "Creation of the reproducer took too long and was terminated. "
                "Are you sure that the reproducer needs to be that long?"
            ) from exc

    @staticmethod
    def _write_reproducer(reproducer: bytes | str, reproducer_path: Path) -> None:
        """write the given reproducer to the given path"""

        if isinstance(reproducer, bytes):
            reproducer_path.write_bytes(reproducer)
        elif isinstance(reproducer, str):
            reproducer_path.write_text(reproducer, encoding="utf-8")
        else:
            raise AssertionError(
                f"This should not happen! An ill-typed reproducer of type {type(reproducer)} snuck in!"
            )

        log.info(f"Suggested reproducer (repr): {repr(reproducer)}")
        log.info(f"Wrote reproducer to: {reproducer_path.as_posix()}")
