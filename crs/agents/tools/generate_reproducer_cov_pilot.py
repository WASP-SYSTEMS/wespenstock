"""generate reproducer tool for 'pov builder'"""

import base64
import datetime
import uuid
from pathlib import Path

from crs.agents.constants import DISPOSITION_TASK_DONE
from crs.agents.pov_builder.coverage.calltree import CallTree
from crs.agents.pov_builder.coverage.coverage import SUPPORTED_LANGUAGES
from crs.agents.pov_builder.coverage.coverage import produce_coverage_artifacts
from crs.agents.pov_builder.coverage.coverage_report import CoverageReport
from crs.agents.pov_builder.coverage.utils import build_coverage_feedback_prompt
from crs.agents.tools.generate_reproducer import GenerateReproducer
from crs.aixcc.scripts import RunResult
from crs.aixcc.scripts import invoke_run_sh
from crs.base.base_state import BaseChatState
from crs.base.context import ReproducerDescription
from crs.base.settings import CALL_TREE_ANALYSIS_DIR
from crs.base.settings import COV_PILOT_MODE
from crs.base.settings import POV_MAX_LEN_BYTES
from crs.base.settings import POV_REPR_CUT_ON_MAX_LEN_TO
from crs.base.settings import POV_REPR_MAX_LEN
from crs.base.settings import POV_TARGET_FUNCTION
from crs.base.settings import POV_USE_COVERAGE
from crs.logger import CRS_LOGGER
from crscommon.logging.settings import OUTPUT_DIR

log = CRS_LOGGER.getChild(__name__)
# pylint: disable=line-too-long


class GenerateReproducerCovPilot(GenerateReproducer):
    """
    Generate a PoV blob trough python code given by the llm using exec().
    Execute it and validating a crash.

    The result of the tool execution is available in tool.pov_description
     if the generated input was successful it will be a ReproducerDescription else it will stay None.
    We are aware that this is a little hacky, compared to just returning the value, but yeah. Framework things :)

    If something went wrong the LLM will be informed about that and tool.pov_description stays None
    """

    # TODO remove unused-argument when code-path branch was merged.
    # pylint: disable=arguments-differ
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    # pylint: disable=unused-argument
    # pylint: disable=no-member
    def _run(self, python_code: str, state: BaseChatState) -> str | dict[str, str]:
        """overridden run method of tool"""
        self.state = state
        target_function = POV_TARGET_FUNCTION.get()

        # the identifier used for the local pov
        pov_local_identifier = datetime.datetime.now().replace(microsecond=0).isoformat()
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
                f"prefix: {pov[:1024]!r}"
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

        reproducer_folder_local = OUTPUT_DIR.get()
        cov_pilot_mode = COV_PILOT_MODE.get()

        # did we crash? (and we're in verifier mode)
        if run_result.scan_for_sanitizer():
            msg = self.handle_no_sanitizer_trigger_case(run_result, pov, reproducer_folder_local, pov_local_identifier)

            return msg

        # from here we assume we crashed (and we're in CovPilot mode)
        if run_result.crash_output is not None and not cov_pilot_mode:

            # for the mypy
            assert run_result.sanitizer is not None
            assert run_result.crash_output is not None

            # map harness name back to id (we've got only the sanitizer id until now)
            harness_id = self.ctx.proj_yaml_model.harness_id_by_name(self.harness_name)

            # prepare for returning
            reproducer_file_local = (
                reproducer_folder_local / f"pov_success_{datetime.datetime.now().replace(microsecond=0).isoformat()}"
            )

            log.info(
                f"The reproducer triggered successfully. The reproducer file can be found at: "
                f"'{reproducer_file_local.as_posix()}'"
            )
            # the file will still keep the name pov_... bc for legacy reasons. all our eval scrips rely on that.
            # also that the thing is just dropped into the current dir. rip us.
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

        if cov_pilot_mode:
            if target_function is None:
                raise ValueError("CovPilot mode requires a target function to be set.")

            call_tree = CallTree.from_project(
                self.ctx.proj_yaml_model.cp_name,
                self.ctx.cp_path_abs,
                self.ctx.cp_path_abs / self.ctx.src_path_rel,
                CALL_TREE_ANALYSIS_DIR.get(),
                self.ctx.harness_id,
            )
            # append the functions that were searched for in the call tree analysis

            current_path = call_tree.get_all_paths("LLVMFuzzerTestOneInput", target_function)[
                0
            ]  # TODO: this is bad. the same happens also in the pov (verifier) agent

            cov_report = CoverageReport.generate_report_for_cp(
                self.ctx.cp_name, self.ctx.harness_id, pov_local_identifier
            )
            target_function_cov = cov_report.get(target_function)

            if target_function_cov is None or target_function_cov.num_covered_lines == 0:

                fn_coverage_msg = self.handle_target_fn_not_reached_case(
                    pov, cov_report, current_path, target_function, reproducer_folder_local
                )

                return fn_coverage_msg

            # we assume we hit the function
            # prepare for returning
            reproducer_file_local = (
                reproducer_folder_local
                / "cov_pilot"
                / f"cov_pilot_success_{target_function}_{datetime.datetime.now().replace(microsecond=0).isoformat()}"
            )

            reproducer_file_local.parent.mkdir(parents=True, exist_ok=True)

            log.info(
                f"The reproducer reached function {target_function}. The reproducer file can be found at: "
                f"'{reproducer_file_local.as_posix()}'"
            )

            self._write_reproducer(pov, reproducer_file_local)

            self.pov_description = ReproducerDescription(
                blob=base64.b64encode(pov if isinstance(pov, bytes) else pov.encode("utf-8")),
                commit_hash="not applicable",
                crash_report=str(cov_report),
                sanitizer_name="not applicable",
                harness_id=self.harness_name,
                path=reproducer_file_local,
            )

            return {
                "content": f"The function {target_function} was successfully reached. Hooray!",
                "disposition": DISPOSITION_TASK_DONE,
            }

        raise RuntimeError(
            f"It should not be possible to reach this state. Yet somehow we did. {cov_pilot_mode=}, {run_result=}"
        )

    def handle_target_fn_not_reached_case(
        self,
        pov: str | bytes,
        cov_report: CoverageReport,
        current_code_path: list[str],
        target_function_name: str,
        reproducer_folder_local: Path,
    ) -> str:
        """CovPilot mode. handle case that the target function is not reached"""
        fn_coverage_msg = (
            f"The target function {target_function_name} was not reached. "
            f"Here is more coverage info about the functions on the code path:"
        )
        for fn_name in current_code_path[:-1]:
            fn_coverage = cov_report[fn_name]
            if fn_coverage is not None:
                fn_coverage_msg += (
                    f"Here is the coverage for function {fn_coverage.name}:"
                    f"\n```{fn_coverage.get_code_with_coverage()}```\n"
                )
            else:
                fn_coverage_msg += f"No coverage data found for {fn_name}. It was most likely never reached."
        reproducer_file_local = (
            reproducer_folder_local
            / "cov_pilot"
            / f"cov_pilot_fail_{target_function_name}_{datetime.datetime.now().replace(microsecond=0).isoformat()}"
        )

        reproducer_file_local.parent.mkdir(parents=True, exist_ok=True)

        log.info(
            f"The reproducer didn't reach function {target_function_name}. The reproducer file can be found at: "
            f"'{reproducer_file_local.as_posix()}'"
        )
        self._write_reproducer(pov, reproducer_file_local)
        return fn_coverage_msg

    def handle_no_sanitizer_trigger_case(
        self,
        run_result: RunResult,
        pov: str | bytes,
        reproducer_folder_local: Path,
        pov_local_identifier: str,
    ) -> str:
        """Case where we're in verifier mode and no crash was triggered"""
        # ensure that we don't dump too massive PoVs into the LLM
        msg = self.format_failed_pov_feedback(pov)
        if run_result.return_code is None:
            msg += "\nThe program timed out. Maybe some details are off and size isn't the issue?"

        reproducer_path = Path(f"pov_failed_{pov_local_identifier}")
        self._write_reproducer(pov, reproducer_path)  # TODO: why do we write this one?

        log.info(f"New prompt: {msg}")

        reproducer_file_local = reproducer_folder_local / f"pov_failed_{pov_local_identifier}"
        self._write_reproducer(pov, reproducer_file_local)

        # TODO: do we kill C++ functionality with this?
        # are we in a C program?
        coverage_file_path: Path | None = None
        if (lang := self.ctx.proj_yaml_model.language.strip().lower()) in SUPPORTED_LANGUAGES and (
            POV_USE_COVERAGE.get() or POV_TARGET_FUNCTION.get()
        ):
            # returned cov can be None if no coverage build exists
            if self.reproducer_file_work and self.state:
                coverage_file_path = produce_coverage_artifacts(self.ctx, self.reproducer_file_work)
                if POV_USE_COVERAGE.get() and coverage_file_path is not None and self.state:
                    # TODO: log coverage info related to call tree
                    cov_feedback = build_coverage_feedback_prompt(self.ctx, self.state, coverage_file_path)
                    msg += f"\n{cov_feedback}\n"
        else:
            log.info(
                f"Can't gather coverage because language '{lang}' is not supported. supported langs: {SUPPORTED_LANGUAGES}"
            )

        log.info(f"New prompt: {msg}")

        return msg

    @staticmethod
    def format_failed_pov_feedback(pov: str | bytes, failed_reason: str = "DONT USE ME") -> str:
        """Given a failed PoV blob, format or abbreviate it into a response to the LLM."""

        pov_repr = repr(pov)
        if len(pov_repr) <= POV_REPR_MAX_LEN.get() or len(pov_repr) <= POV_REPR_CUT_ON_MAX_LEN_TO.get():
            return f"The reproducer failed to trigger the vulnerability. The __repr__ of the reproducer: {pov_repr}"

        prefix_len, suffix_len = divmod(POV_REPR_CUT_ON_MAX_LEN_TO.get(), 2)
        suffix_len += prefix_len
        assert len(pov_repr) >= prefix_len + suffix_len
        # What the code below really needs is suffix_len > 0, but ridiculously short abbreviations are not good, either.
        assert prefix_len >= 2 and suffix_len >= 2

        pov_repr = f"{pov_repr[:prefix_len]}...({len(pov_repr) - prefix_len - suffix_len} more chars)...{pov_repr[-suffix_len:]}"

        log.debug(f"Failed reproducer was too long; truncated before passing to the LLM: {pov_repr}")
        return (
            "The reproducer failed to trigger the vulnerability. The __repr__ of the reproducer is too long to be displayed in full. "
            f"Maybe some details are off and size isn't the issue? Here's a summary of the generated reproducer: {pov_repr}"
        )
