"""Generate reproducer too with code path and coverage information."""

from pathlib import Path

from pydantic import Field

from crs.agents.pov_builder.coverage.calltree import CallTree
from crs.agents.pov_builder.coverage.coverage import SUPPORTED_LANGUAGES
from crs.agents.pov_builder.coverage.coverage import produce_coverage_artifacts
from crs.agents.pov_builder.coverage.utils import build_coverage_feedback_prompt
from crs.agents.tools.generate_reproducer import GenerateReproducer
from crs.aixcc.scripts import RunResult
from crs.base.settings import OSS_FUZZ_LOCATION
from crs.base.settings import POV_TARGET_FUNCTION
from crs.logger import CRS_LOGGER

log = CRS_LOGGER.getChild(__name__)
# pylint: disable=line-too-long


class GenerateReproducerCodePath(GenerateReproducer):
    """
    Adds code path and coverage information to the GenerateReproducer tool output.
    """

    call_tree: CallTree = Field(exclude=True)

    # pylint: disable=no-member
    def handle_no_sanitizer_triggered(
        self,
        run_result: RunResult,
        pov: str | bytes,
    ) -> str:
        msg = super().handle_no_sanitizer_triggered(run_result, pov)

        # TODO: do we kill C++ functionality with this?
        # are we in a C program?
        languages = SUPPORTED_LANGUAGES
        if (lang := self.ctx.proj_yaml_model.language.strip().lower()) in languages:
            # returned cov can be None if no coverage build exists
            if self.reproducer_file_work and self.state:
                coverage_file_path = produce_coverage_artifacts(self.ctx, self.reproducer_file_work)
                if coverage_file_path:
                    self.log_coverage_info_with_path()
                    cov_feedback = build_coverage_feedback_prompt(self.ctx, self.state, coverage_file_path)
                    msg += f"\n{cov_feedback}\n"
        else:
            log.warning(
                f"Cannot gather coverage information because language '{lang}' is not supported. supported langs: {languages}"
            )

        log.info(f"New prompt: {msg}")
        return msg

    def log_coverage_info_with_path(self) -> None:
        """Logs coverage information for the last PoV"""
        if not self.reproducer_file_work:
            log.warning("This should not happen.")
            return

        covered_functions: list[str] = self.extract_covered_functions()

        if not covered_functions:
            log.info(f"No reached functions found for PoV {self.reproducer_file_work}")
            return

        log.info(f"Reached functions for PoV {self.reproducer_file_work}: {covered_functions}")

        identifier = self.reproducer_file_work.stem.split("-")[-1]
        target_func = POV_TARGET_FUNCTION.get()
        harness_name = self.ctx.harness_id
        if not target_func:
            target_func = ""
        cov_log_path = Path(
            ((target_func + "_") if target_func else "") + harness_name + "_" + identifier + "_reached_functions.log"
        )

        with open(cov_log_path, "w", encoding="utf-8") as f:
            f.write(",".join(covered_functions) + "\n")

        log.info(f"Wrote covered functions info to {cov_log_path}")

        paths = self.call_tree.get_all_paths("LLVMFuzzerTestOneInput", target_func)

        cov_x_tree_log_path = Path(
            ((target_func + "_") if target_func else "") + harness_name + "_" + identifier + "_reached_funcs_path.log"
        )

        # for each function in the path write the tuple (func: str, reached: bool)
        with open(cov_x_tree_log_path, "w", encoding="utf-8") as f:
            path_repr = []
            for path in paths:
                path_repr.append([(func, func in covered_functions) for func in path])
            f.write(str(path_repr) + "\n")

        log.info(f"Wrote reached paths info to {cov_x_tree_log_path}")

    def extract_covered_functions(self) -> list[str]:
        """Extracts covered functions from the coverage report file."""
        cov_dir_name: Path = OSS_FUZZ_LOCATION.get() / "build" / "out" / self.ctx.proj_yaml_model.cp_name
        covered_functions: list[str] = []
        if cov_dir_name.is_dir():
            cov_file_path = cov_dir_name / "textcov_reports/" / (self.ctx.harness_id + ".covreport")
            if cov_file_path.exists():
                with open(cov_file_path, encoding="utf-8") as f:
                    raw_cov = f.read()
                for line in raw_cov.splitlines():
                    if line.endswith(":") and not "|" in line:
                        covered_functions.append(line[:-1])
                log.info(f"Reached functions for PoV {self.reproducer_file_work}: {covered_functions}")
        return covered_functions
