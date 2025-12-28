"""We build the initial prompt for the Verifier's CodePath mode here"""

from crs.agents.pov_builder.verifier_prompt import VerifierPrompt
from crs.base.context import CrsContext
from crs.base.context import Vulnerability
from crs.base.settings import POV_USE_COVERAGE


class CodePathPrompt(VerifierPrompt):
    """Prompts for the CodePath Verifier"""

    @staticmethod
    def prompt_coverage_explanation() -> str:
        """Returns coverage explanation prompt"""
        return """Coverage reports for provided reproducers may be provided.
    The format of the coverage reports is as follows:

    It starts with the path to the file that the coverage info was gathered from.
    /src/program/src/lib/some_name.c:

    Then the info of which lines in the code we are looking at
    @Lines 164-165@

    Then the coverage info itself:
    164| 3 |  if (status != SUCCESS && strs != NULL) {
    ------------------
    |  Branch (1:7): [True: 2, False: 1]
    |  Branch (1:28): [True: 1, False: 2]
    ------------------
    165| 10 |    for (i = 0; i < num_strs; i++) {

    This means that line 164 was executed twice.
    The branch info shows that the if statement was evaluated twice, with the first expression being evaluated to twice once and false once.
    The second expression was evaluated to true once and false twice.
    The line 165 was executed 10 times.

    There may be multiple areas of code that are covered in the same file.
    The next area will be denoted by @Lines XXX-XXX@ as well.
    The next file will be denoted by a new path.

    This can continue for multiple files and multiple areas of code for each file."""

    @staticmethod
    def get_prompt(
        ctx: CrsContext,
        vulnerability: Vulnerability,
        harness_name: str,
        harness_code: str,
        target_code: str,
        type_of_target: str = "diff",
        standalone_pov_builder_prompt: bool = False,
        additional_prompting: str = "",
    ) -> str:
        """get verifier prompt for codepath"""
        if POV_USE_COVERAGE.get():
            additional_prompting += CodePathPrompt.prompt_coverage_explanation()

        prompt = super(CodePathPrompt, CodePathPrompt).get_prompt(
            ctx,
            vulnerability,
            harness_name,
            harness_code,
            target_code,
            type_of_target,
            standalone_pov_builder_prompt=standalone_pov_builder_prompt,
            additional_prompting=additional_prompting,
        )
        return prompt
