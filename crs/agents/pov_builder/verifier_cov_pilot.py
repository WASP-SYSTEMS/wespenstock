"""The CovPilot Agent"""

from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

from crs.agents.pov_builder.coverage.calltree import CallTree
from crs.agents.pov_builder.covpilot_prompt import get_covpilot_standalone_prompt
from crs.agents.pov_builder.covpilot_prompt import get_covpilot_system_prompt
from crs.agents.pov_builder.verifier_code_path import VerifierAgentCodePath
from crs.agents.tools.generate_reproducer_cov_pilot import GenerateReproducerCovPilot
from crs.base.context import CrsContext
from crs.base.settings import CALL_TREE_ANALYSIS_DIR
from crs.base.settings import POV_TARGET_FUNCTION


class CovPilot(VerifierAgentCodePath):
    """The CovPilot Agent"""

    def __init__(self, ctx: CrsContext) -> None:
        super().__init__(ctx)
        self.generate_reproducer_tool = GenerateReproducerCovPilot(harness_name=self.harness.name, ctx=self.ctx)

    def _get_initial_messages(self) -> list[BaseMessage]:
        assert self.target_code is not None

        target_function_name = POV_TARGET_FUNCTION.get()
        if target_function_name is None:
            raise ValueError("CovPilot requires a TARGET FUNCTION to be set.")

        call_tree = CallTree.from_project(
            self.ctx.proj_yaml_model.cp_name,
            self.ctx.cp_path_abs,
            self.ctx.cp_path_abs / self.ctx.src_path_rel,
            CALL_TREE_ANALYSIS_DIR.get(),
            self.ctx.harness_id,
        )
        code_paths = call_tree.get_all_paths("LLVMFuzzerTestOneInput", target_function_name)

        # TODO do something better here to account for multiple paths
        #  same sin also in gen pov tool
        analyzed_path = code_paths[0]

        # -1 is target function, so caller is -2
        caller_fn = call_tree[analyzed_path[-2]].get_definition(self.get_definition_tool.editor)

        # this can realistically only happen when the target function is the test_one_input of the harness.
        # ...but mypy demands it.
        if caller_fn is None:
            raise ValueError(
                f"Can't get caller function for target '{target_function_name}' with call-path: {analyzed_path}"
            )

        return [
            SystemMessage(content=get_covpilot_system_prompt()),
            HumanMessage(
                content=get_covpilot_standalone_prompt(
                    self.ctx,
                    self.harness.name,
                    (self.ctx.cp_path_abs / self.harness.source).read_text(),
                    target_function_name,
                    analyzed_path,
                    caller_fn.name,
                    caller_fn.definition,
                )
            ),
        ]
