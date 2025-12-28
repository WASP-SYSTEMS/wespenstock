"""The ReproducerAgent with Code Path Agent"""

from pathlib import Path

from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

from crs.agents.pov_builder.code_path_prompt import CodePathPrompt
from crs.agents.pov_builder.coverage.calltree import CallTree
from crs.agents.pov_builder.pov_builder_agent import PovBuilderAgent
from crs.agents.tools.generate_reproducer import GenerateReproducer
from crs.agents.tools.generate_reproducer_code_path import GenerateReproducerCodePath
from crs.agents.tools.lsp_tools import FindReferences
from crs.base.context import CrsContext
from crs.base.settings import ANALYZER_CHECK_SANITY
from crs.base.settings import CALL_TREE_ANALYSIS_DIR
from crs.base.settings import POV_STANDALONE_MODE
from crs.base.settings import POV_TARGET_FUNCTION
from crs.base.settings import POV_USE_COVERAGE
from crs.logger import CRS_LOGGER

log = CRS_LOGGER.getChild(__name__)


class VerifierAgentCodePath(PovBuilderAgent):
    """The ReproducerAgent with Code Path Agent"""

    def __init__(self, ctx: CrsContext):
        ANALYZER_CHECK_SANITY.set(False)
        self.call_tree: CallTree | None = None
        super().__init__(ctx)
        self.generate_reproducer_tool = (
            GenerateReproducerCodePath(call_tree=self.call_tree, harness_name=self.harness.name, ctx=self.ctx)
            if (POV_USE_COVERAGE.get() or POV_TARGET_FUNCTION.get()) and self.call_tree is not None
            else GenerateReproducer(harness_name=self.harness.name, ctx=self.ctx)
        )
        log.info(f"Using GenerateReproducer tool: {self.generate_reproducer_tool.__class__.__name__}")

        self.tools = [
            self.generate_reproducer_tool,
            self.get_definition_tool,
            FindReferences(editor=self.lsp_ctx.lang_config.editor, symbol_history=self.lsp_ctx.symbol_history),
        ]

    def _get_initial_messages(self) -> list[BaseMessage]:

        msgs = [
            SystemMessage(content=CodePathPrompt.get_system_prompt()),
            HumanMessage(
                content=CodePathPrompt.get_prompt(
                    self.ctx,
                    self.vulnerability,
                    self.harness.name,
                    self.harness_code,
                    self.target_code,
                    type_of_target="function",
                    standalone_pov_builder_prompt=POV_STANDALONE_MODE.get(),
                    additional_prompting=self._get_additional_prompting(),
                )
            ),
        ]
        return msgs

    def _get_additional_prompting(self) -> str:

        call_tree_dir = CALL_TREE_ANALYSIS_DIR.get()
        target_function_name = POV_TARGET_FUNCTION.get()
        if not target_function_name:
            raise ValueError(
                "Reproducer Agent [CodePath] can only run with valid target function, "
                f"received: {target_function_name=}"
            )

        log.info(f"Target function for PoV generation is set to {target_function_name}")

        # TODO: move to function pov builder when exists
        additional_prompting, _ = self.additional_prompting_call_tree(
            target_function_name,
            call_tree_dir,
        )
        return additional_prompting

    def additional_prompting_call_tree(
        self,
        target_function: str,
        calltree_dir: Path,
    ) -> tuple[str, list[list[str]]]:
        """
        Perform call tree analysis to find code paths leading to the target function.
        Returns LLM prompting, list of call trees
        """

        additional_prompting = ""
        log.info(f"Performing call tree analysis for target function {target_function} in {calltree_dir}")

        self.call_tree = CallTree.from_project(
            self.ctx.proj_yaml_model.cp_name,
            self.ctx.cp_path_abs,
            self.ctx.cp_path_abs / self.ctx.src_path_rel,
            calltree_dir,
            self.ctx.harness_id,
        )

        self.lsp_ctx.add_files(self.call_tree.files)

        # append the functions that were searched for in the call tree analysis
        code_paths = self.call_tree.get_all_paths("LLVMFuzzerTestOneInput", target_function)

        if not code_paths:
            log.info(f"No code paths found for target function {target_function}")
        else:
            log.info(f"Found {len(code_paths)} code paths for target function {target_function}")

            # get the code of the functions leading to the target function
            unique_functions = set()
            for path in code_paths:
                unique_functions.update(path)
                # sort paths by length
            code_paths = sorted(code_paths, key=len)
            function_code = ""
            retrieved_functions = set()

            # get the code of the functions leading to the target function by order of appearance
            for path in code_paths:
                for function in path:
                    if function not in retrieved_functions:
                        try:
                            symbol_desc = self.call_tree[function].get_definition(self.get_definition_tool.editor)
                            func_def = "N/A" if not symbol_desc else symbol_desc.definition
                            function_code += f"{function}:\n{func_def}\n"  # pylint: disable=line-too-long
                            retrieved_functions.add(function)
                        except (ValueError, RuntimeError, AttributeError) as e:
                            log.warning(f"Could not retrieve code for function {function}: {e}")

                # add code paths to additional prompting
            additional_prompting = (
                f"The target function is `{target_function}`.\n"
                f"The following code paths may lead to the target function:\n"
                f"{chr(10).join(' -> '.join(path) for path in code_paths)}\n"
                f"Use these code paths to generate a PoV that triggers the target function. "
                f"The code of the functions can be found below:\n\n"
                f"{function_code}"
            )

        return additional_prompting, code_paths
