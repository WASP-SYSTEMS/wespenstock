"""POV builder agent."""

import dotenv
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.runnables.config import RunnableConfig

from crs.agents.pov_builder.verifier_prompt import VerifierPrompt
from crs.agents.tools.generate_reproducer import GenerateReproducer
from crs.agents.tools.lsp_tools import FindReferences
from crs.agents.tools.lsp_tools import GetSymbolDefinition
from crs.aixcc.env import SubprojectCommit
from crs.aixcc.project_yaml import HarnessValue
from crs.base.agent_base import BaseAgent
from crs.base.base_state import BaseChatState
from crs.base.context import CrsContext
from crs.base.context import Vulnerability
from crs.base.exceptions import TooManyErrorsError
from crs.base.settings import AI_MODEL_NAME_POV
from crs.base.settings import CHECK_POV_AT
from crs.base.settings import POV_SKIP_BUILD
from crs.base.settings import POV_STANDALONE_MODE
from crs.base.settings import RECURSION_POV
from crs.base.util import dedupe
from crs.logger import CRS_LOGGER

log = CRS_LOGGER.getChild(__name__)


class PovBuilderAgentState(BaseChatState):
    """
    the state where messages of the pov builder are stored in and more
    currently just a wrapper for the case that we need more at some point
    """


# pylint: disable=too-many-instance-attributes
class PovBuilderAgent(BaseAgent[CrsContext]):
    """
    Build POV based on description of analyzer (or not when using Standalone Mode, lol).
    """

    # pylint: disable=too-many-instance-attributes

    state_type = PovBuilderAgentState

    NAME = "verifier_agent"

    def __init__(self, ctx: CrsContext):
        """
        The agent itself? Yeah. It just finds vulnerabilities in software - hopefully. simple as that (:
        """
        super().__init__(ctx=ctx)

        if ctx.proj_yaml_model.language == "c":
            # add -g to compiler flags to get paths ASAN report to get
            dotenv.set_key(ctx.cp_path_abs / ".env.docker", "CP_BASE_EXTRA_CFLAGS", "-g")
            dotenv.set_key(ctx.cp_path_abs / ".env.docker", "CP_HARNESS_EXTRA_CFLAGS", "-g")

        if check_ref := CHECK_POV_AT.get():
            commit = SubprojectCommit(src_id=ctx.subproj_target_ref.src_id, ref=check_ref)
            repository = ctx.comp_env.checkout(commit)
        else:
            repository = ctx.comp_env.checkout(ctx.subproj_target_ref)

        if not POV_SKIP_BUILD.get():
            # build project
            self.ctx.comp_env.build()

        self.lsp_ctx, self.target_code = self.init_lsp(repository)
        self.generate_reproducer_tool = GenerateReproducer(harness_name=self.harness.name, ctx=self.ctx)
        self.get_definition_tool = GetSymbolDefinition(
            editor=self.lsp_ctx.lang_config.editor, symbol_history=self.lsp_ctx.symbol_history
        )

        self.tools = [
            self.generate_reproducer_tool,
            self.get_definition_tool,
            FindReferences(editor=self.lsp_ctx.lang_config.editor, symbol_history=self.lsp_ctx.symbol_history),
        ]

        self.vulnerability = Vulnerability()

    def _run(self, vuln_idx: int) -> CrsContext:
        """
        Runs the agent, submits the PoV, returns the context with new PoV Description if successful.
        Raises RuntimeError if PoV was not accepted or nothing of value was generated at all.
        """
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements
        self.vulnerability = self.ctx.vulnerabilities[vuln_idx]

        log.info(f"Running Verifier on vulnerability with index {vuln_idx}")

        self.state = self.get_base_chat_state()

        self.build_and_compile_default_graph(AI_MODEL_NAME_POV)
        self.exec_graph(RunnableConfig(recursion_limit=RECURSION_POV.get()))

        self.ctx.vulnerabilities[vuln_idx].reproducer = self.generate_reproducer_tool.pov_description

        viewed_files = self.lsp_ctx.symbol_history.export_viewed_files(self.ctx.cp_path_abs)
        self.ctx.viewed_files.extend(dedupe(viewed_files))

        self.ctx.save()

        return self.ctx

    def _get_initial_messages(self) -> list[BaseMessage]:

        msgs = [
            SystemMessage(content=VerifierPrompt.get_system_prompt()),
            HumanMessage(
                content=VerifierPrompt.get_prompt(
                    self.ctx,
                    self.vulnerability,
                    self.harness.name,
                    self.harness_code,
                    self.target_code,
                    standalone_pov_builder_prompt=POV_STANDALONE_MODE.get(),
                    additional_prompting=self._get_additional_prompting(),
                )
            ),
        ]
        return msgs

    def _get_additional_prompting(self) -> str:
        """Additional prompting."""
        return ""

    @property
    def harness(self) -> HarnessValue:
        """get the HarnessValue"""
        return self.ctx.proj_yaml_model.harnesses[self.ctx.harness_id]

    @property
    def harness_code(self) -> str:
        """get harness code"""
        harness_code = (self.ctx.cp_path_abs / self.harness.source).read_text()

        return harness_code

    def run(self) -> CrsContext:
        """method to run the agent"""

        if POV_STANDALONE_MODE.get():
            # In standalone mode the analyzer is skipped, so we have no vulnerability report.
            # Therefore we insert an empty vulnerability here,so the reproducer field can be
            # set by the Verifier.
            self.ctx.vulnerabilities.append(Vulnerability())
        elif not self.ctx.vulnerabilities:
            log.info("No vulnerability reports found. Exiting.")
            return self.ctx

        exceptions: list[Exception] = []

        for i in range(0, len(self.ctx.vulnerabilities)):
            # change agent name to prevent overwriting of .md files
            self.NAME = f"verifier_agent_{i}"  # pylint: disable=invalid-name
            try:
                # pylint: disable=protected-access
                self.ctx = self._run(i)
            # pylint: disable=broad-except
            except Exception as e:
                exceptions.append(e)
                log.critical(f"Verifier failed for vulnerability with index {i}: {e}")

        if exceptions:
            raise TooManyErrorsError(exceptions) if len(exceptions) > 1 else exceptions[0]

        return self.ctx
