"""Analyzer agent."""

from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.runnables.config import RunnableConfig

from crs.agents.analyzer.prompt import AnalyzerPrompt
from crs.agents.tools.lsp_tools import FindReferences
from crs.agents.tools.lsp_tools import GetSymbolDefinition
from crs.agents.tools.lsp_tools import LSPContext
from crs.base.agent_base import BaseAgent
from crs.base.base_state import BaseChatState
from crs.base.context import CrsContext
from crs.base.settings import AI_MODEL_NAME_ANALYZER
from crs.base.settings import RECURSION_ANALYZER
from crs.base.util import dedupe
from crs.logger import CRS_LOGGER

from .analyzer_tools import EndAnalysis
from .analyzer_tools import SubmitVulnerabilityReport

log = CRS_LOGGER.getChild(__name__)


class AnalyzerAgentState(BaseChatState):
    """State of the analyzer agent. Nothing specific for now."""


class AnalyzerAgent(BaseAgent[CrsContext]):
    """
    Analyze commits and find vulnerabilities.
    """

    state_type = AnalyzerAgentState

    NAME = "analyzer_agent"

    def __init__(self, ctx: CrsContext, target_override: tuple[LSPContext, str] | None = None):
        """init"""
        super().__init__(ctx)

        if target_override is not None:
            self.lsp_ctx, self.target_code = target_override
        else:
            self.lsp_ctx, self.target_code = self.init_lsp(ctx.comp_env.checkout(ctx.subproj_commit))

        self.get_definition_tool = GetSymbolDefinition(
            editor=self.lsp_ctx.lang_config.editor, symbol_history=self.lsp_ctx.symbol_history
        )
        self.find_refs_tool = FindReferences(
            editor=self.lsp_ctx.lang_config.editor, symbol_history=self.lsp_ctx.symbol_history
        )
        self.tools = [
            self.get_definition_tool,
            self.find_refs_tool,
            SubmitVulnerabilityReport(ctx=self.ctx),
            EndAnalysis(),
        ]

    def run(self) -> CrsContext:  # pylint: disable=R0914

        log.info("Entering analyzer agent...")
        self.state = self.get_base_chat_state()

        self.build_and_compile_default_graph(AI_MODEL_NAME_ANALYZER)
        self.exec_graph(RunnableConfig(recursion_limit=RECURSION_ANALYZER.get()))

        viewed_files = self.lsp_ctx.symbol_history.export_viewed_files(self.ctx.cp_path_abs)
        self.ctx.viewed_files.extend(dedupe(viewed_files))

        return self.ctx

    def _get_initial_messages(self) -> list[BaseMessage]:
        msgs = [
            SystemMessage(content=AnalyzerPrompt.get_system_prompt()),
            HumanMessage(
                content=AnalyzerPrompt.get_prompt(
                    self.ctx.proj_yaml_model.language,
                    self.ctx.cp_name,
                    self.target_code,
                )
            ),
        ]
        return msgs
