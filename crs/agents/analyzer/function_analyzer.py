"""POV builder agent."""

from git import Repo
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

from crs.agents.analyzer.analyzer_agent import AnalyzerAgent
from crs.agents.analyzer.prompt import AnalyzerPrompt
from crs.agents.pov_builder.verifier_function import FunctionVerifier
from crs.agents.tools.lsp_tools import LSPContext
from crs.aixcc.env import SubprojectCommit
from crs.base.context import CrsContext
from crs.logger import CRS_LOGGER

log = CRS_LOGGER.getChild(__name__)


class FunctionAnalyzer(AnalyzerAgent):
    """
    Analyze a specific function in the codebase.
    """

    def __init__(self, ctx: CrsContext, target_override: tuple[LSPContext, str] | None = None):
        """
        Inits the function analyzer.
        """
        super().__init__(ctx=ctx, target_override=target_override)

        if target_override is None:
            target_functions = FunctionVerifier._fetch_target_functions()
            all_functions_json = FunctionVerifier._prepare_all_functions_json(self.ctx)
            function_file_paths = FunctionVerifier.get_function_file_paths(target_functions, all_functions_json)

            # Obtain the function source using LSP
            self.target_code = FunctionVerifier._get_function_sources(
                self.get_definition_tool, self.ctx, target_functions, function_file_paths
            )

    def init_lsp(self, checkout_repo_at: None | SubprojectCommit | Repo) -> tuple[LSPContext, str]:
        """Initialize LSP context for the current project."""
        self.lsp_ctx = LSPContext(self.ctx)
        self.lsp_ctx.add_known_symbols()
        return self.lsp_ctx, ""

    def _get_initial_messages(self) -> list[BaseMessage]:
        msgs = [
            SystemMessage(content=AnalyzerPrompt.get_system_prompt()),
            HumanMessage(
                content=AnalyzerPrompt.get_prompt(
                    self.ctx.proj_yaml_model.language,
                    self.ctx.cp_name,
                    self.target_code,
                    "function",
                )
            ),
        ]
        return msgs
