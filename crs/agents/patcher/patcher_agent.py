"""Patching agent."""

import uuid
from typing import Literal

import dotenv
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import StateGraph

from crs.agents.constants import DISPOSITION_PATCH_SUCCESS
from crs.agents.patcher.models import PatcherAgentState
from crs.agents.patcher.patcher_tools import create_verify_tool
from crs.agents.patcher.prompt import PromptGenerator
from crs.agents.tools.lsp_tools import FindReferences
from crs.agents.tools.lsp_tools import GetSymbolDefinition
from crs.agents.tools.lsp_tools import UpdateSymbolDefinition
from crs.agents.tools.named_base_tool import NamedBaseTool
from crs.base.agent_base import MOTIVATOR_NODE
from crs.base.agent_base import BaseAgent
from crs.base.agent_base import CallModelNode
from crs.base.agent_base import SequentialToolNode
from crs.base.agent_base import tool_message_disposition
from crs.base.context import CrsContext
from crs.base.settings import AI_MODEL_NAME_PATCHER
from crs.base.settings import IGNORE_INTERNAL_ONLY
from crs.base.settings import RECURSION_PATCHER
from crs.base.util import dedupe
from crs.logger import CRS_LOGGER

log = CRS_LOGGER.getChild(__name__)


class PatcherAgent(BaseAgent[CrsContext]):
    """
    Produce actual patch for vulnerability based on POV and vulnerability description.
    """

    # state_type: PatcherAgentState

    NAME = "patcher_agent"

    def __init__(self, ctx: CrsContext):
        """init"""
        super().__init__(ctx)
        self.lsp_context, self.target_code = self.init_lsp(ctx.comp_env.checkout(ctx.subproj_target_ref))

    def run_on_vuln(self, vuln_idx: int) -> None:
        """
        Run patcher on one vulnerability.
        """

        vulnerability = self.ctx.vulnerabilities[vuln_idx]

        if vulnerability.reproducer is None:
            log.info(f"No reproducers found for vulnerability with index {vuln_idx}")
            return

        log.info(f"Running Patcher on vulnerability with index {vuln_idx}")

        self.lsp_ctx.add_crash_report(
            self.lsp_ctx.lang_config.crash_report_t(vulnerability.reproducer.crash_report, self.ctx)
        )

        # checkout target ref
        repository = self.ctx.comp_env.checkout(self.ctx.subproj_target_ref)
        # Create patch branch on which all error free changes are committed.
        # That allows us to easily rollback changes with errors
        patch_branch = repository.create_head(f"patch_{uuid.uuid4()}")
        patch_branch.checkout()

        if self.ctx.proj_yaml_model.language == "c":
            # add -g to compiler flags to get paths ASAN report to get
            dotenv.set_key(self.ctx.cp_path_abs / ".env.docker", "CP_BASE_EXTRA_CFLAGS", "-g")
            dotenv.set_key(self.ctx.cp_path_abs / ".env.docker", "CP_HARNESS_EXTRA_CFLAGS", "-g")

        # run private tests if they exist
        if not IGNORE_INTERNAL_ONLY.get() and (self.ctx.cp_path_abs / ".internal_only").exists():
            dotenv.set_key(
                self.ctx.cp_path_abs / ".env.project",
                "CP_DOCKER_EXTRA_ARGS",
                f"-v {self.ctx.cp_path_abs}/.internal_only:/.internal_only",
            )

        # place pov in file for run.sh
        pov_path = self.ctx.cp_path_abs / "work" / f"pov-{uuid.uuid4()}.blob"
        pov_path.write_bytes(vulnerability.reproducer.blob)

        # setup lsp
        # TODO: untangle this
        lsp_context, diff = self.lsp_context, self.target_code
        # TODO untangle this
        lang_config, symbol_history = lsp_context.lang_config, lsp_context.symbol_history

        verify_tool = create_verify_tool(
            pov_path,
            symbol_history,
            lang_config.editor,
            lang_config,
            self.ctx,
        )
        self.tools: list[NamedBaseTool] = [
            GetSymbolDefinition(editor=lang_config.editor, symbol_history=symbol_history),
            UpdateSymbolDefinition(editor=lang_config.editor, symbol_history=symbol_history, repository=repository),
            FindReferences(editor=lang_config.editor, symbol_history=symbol_history),
            verify_tool,
        ]

        call_model = CallModelNode(self.tools, model_name=AI_MODEL_NAME_PATCHER.get())
        tool_node = SequentialToolNode(self.tools)

        graph = StateGraph(PatcherAgentState)
        graph.add_node("call_model", call_model)
        graph.add_node("run_tool", tool_node)
        graph.add_node("motivate", MOTIVATOR_NODE)

        def from_call_model_to(state: PatcherAgentState) -> Literal["run_tool", "motivate", "__end__"]:
            last_message = state.messages[-1]
            if isinstance(last_message, AIMessage):
                if last_message.tool_calls:
                    return "run_tool"
                if MOTIVATOR_NODE.matches_give_up(last_message):
                    return "__end__"
                return "motivate"
            return "__end__"

        def from_run_tool_to(state: PatcherAgentState) -> Literal["call_model", "__end__"]:
            for tool_msg in state.latest_tool_messages:
                if tool_message_disposition(tool_msg) == DISPOSITION_PATCH_SUCCESS:
                    return "__end__"
            return "call_model"

        graph.add_conditional_edges("call_model", from_call_model_to)
        graph.add_conditional_edges("run_tool", from_run_tool_to)

        graph.set_entry_point("call_model")

        self.graph = graph.compile()
        self.state = PatcherAgentState.from_messages(
            messages=[HumanMessage(content=PromptGenerator(self.ctx, vulnerability, diff, verify_tool.name).get())]
        )

        self.exec_graph(RunnableConfig(recursion_limit=RECURSION_PATCHER.get()))

        viewed_files = symbol_history.export_viewed_files(self.ctx.cp_path_abs)
        self.ctx.viewed_files.extend(dedupe(viewed_files))

    # TODO: we can only do this if the verify tol gets refactored out of the factory function.
    # def _get_initial_messages(self) -> list[BaseMessage]:
    #     return [HumanMessage(content=PromptGenerator(self.ctx, self. target_code, verify_tool.name).get())]

    def _get_initial_messages(self) -> list[BaseMessage]:
        raise NotImplementedError("This cant be called yet, see todo in the code above!")

    def run(self) -> CrsContext:

        if not self.ctx.vulnerabilities:
            log.info("No vulnerabilities found. Exiting.")
            return self.ctx

        for i in range(0, len(self.ctx.vulnerabilities)):
            self.run_on_vuln(i)

        return self.ctx
