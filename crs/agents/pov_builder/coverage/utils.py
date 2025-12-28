"""
Provides methods used in chat_agent_base in case coverage is enabled.
"""

from pathlib import Path

from langchain_core.messages import AIMessage

from crs.agents.pov_builder.coverage.calltree import CallTree
from crs.agents.pov_builder.coverage.prompt import get_coverage_prompt
from crs.agents.tools.lsp_tools import GetSymbolDefinition
from crs.base.base_state import BaseChatState
from crs.base.context import CrsContext
from crs.base.settings import CALL_TREE_ANALYSIS_DIR
from crs.base.settings import POV_COV_INCLUDE_SEARCHES
from crs.base.settings import POV_TARGET_FUNCTION
from crs.base.settings import POV_USE_CALL_TREE_PROMPTING


# pylint: disable=too-many-branches
def build_coverage_feedback_prompt(ctx: CrsContext, state: BaseChatState, coverage_file_path: Path) -> str:
    """
    Build up the prompt with coverage information.
    """
    symbols = set()
    for msg in state.messages:
        if isinstance(msg, AIMessage):
            for tool_call in msg.tool_calls:
                if tool_call["name"] == GetSymbolDefinition.NAME:
                    symbols.add(tool_call["args"]["symbol_name"])

    textcov_report = coverage_file_path.read_text()
    # if call-tree analysis prompting is enabled and a target function is set,
    # we will also include the symbols that were searched for
    pov_target_function = POV_TARGET_FUNCTION.get()
    target_function_coverage = None
    if POV_USE_CALL_TREE_PROMPTING.get() and pov_target_function:
        call_tree = CallTree.from_project(
            ctx.proj_yaml_model.cp_name,
            ctx.cp_path_abs,
            ctx.cp_path_abs / ctx.src_path_rel,
            CALL_TREE_ANALYSIS_DIR.get(),
            ctx.harness_id,
        )
        # append the functions that were searched for in the call tree analysis
        for path in call_tree.get_all_paths("LLVMFuzzerTestOneInput", pov_target_function):
            for symbol in path:
                # if its not LLVMFuzzerTestOneInput, add it to the symbols
                if symbol != "LLVMFuzzerTestOneInput":
                    symbols.add(symbol)

        target_function_coverage = extract_cov_for_symbol(pov_target_function, textcov_report)

    # iterate over the reports and check if the symbols are in there
    if POV_COV_INCLUDE_SEARCHES.get():
        searched_symbols_coverage = ""
        for symbol in symbols:
            searched_symbols_coverage += extract_cov_for_symbol(symbol, textcov_report)
    else:
        searched_symbols_coverage = None

    final_cov_prompt = get_coverage_prompt(target_function_coverage, searched_symbols_coverage)

    return final_cov_prompt


def extract_cov_for_symbol(symbol: str, textcov_report: str) -> str:
    """extract_cov_for_symbol"""
    # add everything from symbol: to the next completely empty line / line that contains no |
    symbol_coverage = ""
    adding = False
    for line in textcov_report.splitlines():
        if adding:
            if line.strip() == "" or "|" not in line.strip():
                break
            symbol_coverage += line + "\n"
        if f"{symbol}:".strip() == line.strip():
            symbol_coverage = f"{symbol}:\n"
            adding = True
    return symbol_coverage
