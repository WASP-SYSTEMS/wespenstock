"""Function Verifier agent."""

import json
from pathlib import Path

from git import Repo
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

from crs.agents.pov_builder.pov_builder_agent import PovBuilderAgent
from crs.agents.pov_builder.verifier_prompt import VerifierPrompt
from crs.agents.tools.lsp_tools import GetSymbolDefinition
from crs.agents.tools.lsp_tools import LSPContext
from crs.aixcc.env import SubprojectCommit
from crs.base.context import CrsContext
from crs.base.settings import INTROSPECTOR_ARTIFACTS_DIR
from crs.base.settings import POV_STANDALONE_MODE
from crs.base.settings import TARGET_FUNCTION
from crs.base.util import docker_to_local_path
from crs.logger import CRS_LOGGER

log = CRS_LOGGER.getChild(__name__)


class FunctionVerifier(PovBuilderAgent):
    """
    Build Reproducer for a specific function in the codebase.
    """

    def __init__(self, ctx: CrsContext):
        """
        The agent itself? Yeah. It just finds vulnerabilities in software - hopefully. simple as that (:
        """
        super().__init__(ctx=ctx)

    def init_lsp(self, checkout_repo_at: None | SubprojectCommit | Repo) -> tuple[LSPContext, str]:
        """Initialize LSP context for the current project."""
        self.lsp_ctx = LSPContext(self.ctx)
        self.lsp_ctx.add_known_symbols()
        return self.lsp_ctx, ""

    def _run(self, vuln_idx: int) -> CrsContext:
        self.vulnerability = self.ctx.vulnerabilities[vuln_idx]

        report = self.vulnerability.report
        if report is not None:
            target_functions = report.vuln_functions
            function_file_paths = [p.as_posix() for p in self.ctx.viewed_files]
            if not target_functions or not function_file_paths:
                raise ValueError("No target functions found for analysis.")
        else:
            target_functions = FunctionVerifier._fetch_target_functions()
            all_functions_json = FunctionVerifier._prepare_all_functions_json(self.ctx)
            function_file_paths = FunctionVerifier.get_function_file_paths(target_functions, all_functions_json)

        # Obtain the function source using LSP
        self.target_code = FunctionVerifier._get_function_sources(
            self.get_definition_tool, self.ctx, target_functions, function_file_paths
        )

        return super()._run(vuln_idx)

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
                    type_of_target="function",
                    standalone_pov_builder_prompt=POV_STANDALONE_MODE.get(),
                    additional_prompting=self._get_additional_prompting(),
                )
            ),
        ]
        return msgs

    @staticmethod
    def _get_function_sources(
        get_definition_tool: GetSymbolDefinition,
        ctx: CrsContext,
        target_functions: list[str],
        function_file_paths: list[str],
    ) -> str:
        """Use LSP to get the definition/source of the target function, updating symbol history as needed."""
        if get_definition_tool is None:
            log.error("GetSymbolDefinition tool is not available, cannot get function definition.")
            raise ValueError("GetSymbolDefinition tool is not available, cannot get function definition.")

        target_function_sources = []
        for target_function, file_path in zip(target_functions, function_file_paths):
            if not file_path:
                # continue to next function if file path is empty
                continue
            if ctx.proj_yaml_model.language != "python":
                # Non-Python: file path is already a filesystem path
                path = docker_to_local_path(Path(file_path), ctx.cp_path_abs)
                if path is None:
                    path = docker_to_local_path(Path(file_path), ctx.cp_path_abs / ctx.src_path_rel)
                if path is None:
                    raise ValueError(
                        f"Source file for function {target_function} not found locally (in docker {file_path})."
                    )
                get_definition_tool.symbol_history.update(path)
                target_function_source = get_definition_tool.get_definition(target_function)
            else:
                # Python: function path may be a module path; normalize to a .py file and extract simple name
                py_file_path = FunctionVerifier._python_module_path(ctx, file_path)
                get_definition_tool.symbol_history.update(py_file_path)
                simple_name = target_function.split(".")[-1]
                target_function_source = get_definition_tool.get_definition(simple_name)

            if target_function_source is None:
                log.error(
                    "Function '%s' not found in the codebase using our LSP. Skipping AnalyzerAgent.",
                    target_function,
                )
                raise ValueError(f"Function '{target_function}' not found in the codebase using our LSP.")

            target_function_sources.append(target_function_source.definition)
        return "\n\n".join(target_function_sources)

    @staticmethod
    def _fetch_target_functions() -> list[str]:
        """Return the target function name from settings or raise if missing."""
        target_functions = TARGET_FUNCTION.get()
        if target_functions:
            log.info("Function to analyze: %s", target_functions.split(","))
            return target_functions.split(",")
        log.info("No function to analyze, skipping AnalyzerAgent.")
        raise ValueError("No function to analyze, skipping AnalyzerAgent.")

    @staticmethod
    def _prepare_all_functions_json(ctx: CrsContext) -> Path:
        """Ensure introspector artifacts exist and return the path to all_functions.json."""
        all_functions_json = Path(INTROSPECTOR_ARTIFACTS_DIR.get()) / ctx.proj_yaml_model.cp_name / "all_functions.json"
        if not all_functions_json.exists():
            log.error("Introspector artifact '%s' not found.", all_functions_json)
            raise FileNotFoundError(f"Introspector artifact '{all_functions_json}' not found.")
        return all_functions_json

    @staticmethod
    def _python_module_path(ctx: CrsContext, function_file_path: str) -> Path:
        """Convert a Python module path (dots) to a filesystem path ending with .py and return absolute path."""
        if function_file_path.startswith("..."):
            normalized = "../" + function_file_path[3:].replace(".", "/")
        else:
            normalized = function_file_path.replace(".", "/")
        normalized = normalized + ".py"
        return ctx.cp_path_abs / ctx.src_path_rel / normalized.removeprefix("/")

    @staticmethod
    def get_function_file_paths(function_names: list[str], all_functions_json: Path) -> list[str]:
        """
        Get the local path to a file from the all_functions.json.
        """
        with open(all_functions_json, encoding="utf-8") as f:
            data = json.load(f)
        file_paths = []
        for function_name in function_names:
            for func in data["functions"]:
                if func.get("function_name") == function_name:
                    file_paths.append(func.get("function_filename"))
                    break
            else:
                log.warning(f"Could find function {function_name} in all_functions.json")

        if len(file_paths) <= 0:
            raise ValueError(f"Functions {function_names} not found in all_functions.json.")

        return file_paths
