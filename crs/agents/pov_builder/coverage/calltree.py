"""
Code for call tree generation.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import rustworkx as rx
import yaml

from crs.agents.pov_builder.coverage.introspector_api import IntrospectorApi
from crs.agents.pov_builder.coverage.introspector_api import ValidProjectFile
from crs.base.util import docker_to_local_path
from crs.logger import CRS_LOGGER
from crscommon.editor.editor import SourceEditor
from crscommon.editor.language_helper import CLanguageHelper
from crscommon.editor.lsp.clangd_client import ClangdClient
from crscommon.editor.lsp.compilation_db_preparer import CompilationDbPreparer
from crscommon.editor.lsp_interface import LspInterface
from crscommon.editor.symbol import LineInSymbolDefinition
from crscommon.editor.symbol import SymbolDescription

log = CRS_LOGGER.getChild(__name__)


VALID_PROJECTS_FILE = Path("oss_fuzz_integration/introspector_projects.json")


class AllFunctionsYaml:
    """All functions yaml file produced by introspector."""

    def __init__(self, data: dict) -> None:
        self.data = data

    def __iter__(self) -> Iterator[dict]:
        """Iterate over all functions."""
        return iter(self.data["All functions"]["Elements"])

    @staticmethod
    def from_artifact_dir(artifact_dir: Path, harness: str | int) -> AllFunctionsYaml:
        """
        Load file from artifacts directory.
        harness can be the harness name or the index of the harness.
        """

        correlation_file = artifact_dir / IntrospectorApi.CORRELATION_FILENAME

        if not correlation_file.exists():
            raise FileNotFoundError(f"Correlation file {correlation_file} not found.")

        with open(correlation_file, encoding="utf-8") as file:
            data = yaml.safe_load(file)

        fuzzer_log_file_name: str | None = None

        if isinstance(harness, str):
            for entry in data["pairings"]:
                exe_name = os.path.basename(entry["executable_path"])
                if exe_name == harness:
                    harness_name = entry["executable_path"]
                    fuzzer_log_file_name = entry["fuzzer_log_file"]
        else:
            harness_name = os.path.basename(data["pairings"][harness]["executable_path"])
            fuzzer_log_file_name = data["pairings"][harness]["fuzzer_log_file"]

        if fuzzer_log_file_name is None:
            raise FileNotFoundError(f"fuzzer_log_file for harness {harness_name} not found in {correlation_file}.")

        # Sometimes the fuzzer_log_file_name ends with '.data'. This must be removed.
        if fuzzer_log_file_name.endswith(".data"):
            fuzzer_log_file_name = fuzzer_log_file_name[:-5]

        pickled_yaml_file = artifact_dir / (fuzzer_log_file_name + ".data.yaml.pickle")

        if not pickled_yaml_file.exists():
            raise FileNotFoundError(f"Pickled YAML file {pickled_yaml_file} not found.")

        with open(pickled_yaml_file, "rb") as file:
            log.info(f"Loading YAML from pickle file {pickled_yaml_file}")
            all_functions = pickle.load(file)

        return AllFunctionsYaml(all_functions)


@dataclass(frozen=True)  # make hashable
class Callsite:
    """Information about callsite."""

    name: str
    """Name of the called function."""

    file: Path
    """File in which it was called."""

    line: int
    """Line in which it was called."""

    column: int
    """Column in which it was called."""

    @staticmethod
    def from_yaml_info(callsite_info: dict[str, str], src_path: Path) -> Callsite:
        """
        Create callsite from the function's name, the path to the source file
        and the source string from the 'Callsites' field in the "All functions' yaml
        produced by the introspector (e.g. 'fuzz_project.c:12,3').
        """

        # src_filed has form 'fuzz_project.c:12,3'
        splitted_line = callsite_info["Src"].split(":")
        assert len(splitted_line) > 0
        line_col = splitted_line[-1].split(",")
        assert len(line_col) > 1

        return Callsite(
            name=callsite_info["Dst"],
            file=src_path,
            line=int(line_col[0]),
            column=int(line_col[1]),
        )


@dataclass(frozen=True)  # make hashable
class CallTreeFunction:
    """
    Information about a function in the call tree.
    """

    name: str
    """Name of the function."""

    file: Path | None
    """File in which the function is defined."""

    start_line: int
    """Start of function definition."""

    end_line: int
    """End of function definition."""

    callsites: list[Callsite]
    """Functions called by this function."""

    def get_definition(self, editor: SourceEditor) -> SymbolDescription | None:
        """Get the functions definition."""

        # cannot extract definition without file path
        if self.file is None:
            return None

        return editor.get_symbol_definition(
            # set line=lvl.line - 1, because lang server uses 0-based lines
            LineInSymbolDefinition(name=self.name, file=self.file.absolute(), line=self.start_line - 1)
        )


class CallTree:
    """
    Holds the whole call tree.
    """

    def __init__(self, all_functions: AllFunctionsYaml, project_path: Path, project_src_path: Path) -> None:
        # mapping: function name -> int
        self.func_mapping: dict[str, int] = {}
        # store tree in directed graph
        self.graph = rx.PyDAG(check_cycle=True)  # pylint: disable=no-member
        # all files in which functions occur
        self.files: set[Path] = set()

        log.info("Creating graph from call tree")

        for func in all_functions:
            func_src_file = Path(func["functionSourceFile"])
            src_path = docker_to_local_path(func_src_file, project_path)
            if src_path is None:
                src_path = docker_to_local_path(func_src_file, project_src_path)

            if src_path is None:
                log.warning(
                    f"Not adding function `{func['functionName']}` to call tree "
                    f"(docker source '{func['functionSourceFile']})'"
                )
                continue

            self.files.add(src_path)

            callsites = [Callsite.from_yaml_info(cs, src_path) for cs in func["Callsites"]]

            call_tree_func = CallTreeFunction(
                name=func["functionName"],
                file=src_path,
                start_line=func["functionLinenumber"],
                end_line=func["functionLinenumberEnd"],
                callsites=callsites,
            )

            # store call_tree_func in node
            self.func_mapping[call_tree_func.name] = self.graph.add_node(call_tree_func)

        for caller_idx in self.graph.node_indices():
            caller: CallTreeFunction = self.graph.get_node_data(caller_idx)
            for i, callee in enumerate(caller.callsites):
                try:
                    callee_idx = self.func_mapping[callee.name]
                    # rx.DAGWouldCycle is not thrown when caller_idx == callee_idx
                    if caller_idx != callee_idx:
                        # Use (line in which callee was called, i) as weight for the edge.
                        # We add i to make sure functions called in the same line have the right order.
                        self.graph.add_edge(caller_idx, callee_idx, (callee.line, i))
                except (KeyError, rx.DAGWouldCycle):  # pylint: disable=no-member
                    pass

        if "LLVMFuzzerTestOneInput" not in self.func_mapping:
            raise RuntimeError("Could not find fuzzer entry function LLVMFuzzerTestOneInput in graph")

        # pylint: disable=no-member
        cycle = rx.digraph_find_cycle(self.graph, self.func_mapping["LLVMFuzzerTestOneInput"])
        if cycle:
            raise RuntimeError(f"Cycle: {[self.graph.get_node_data(i).name for i in cycle[0]]}")

    def __getitem__(self, function: str) -> CallTreeFunction:
        """Get function though dict like access."""
        return self.graph.get_node_data(self.func_mapping[function])

    def __iter__(self) -> Iterator[CallTreeFunction]:
        """Iterate over all functions in tree."""
        return iter(self.graph.nodes())

    def get_all_paths(self, start_func: str, target_func: str) -> list[list[str]]:
        """Get all paths from a start function to a target function."""
        paths: list[list[str]] = []

        for path in rx.all_simple_paths(self.graph, self.func_mapping[start_func], self.func_mapping[target_func]):
            paths.append([self.graph.get_node_data(i).name for i in path])

        return paths

    def get_indented_call_tree(self, root_func: str) -> list[tuple[int, str]]:
        """
        Returns a tree preserving the order in which functions are called.
        Returns list of tuple (depth, function name)
        """
        return self._get_indented_call_tree(root_func, [(0, root_func)], 1)

    def _get_indented_call_tree(
        self, root_func: str, tree_list: list[tuple[int, str]], depth: int
    ) -> list[tuple[int, str]]:
        root_idx = self.func_mapping[root_func]

        out_edges = list(self.graph.out_edges(root_idx))

        # out_edges tuple (node_index, child_index, edge_data).
        # edge_data has form (callee.line, i), where i is the index of callee sorted by the order
        # in which all callees are called. By adding callee.line + i we preserve the correct order
        # even if multiple functions are called in the same line.
        out_edges.sort(key=lambda e: sum(e[2]))

        for e in out_edges:
            callee_idx = e[1]
            callee: CallTreeFunction = self.graph.get_node_data(callee_idx)
            tree_list.append((depth, f"{callee.name}"))
            depth += 1
            tree_list = self._get_indented_call_tree(callee.name, tree_list, depth)
            depth -= 1

        return tree_list

    @staticmethod
    def from_project(
        project_name: str, project_path: Path, project_src_path: Path, call_tree_out_path: Path, harness: str | int
    ) -> CallTree:
        """
        Perform call tree analysis for a harness.
        """

        all_functions = AllFunctionsYaml.from_artifact_dir(call_tree_out_path / project_name, harness)

        return CallTree(all_functions, project_path, project_src_path)


if __name__ == "__main__":
    v_projects = ValidProjectFile.model_validate_json(VALID_PROJECTS_FILE.read_text(encoding="utf-8"))

    introspector_out_path = Path("introspector_out")

    IntrospectorApi(v_projects.projects).download(introspector_out_path)

    for project in v_projects.projects:

        if project.name != "lighttpd":
            continue

        cp_path = Path(f"/data/fast2/annika/prep-test-lab/of-all-c/cp-{project.name}-HEAD").absolute()

        call_tree = CallTree.from_project(
            project.name,
            cp_path,
            cp_path / "src/lighttpd1.4",
            introspector_out_path,
            0,
        )

        file_extensions = [".c", ".cc", ".cpp", ".h"]
        clangd = ClangdClient(cp_path / "src/lighttpd1.4", CompilationDbPreparer(cp_path))
        c_editor = SourceEditor(
            LspInterface(clangd, CLanguageHelper(cp_path / "src/lighttpd1.4"), file_extensions), file_extensions
        )

        log.info("Finding paths")

        t = call_tree.get_indented_call_tree("LLVMFuzzerTestOneInput")

        print()

        for n in t:
            print(f"{' ' * n[0] * 2}{n[1]}")

        print()

        for p in call_tree.get_all_paths("LLVMFuzzerTestOneInput", "buffer_clen"):
            print(p)

        for f in call_tree:
            print("Found" if f.get_definition(c_editor) is not None else "NOT FOUND", f.name)
