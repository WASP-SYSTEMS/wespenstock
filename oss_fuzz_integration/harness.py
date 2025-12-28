# pylint: disable=line-too-long
"""
Parse git commit for changed lines of code and determine which harnesses can reach these changed lines of code.
"""

import argparse
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel
from unidiff import PatchSet

from crscommon.logging.logging_provider import LOGGING_PROVIDER
from oss_fuzz_integration.introspector_api import AllFunctionsResponse
from oss_fuzz_integration.introspector_api import fetch_and_cache_api_data
from oss_fuzz_integration.introspector_api import write_json_to_file

log = LOGGING_PROVIDER.new_logger("HARNESS", hook_exception=True)

T = TypeVar("T")


class FileToFunction(BaseModel):
    """
    Dict entry for the function get_all_functions from harness.py
    The function returns a list of functions that were reached by at least one fuzzer according to the
    Fuzz Introspector API
    """

    function_name: str
    start_line: int
    end_line: int
    harnesses: list[str]


def get_all_functions(project: str, outputpath: Path, retries: int = 6) -> dict[str, list[FileToFunction]]:
    """
    Get all functions for a given project using the Introspector API.
    Parse the API data and return a dictionarry that maps received functions (with line info and reached harness) to files
    """
    url = "https://introspector.oss-fuzz.com/api/all-functions"
    params = {"project": project}
    filepath = outputpath / project / "all_functions.json"

    json_data = fetch_and_cache_api_data(
        url=url,
        params=params,
        filepath=filepath,
        data_extractor=AllFunctionsResponse.model_validate,
        retries=retries,
        log=log,
    )

    if not json_data:
        log.error(f"API did not provide any functions for project {project}")
        return {}

    file_to_function: dict[str, list[FileToFunction]] = defaultdict(list)

    for func in json_data.functions:
        try:
            filename = func.function_filename
            function_name = func.function_name

            # Handle line information
            if func.source_line_begin is not None and func.source_line_end is not None:
                start_line = func.source_line_begin
                end_line = func.source_line_end
            elif func.source_line is not None:
                start_line = end_line = func.source_line
            else:
                log.warning(
                    f"No usable line info for function {function_name}@{project} in file {func.function_filename}"
                )
                continue

            entry = FileToFunction(
                function_name=function_name,
                start_line=start_line,
                end_line=end_line,
                harnesses=func.reached_by_fuzzers,
            )

            file_to_function[filename].append(entry)

        except Exception as e:  # pylint: disable=broad-exception-caught
            log.exception(f"Error processing function {func} in project {project}: {e}")

    return file_to_function


def collect_modified_functions(
    commit_diff: str, functions_by_file: dict[str, list[FileToFunction]], project: str
) -> dict[str, list[str]]:
    """
    Collect all functions affected by the given patch.
    """
    patch = PatchSet(commit_diff)
    affected_functions: set[str] = set()
    globally_changed_files: set[str] = set()
    relevant_harnesses: set[str] = set()

    # for each file that had some kind of change in the commit diff
    for patched_file in patch:
        # find all functions in this file
        patched_path = patched_file.path
        matched_file_funcs = None
        for fpath in functions_by_file:
            if fpath.endswith(patched_path):
                matched_file_funcs = functions_by_file[fpath]
                break
        log.info(
            f"Changed file {patched_path} of project {project} is associated with the following functions: {matched_file_funcs}"
        )
        if not matched_file_funcs:
            continue

        # get the function ranges of all functions in the affected file
        func_ranges = [(func.function_name, func.start_line, func.end_line) for func in matched_file_funcs]
        # use the function ranges to determine whether the change occured in a function or globally
        # Collect all modified lines in this file (line numbers in target file) as a set of integers
        modified_lines: set[int] = set()
        for hunk in patched_file:
            for line in hunk:
                if line.is_added and line.target_line_no is not None:
                    modified_lines.add(line.target_line_no)
                elif line.is_removed and line.source_line_no is not None:
                    modified_lines.add(line.source_line_no)
        log.info(
            f"In file {patched_path} of project {project} the following lines were changed: {sorted(modified_lines)}"
        )

        # match changed lines with functions or global changes
        for mod_line in modified_lines:
            hit_function = False
            for name, start, end in func_ranges:
                if start <= mod_line <= end:
                    affected_functions.add(name)
                    relevant_harnesses.update(get_harness_for_function(functions_by_file, patched_path, name))
                    hit_function = True
            if not hit_function:
                globally_changed_files.add(patched_path)

    # cast to sorted lists for json serializability and prettier logging
    affected_functions_list = sorted(affected_functions)
    globally_changed_files_list = sorted(globally_changed_files)
    relevant_harnesses_list = sorted(relevant_harnesses)

    # more logging for easier debugging
    log.info(f"affected functions of project {project}: {affected_functions_list}")
    log.info(f"globally affected files of project {project}: {globally_changed_files_list}")
    log.info(f"relevant harnesses of project {project}: {relevant_harnesses_list}")

    return {
        "affected_functions": affected_functions_list,
        "affected_files": globally_changed_files_list,
        "harnesses": relevant_harnesses_list,
    }


def get_harness_for_function(
    file_to_function: dict[str, list[FileToFunction]], filename: str, target_func_name: str
) -> list[str]:
    """
    Given a filename and function name, return the harness (reached_by_fuzzers) list.
    Returns empty list if not found.
    """
    functions = file_to_function.get(filename)
    if not functions:
        return []

    for func in functions:
        if func.function_name == target_func_name:
            return func.harnesses

    return []


def analyze_commit(
    project: str,
    project_path: Path,
    introspector_path: Path,
    commit: str,
    output_path: Path,
    single_output: bool = False,
) -> dict[str, list[str]]:
    """
    Analyze a commit for changed lines and return any harness that reaches any of the changed lines.
    Write results to file.
    Results include requested API data and affected files, functions and responsible harnesses
    """

    if not project_path.exists():
        raise FileNotFoundError(f"Project {project} does not exist at path: {project_path}")

    file_to_function_mapping = get_all_functions(project, introspector_path)

    # diffs against a merge commit's first parent -> isolated the changes introduced by the merge
    # might be a huge diff for big merges
    # but easier and more reliable than tracking down all commits of the merging branch
    commit_diff = subprocess.check_output(["git", "-C", project_path, "diff", commit, f"{commit}~"], text=True)

    data = collect_modified_functions(commit_diff, file_to_function_mapping, project)
    output_file = output_path if single_output else output_path / project / commit / "affected_stuff_and_harnesses.json"
    log.info(f"Writing results of affected functions, files, harnesses to {output_file}")
    write_json_to_file(data, output_file)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a project with introspector data and commit info.")

    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--project_path", type=Path, required=True)
    parser.add_argument("--introspector_path", type=Path, required=True)
    parser.add_argument("--commit", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--single-output", action="store_true")

    args = parser.parse_args()

    args.introspector_path.resolve().mkdir(parents=True, exist_ok=True)
    LOGGING_PROVIDER.init_logging(args.introspector_path.resolve())

    analyze_commit(
        project=args.project,
        project_path=args.project_path,
        introspector_path=args.introspector_path.resolve(),
        commit=args.commit,
        output_path=args.output,
        single_output=args.single_output,
    )
