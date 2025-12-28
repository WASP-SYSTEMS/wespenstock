"""
Parse CRS test series log files to create a Dataframe that can be used for further processing
(e.g. creating plots and tables)
"""

import json
import os
import re
from collections import defaultdict
from numbers import Integral
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import TypeVar
from typing import cast

from pydantic import BaseModel
from pydantic import Field

# List of tools, errors, and abbreviations used when creating graphs.
# Defined at the top to simplify updates (e.g., renaming tools).

# Tools
TOOL_NAMES_ANALYZER = ["GetSymbolDefinition", "FindReferences", "SubmitVulnerabilityReport"]
TOOL_NAMES_VERIFIER = ["GetSymbolDefinition", "FindReferences", "generate_crashing_reproducer"]

# Metrics to be auto-initialized with 0 when parsing json files
EXPECTED_METRICS = [
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "tool_calls_total",
    "tool_calls_success",
    "tool_calls_failure",
    "motivator_node_calls",
    "secure",
    "vulnerable",
    "llm_invocations",
]

# Abbreviations so that tables and stargraphs do not overlap
NAME_MAP = {
    "series": "Series",
    "tool_calls_total": "Tool Total",
    "tool_calls_success": "Tool S",
    "tool_calls_failure": "Tool F",
    "motivator_node_calls": "Motivator",
    "total_tokens": "Tokens",
    "prompt_tokens": "Prompt",
    "completion_tokens": "Completion",
    "successful_povs": "PoV_S",
    "failed_povs": "PoV_F",
    "GetSymbolDefinition": "getSym",
    "FindReferences": "findRef",
    "SubmitVulnerabilityReport": "subReport",
    "generate_crashing_reproducer": "genPoV",
    "SearchSymbol": "getSym",
    "generate_python_pov_and_execute": "genPoV",
    "successful_samples": "s_samp",
    "failed_samples": "f_samp",
    "secure": "sec",
    "vulnerable": "vuln",
    "llm_invocations": "LLM",
}

# Known critical error identifiers (simplified or full phrases)
COMMON_ERRORS = [
    "ContextWindowExceededError",
    "BadRequestError",
    "Recursion limit",
    "InvalidRequestError",
    "RateLimitError",
    "ToolException",
    "Invalid harness ID",
    "APITimeoutError",
    "No such file or directory",
    "Tried to call tool",
    "'utf-8' codec can't encode",
    "'utf-8' codec can't decode",
    "Hosted_vllmException",
]


class CommitLogInfo(BaseModel):
    """
    Basic Information that every json log should contain
    """

    commit: str
    file: Path
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    tool_calls_total: int
    tool_calls_success: int
    tool_calls_failure: int
    motivator_node_calls: int
    tool_calls_per_tool: dict[str, int]
    llm_invocations: int
    errors: dict[str, int] = Field(default_factory=dict)


class AnalyzerLogInfo(CommitLogInfo):
    """
    Additional Analyzer-specific information from json logs
    """

    secure: int
    vulnerable: int


class VerifierLogInfo(CommitLogInfo):
    """
    Additional Verifier-specific information from json logs
    """

    successful_povs: int
    failed_povs: int
    total_samples: int
    successful_samples: int
    failed_samples: int


class TestSeriesInfo(BaseModel):
    """
    Aggregated information about a whole test series
    """

    folder: Path
    analyzer_info: list[AnalyzerLogInfo]
    verifier_info: list[VerifierLogInfo]


T = TypeVar("T", bound=CommitLogInfo)


def get_file_count(input_dir: Path, prefix: str) -> int:
    """
    Get number of files in a directory (and its subdirs) that match the exact filename pattern:
    <prefix>_YYYY-MM-DDTHH:MM:SS (no extension).
    Example: pov_success_2025-07-18T14:41:09
    """
    pattern = re.compile(rf"^{re.escape(prefix)}_\d{{4}}-\d{{2}}-\d{{2}}T\d{{2}}:\d{{2}}:\d{{2}}$")
    count = sum(1 for path in input_dir.rglob("*") if path.is_file() and pattern.fullmatch(path.name))
    return count


def get_json_data(file_path: Path) -> Any:
    """
    Return json log contents
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        data = None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        data = None
    except OSError as e:
        print(f"OS error when opening {file_path}: {e}")
        data = None
    return data


# pylint: disable=too-many-branches
def group_and_sum_models(logs: List[T], model_cls: Type[T]) -> List[T]:
    """
    Summarize values per commit and per complete test series.
    Includes summing of errors and tool_calls_per_tool.

    Returns a list where the first element is the summary (commit == "summary"),
    followed by one instance per commit (chronological order).
    """
    # keyed by commit -> aggregated data dict
    grouped: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"tool_calls_per_tool": defaultdict(int), "errors": defaultdict(int)}
    )

    # keep track of which numeric keys / tools / errors we saw
    numeric_keys = set()
    tool_keys = set()
    error_keys = set()

    # Aggregate values per commit (ignore any incoming 'summary' rows to avoid double counting)
    for log in logs:
        commit = log.commit
        if commit == "summary":
            # skip pre-aggregated summary rows
            continue
        g = grouped[commit]

        # model_dump -> dict of fields
        for field, value in log.model_dump().items():
            if field in {"commit", "file", "tool_calls_per_tool", "errors"}:
                continue
            if isinstance(value, Integral):
                numeric_keys.add(field)
                g[field] = g.get(field, 0) + int(value)

        # aggregate per-tool counts
        for tool, count in getattr(log, "tool_calls_per_tool", {}).items():
            tool_keys.add(tool)
            g["tool_calls_per_tool"][tool] += int(count)

        # aggregate errors
        for err, count in getattr(log, "errors", {}).items():
            error_keys.add(err)
            g["errors"][err] += int(count)

    results: List[T] = []

    # Prepare accumulators for overall summary
    overall_numeric = {k: 0 for k in numeric_keys}
    overall_tool_calls: dict[str, int] = defaultdict(int)
    overall_errors: dict[str, int] = defaultdict(int)

    # Build per-commit instances in stable (chronological) order
    for commit in sorted(grouped.keys()):  # ISO8601 timestamps sort lexicographically
        data = grouped[commit]

        # Build inst_data using numeric_keys to ensure consistent fields (default 0 if missing)
        inst_data: Dict[str, Any] = {k: int(data.get(k, 0)) for k in numeric_keys}

        # include any other non-numeric fields that might be present in data
        # (rare, but keep original behavior)
        for k, v in data.items():
            if k in {"tool_calls_per_tool", "errors"}:
                continue
            if k not in inst_data:
                inst_data[k] = v

        inst_data["commit"] = commit
        inst_data["file"] = Path(f"{commit}.json")
        inst_data["tool_calls_per_tool"] = dict(data["tool_calls_per_tool"])
        inst_data["errors"] = dict(data["errors"])

        # instantiate model (cast numeric types to int)
        for k in list(inst_data.keys()):
            if isinstance(inst_data[k], Integral):
                inst_data[k] = int(inst_data[k])

        inst = model_cls(**inst_data)
        results.append(inst)

        # accumulate into overall
        for k in numeric_keys:
            overall_numeric[k] += int(inst_data.get(k, 0))
        for tool, c in inst_data["tool_calls_per_tool"].items():
            overall_tool_calls[tool] += int(c)
        for err, c in inst_data["errors"].items():
            overall_errors[err] += int(c)

    # Build summary (overall) instance
    summary_data: Dict[str, Any] = {k: int(v) for k, v in overall_numeric.items()}
    summary_data["commit"] = "summary"
    summary_data["file"] = Path("summary.json")
    summary_data["tool_calls_per_tool"] = dict(overall_tool_calls)
    summary_data["errors"] = dict(overall_errors)

    # insert summary first so plotting code that uses info_list[0] finds the summary
    results.insert(0, model_cls(**summary_data))

    return results


def commit_is_vulnerable(file_path: Path) -> bool:
    """
    Check analyze.json for a vulnerability report -> non-empty vuln_functions field in at least one report
    """
    data = get_json_data(file_path)
    if not data:
        return False

    vulnerabilities = data.get("vulnerabilities", [])
    for vuln in vulnerabilities:
        report = vuln.get("report", {})
        if report.get("vuln_functions"):  # non-empty list
            return True

    return False
    # raise FileNotFoundError(f"File {file_path} is empty or does not exist")


# pylint: disable=too-many-branches
def extract_data_from_json_file(file_path: Path, agent: str) -> CommitLogInfo:
    """
    Parse a Analyzer or Verfifier Log file (json format) to parse tokens, tool calls etc
    """
    data = get_json_data(file_path)

    # parse commit from input path
    if "analyzer" in str(file_path):
        commit = file_path.parents[1].name
    else:
        commit = file_path.parents[1].name

    # Initialize counters
    counters = defaultdict(int)
    for metric in EXPECTED_METRICS:
        counters[metric] = 0

    tool_calls_per_tool: dict[str, int] = {}
    tool_list = []
    if agent == "analyzer":
        tool_list = TOOL_NAMES_ANALYZER
        vulnerable = commit_is_vulnerable(file_path.parent / "analyze.json")
    if agent == "verifier":
        tool_list = TOOL_NAMES_VERIFIER
    for tool in tool_list:
        tool_calls_per_tool[tool] = 0

    # Assuming messages is a list of dicts
    for message in data.get("messages", []):
        # Human messages and tool answers
        if message["type"] == "tool":
            if message["status"] == "success":
                counters["tool_calls_success"] += 1
            else:
                counters["tool_calls_failure"] += 1

        elif message["type"] == "ai":
            # count number of invocations
            counters["llm_invocations"] += 1

            # collect tokens
            counters["completion_tokens"] += (
                message.get("response_metadata", {}).get("token_usage", {}).get("completion_tokens", 0)
            )
            counters["prompt_tokens"] += (
                message.get("response_metadata", {}).get("token_usage", {}).get("prompt_tokens", 0)
            )
            counters["total_tokens"] += (
                message.get("response_metadata", {}).get("token_usage", {}).get("total_tokens", 0)
            )

            # check for tool calls
            tool_calls = message.get("additional_kwargs", {}).get("tool_calls", [])
            for tool_call in tool_calls:
                counters["tool_calls_total"] += 1
                tool_name = tool_call.get("function", {}).get("name")
                if tool_name:
                    tool_calls_per_tool[tool_name] = tool_calls_per_tool.get(tool_name, 0) + 1

        elif message["type"] == "human":
            if "You have not performed any tool calls. To make progress, you must call tools." in message["content"]:
                counters["motivator_node_calls"] += 1

    # reformat tool call counts
    tools = {}
    for tool, count in tool_calls_per_tool.items():
        tools[tool] = count

    # collect fields that are included in every agent
    return_data = CommitLogInfo(
        commit=commit,
        file=file_path,
        total_tokens=counters["total_tokens"],
        prompt_tokens=counters["prompt_tokens"],
        completion_tokens=counters["completion_tokens"],
        tool_calls_total=counters["tool_calls_total"],
        tool_calls_success=counters["tool_calls_success"],
        tool_calls_failure=counters["tool_calls_failure"],
        motivator_node_calls=counters["motivator_node_calls"],
        tool_calls_per_tool=tool_calls_per_tool,
        llm_invocations=counters["llm_invocations"],
    )

    # collect all necessary data fields per agent
    if agent == "verifier":
        total_samples, successful_samples = count_samples(file_path.parents[1])
        return_data = VerifierLogInfo(
            **return_data.model_dump(),
            successful_povs=get_file_count(file_path.parent, "pov_success"),
            failed_povs=get_file_count(file_path.parent, "pov_failed"),
            total_samples=total_samples,
            successful_samples=successful_samples,
            failed_samples=total_samples - successful_samples,
        )
    elif agent == "analyzer":
        return_data = AnalyzerLogInfo(
            **return_data.model_dump(), secure=0 if vulnerable else 1, vulnerable=1 if vulnerable else 0
        )

    return return_data


def count_samples(commit_dir: Path) -> tuple[int, int]:
    """Evaluate the given directory for number for samples and pov_successful_TIMESTAMP files."""
    total_samples = 0
    successful_samples = 0

    # Iterate through sample directories
    for sample_path in commit_dir.iterdir():
        if not os.path.isdir(sample_path):
            continue
        total_samples += 1

        # Check for pov_successful_TIMESTAMP files
        if any(f.startswith("pov_success_") for f in os.listdir(sample_path)):
            successful_samples += 1
    return (total_samples, successful_samples)


def process_logs(input_path: Path, agent_type: str) -> List[CommitLogInfo]:
    """
    Process log folder of one test series to analyze state dump JSON files.
    """

    # choose file pattern and class based on agent
    if agent_type == "verifier":
        pattern = "pov_builder_agent_state_dump*.json"
        model_cls: Type[CommitLogInfo] = VerifierLogInfo
    elif agent_type == "analyzer":
        pattern = f"{agent_type}_agent_state_dump*.json"
        model_cls: Type[CommitLogInfo] = AnalyzerLogInfo  # type: ignore[no-redef]
        # i tried to fix this but won't work
    else:
        raise NotImplementedError(f"Unknown agent_type: {agent_type}")

    # find all relevant log files
    json_files = list(input_path.rglob(pattern))
    if not json_files:
        return []

    # collect all log data per file
    data_per_json_file: List[CommitLogInfo] = []
    for json_file in json_files:
        cur_data_frame = extract_data_from_json_file(json_file, agent_type)  # inferred CommitLogInfo subclass
        crs_log = find_crs_logs(json_file, agent_type)
        errors = process_log_logs(crs_log)
        cur_data_frame.errors = errors
        data_per_json_file.append(cur_data_frame)

    if not data_per_json_file:
        raise ValueError(f"No valid data rows extracted from logs in {input_path}.")

    # aggregate values
    grouped: List[CommitLogInfo] = group_and_sum_models(data_per_json_file, model_cls)
    return grouped


def find_crs_logs(json_path: Path, agent_type: str) -> Path:
    """
    Given a JSON file path, find all crs*.log files in the same directory.
    Return the one that maps to the given agent_type
    """
    input_path = json_path.parent
    log_files = list(input_path.glob("crs*.log"))
    for log in log_files:
        if detect_agent_type_in_log(log) == agent_type:
            return log
    raise FileNotFoundError(f"No crs*.log file in Path {input_path}")


def detect_agent_type_in_log(log_file: Path) -> str:
    """
    Parse .log file to check whether it belongs to an analyzer or a verifier invocation
    (agent type cannot be derived from the file name atm)
    """
    try:
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                if "CRS arguments:" in line:
                    if "command='pov'" in line:
                        return "verifier"
                    if "command='analyze'" in line:
                        return "analyzer"
    except (FileNotFoundError, PermissionError, OSError) as e:
        print(f"Could not read {log_file}: {e}")
    return "unknown"


def process_log_logs(log_file: Path) -> dict[str, int]:
    """
    Analyze .log files in the given folder to count occurrences of known critical errors.
    Returns a dictionary mapping error types to their occurrence count.
    """
    errors = {}
    for error in COMMON_ERRORS:
        errors[error] = 0
    errors["Unknown"] = 0
    errors["total"] = 0

    try:
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                if "[CRITICAL]" in line:
                    matched_any = False
                    for error in COMMON_ERRORS:
                        if error in line:
                            errors[error] += 1
                            matched_any = True
                    if not matched_any:
                        errors["Unknown"] += 1

                    # Count line once for total
                    errors["total"] += 1

    except (FileNotFoundError, PermissionError, OSError) as e:
        print(f"Error processing {log_file.name}: {e}")
    return errors


def get_test_series_data(test_series_path: Path) -> list[TestSeriesInfo]:
    """
    Aggregate all log information in the test series (json and log files) for both Analyzer and Verfier
    """
    # test series path will contains one test series
    test_series_info: list[TestSeriesInfo] = []
    print(f"Processing folder {test_series_path}")
    if not test_series_path.exists() or not test_series_path.is_dir():
        print(f"Invalid directory: {test_series_path}")

    try:
        analyzer_json_results: List[AnalyzerLogInfo] = cast(
            List[AnalyzerLogInfo], process_logs(test_series_path, "analyzer")
        )
        verifier_json_results: List[VerifierLogInfo] = cast(
            List[VerifierLogInfo], process_logs(test_series_path, "verifier")
        )
    except FileNotFoundError as e:
        print(f"Folder {test_series_path} does not contain log files: {e}")
    except ValueError as e:
        print(f"Error processing log files in folder {test_series_path}: {e}")

    series = TestSeriesInfo(
        folder=test_series_path, analyzer_info=analyzer_json_results, verifier_info=verifier_json_results
    )
    test_series_info.append(series)

    return test_series_info


def split_series_per_commit(series_list: List[TestSeriesInfo]) -> dict[str, List[TestSeriesInfo]]:
    """
    Split a list of TestSeriesInfo objects into commit-wise sub-series.

    Returns:
        dict[commit_id, List[TestSeriesInfo]]
    """
    commit_map: dict[str, List[TestSeriesInfo]] = defaultdict(list)

    for series in series_list:
        # collect all commit IDs present in analyzer and verifier
        commits = {info.commit for info in series.analyzer_info} | {info.commit for info in series.verifier_info}

        for commit in commits:
            # filter infos for this commit
            analyzer_info = [info for info in series.analyzer_info if info.commit == commit]
            verifier_info = [info for info in series.verifier_info if info.commit == commit]

            sub_series = TestSeriesInfo(
                folder=series.folder,
                analyzer_info=analyzer_info,
                verifier_info=verifier_info,
            )
            commit_map[commit].append(sub_series)

    return commit_map
