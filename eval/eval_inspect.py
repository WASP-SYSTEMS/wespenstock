"""Inspect output directories of evaluation run."""

from __future__ import annotations

import argparse
import json
from functools import cached_property
from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from tabulate import tabulate

CRS_STARTED = "Entering analyzer agent..."
REPRODUCER_FOUND = "The reproducer triggered successfully."
VULNERABILITY_FOUND = "Analyzer identified a potential vulnerability of type"
INVALID_TOOL_NAME = "could not be found in the registered tools"
INVALID_TOOL_PARAMS = ", in _to_args_and_kwargs"
NO_TOOL_CALLED = "You have not performed any tool calls."


def load_input_file(file: Path) -> dict[str, list[str]]:
    """
    Load file containing model names with output directories.
    """

    with open(file, "r+t", encoding="utf-8") as f:
        return json.load(f)


class EvalResult(BaseModel):
    """Result of the evaluation of one instance."""

    total: int
    reproducer_found: int
    marked_as_vulnerable: int
    invalid_tool_name: int
    invalid_tool_params: int
    no_tool_called: int
    total_tool_failure: int

    @staticmethod
    def from_log_data(log_data: LogData) -> EvalResult:
        """Create from log data."""
        return EvalResult(
            total=log_data.runs,
            reproducer_found=1 if log_data.reproducers_found > 0 else 0,
            marked_as_vulnerable=1 if log_data.marked_as_vulnerable > 0 else 0,
            invalid_tool_name=log_data.invalid_tool_name,
            invalid_tool_params=log_data.invalid_tool_params,
            no_tool_called=log_data.no_tool_called,
            total_tool_failure=log_data.total_tool_failure,
        )

    def add_log_data(self, log_data: LogData) -> None:
        """Add log data to result."""
        # pylint: disable=no-member
        self.total += log_data.runs
        self.reproducer_found += 1 if log_data.reproducers_found > 0 else 0
        self.marked_as_vulnerable += 1 if log_data.marked_as_vulnerable > 0 else 0
        self.invalid_tool_name += log_data.invalid_tool_name
        self.invalid_tool_params += log_data.invalid_tool_params
        self.no_tool_called += log_data.no_tool_called
        self.total_tool_failure += log_data.total_tool_failure


class LogData:
    """Result of one run on one instance (e.g. commit or function)."""

    def __init__(self, log_data: str):
        self._log_data = log_data

    @cached_property
    def runs(self) -> int:
        """Number of CRS runs."""
        return self._log_data.count(CRS_STARTED)

    @cached_property
    def reproducers_found(self) -> int:
        """Number of reproducer was found."""
        return self._log_data.count(REPRODUCER_FOUND)

    @cached_property
    def marked_as_vulnerable(self) -> int:
        """Number of vulnerabilities  detected."""
        return self._log_data.count(VULNERABILITY_FOUND)

    @cached_property
    def invalid_tool_name(self) -> int:
        """Number of tool calls with invalid name."""
        return self._log_data.count(INVALID_TOOL_NAME)

    @cached_property
    def invalid_tool_params(self) -> int:
        """Number of tool calls with invalid params."""
        return self._log_data.count(INVALID_TOOL_PARAMS)

    @cached_property
    def no_tool_called(self) -> int:
        """Number of times no tool was called."""
        return self._log_data.count(NO_TOOL_CALLED)

    @cached_property
    def total_tool_failure(self) -> int:
        """Total failed too calls."""
        return self.invalid_tool_name + self.invalid_tool_params + self.no_tool_called


class ResultsPerModel:
    """Results of one model."""

    def __init__(self, model_name: str, out_dirs: list[Path]):
        self.model_name = model_name
        self.out_dirs = out_dirs
        self.result: dict[str, EvalResult] = {}

        self.max_logs: int | None = None

    def to_pandas(self) -> pd.DataFrame:
        """Convert results to pandas data frame."""
        return pd.DataFrame.from_dict(data=self.to_dict(), orient="index")

    def to_dict(self) -> dict[str, dict[str, dict[str, int]]]:
        """Get results as json."""
        return {k: v.model_dump() for k, v in self.result.items()}

    def load(self, max_logs: int | None = None) -> None:
        """Load results from output directories."""
        self.max_logs = max_logs
        for out_dir in self.out_dirs:
            self._load_out_dir(out_dir)

    def _load_out_dir(self, out_dir: Path) -> None:
        """Load data from one output dir"""
        for instance_dir in out_dir.iterdir():
            if instance_dir.is_dir():
                self._load_instance(instance_dir)

    def _load_instance(self, instance_dir: Path) -> None:
        """Load data from one instance (e.g. commit or function)."""
        for log_dir in instance_dir.iterdir():
            if (
                self.max_logs is not None
                and instance_dir.name in self.result
                and self.result[instance_dir.name].total >= self.max_logs
            ):
                break
            if (log_data_text := self._get_all_log_data(log_dir)) != "":
                print(f"Loading logs from {log_dir.as_posix()}")
                log_data = LogData(log_data_text)
                if instance_dir.name in self.result:
                    self.result[instance_dir.name].add_log_data(log_data)
                else:
                    self.result[instance_dir.name] = EvalResult.from_log_data(log_data)

    @staticmethod
    def _get_all_log_data(log_dir: Path) -> str:
        """Aggregate all log data."""
        log_data = ""
        for file in log_dir.iterdir():
            if file.is_file():
                if file.suffix == ".md":
                    log_data += file.read_text(encoding="utf-8")
                if file.name.startswith("crs_") and file.suffix == ".log":
                    log_data += file.read_text(encoding="utf-8")

        return log_data


def main() -> None:
    """main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path, help="json file containing model names and output directories")
    parser.add_argument("--max-logs", type=int, default=None, help="Maximum number of logs to inspect")
    parser.add_argument("--json", type=Path, default=None, help="Output results to json file.")
    args = parser.parse_args()

    results: list[ResultsPerModel] = []

    for model_name, out_dirs in load_input_file(args.input_file).items():
        result = ResultsPerModel(model_name, [Path(out_dir) for out_dir in out_dirs])
        result.load(args.max_logs)
        results.append(result)

    results_dict = {}

    for r in results:
        print()
        print()
        print(r.model_name)
        print(tabulate(r.to_pandas(), headers="keys", tablefmt="psql"))  # type: ignore
        results_dict[r.model_name] = r.to_dict()

    if args.json is not None:
        args.json.write_text(json.dumps(results_dict, indent=2))


if __name__ == "__main__":
    main()
