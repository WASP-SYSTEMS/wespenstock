"""
Parse CRS test series log files to create stragraphs and tables
"""

import argparse
import itertools
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from eval.parse_logs import CommitLogInfo
from eval.parse_logs import TestSeriesInfo
from eval.parse_logs import get_test_series_data
from eval.parse_logs import split_series_per_commit

BASE_VALUE_GROUPS = {
    "Tokens": ["total_tokens", "prompt_tokens", "completion_tokens"],
    "LLM Interactions": [
        "tool_calls_total",
        "tool_calls_success",
        "tool_calls_failure",
        "motivator_node_calls",
        "llm_invocations",
    ],
    "Errors": [],  # dynamically filled
    "Prediction": ["secure", "vulnerable"],
    "PoVs": ["successful_povs", "failed_povs"],
}


def plot_testseries_barplots(series_list: list[TestSeriesInfo], output_path: Path) -> None:
    """
    Create bar plots for analyzer and verifier info.

    - For 'summary' mode: plots only the summary instance per series (includes errors)
    - For per-commit mode: create one plot per commit per series (info only, no errors)
    """
    agents = ["analyzer", "verifier"]

    cmap = plt.get_cmap("tab10")
    cmap_colors = [cmap(i / 9) for i in range(10)]  # tab10 has 10 colors

    # --- Setup figure and axes ---
    ncols = _compute_max_columns(agents)
    fig, axes = plt.subplots(
        nrows=len(agents),
        ncols=ncols,
        figsize=(6 * ncols, 4 * len(agents)),
        squeeze=False,
    )

    for row, agent in enumerate(agents):
        _plot_agent_row(axes[row], agent, series_list, cmap_colors)

    _merge_legends(fig, axes)

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    print(f"Saving plot to: {output_path}")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


# --- Helpers for plot_test_series_barplots--- #
def _compute_max_columns(agents: list[str]) -> int:
    return max(
        sum(
            not ((name == "Prediction" and agent != "analyzer") or (name == "PoVs" and agent != "verifier"))
            for name in BASE_VALUE_GROUPS
        )
        for agent in agents
    )


def _plot_agent_row(
    axes_row: Sequence[mpl.axes.Axes],
    agent: str,
    series_list: list[TestSeriesInfo],
    cmap_colors: list[tuple[float, float, float, float]],
) -> None:
    """Plot all groups (columns) for a single agent (row)."""
    value_groups = {
        name: keys
        for name, keys in BASE_VALUE_GROUPS.items()
        if not ((name == "Prediction" and agent != "analyzer") or (name == "PoVs" and agent != "verifier"))
    }

    for col, (group_name, keys) in enumerate(value_groups.items()):
        ax = axes_row[col]
        ax.set_title(f"{agent.capitalize()} – {group_name}")
        if group_name == "Tokens":
            ax.set_yscale("log")

        _plot_value_group(ax, group_name, keys, agent, series_list, cmap_colors)


def _plot_value_group(
    ax: mpl.axes.Axes,
    group_name: str,
    base_keys: list[str],
    agent: str,
    series_list: list[TestSeriesInfo],
    cmap_colors: list[tuple[float, float, float, float]],
) -> None:
    """Plot one group of values (e.g. Tokens, Errors) for all series."""
    bar_width = 0.8 / len(series_list)

    # Collect ALL keys across series first
    all_keys = _collect_all_keys(base_keys, group_name, agent, series_list)
    x = np.arange(len(all_keys))

    for idx, (color, series) in enumerate(zip(itertools.cycle(cmap_colors), series_list)):
        entry = _get_entry(series, agent)
        if not entry:
            continue

        values = _extract_values(entry, group_name)
        bar_values = [values.get(k, 0) for k in all_keys]

        ax.bar(
            x + idx * bar_width,
            bar_values,
            bar_width,
            label=series.folder.name,
            color=color,
            alpha=0.7,
        )

    # Set consistent ticks and labels
    if all_keys:
        ax.set_xticks(x + bar_width * (len(series_list) - 1) / 2)
        ax.set_xticklabels(all_keys, rotation=45, ha="right")


def _collect_all_keys(
    base_keys: list[str], group_name: str, agent: str, series_list: list[TestSeriesInfo]
) -> list[str]:
    """Find all keys used across all series for this group."""
    all_keys = list(base_keys)
    for series in series_list:
        info_list = series.analyzer_info if agent == "analyzer" else series.verifier_info
        if not info_list:
            continue
        entry = info_list[0]
        if group_name == "LLM Interactions":
            for tool in getattr(entry, "tool_calls_per_tool", {}):
                tool_key = f"tool:{tool}"
                if tool_key not in all_keys:
                    all_keys.append(tool_key)
        if group_name == "Errors":
            for err in entry.errors.keys():
                if err not in all_keys:
                    all_keys.append(err)
    return all_keys


def _get_entry(series: TestSeriesInfo, agent: str) -> CommitLogInfo | None:
    """Pick the right CommitLogInfo entry depending on series type."""
    info_list = series.analyzer_info if agent == "analyzer" else series.verifier_info
    if not info_list:
        return None
    if isinstance(info_list, CommitLogInfo):
        return info_list
    if isinstance(info_list, list):
        return info_list[0]  # summary or first commit
    raise TypeError(f"Unexpected type for info_list: {type(info_list)}")


def _extract_values(entry: CommitLogInfo, group_name: str) -> dict[str, int]:
    """Extract numeric values from an entry for plotting."""
    values = {k: v for k, v in entry.model_dump().items() if isinstance(v, int)}
    if group_name == "LLM Interactions":
        values.update({f"tool:{tool}": count for tool, count in getattr(entry, "tool_calls_per_tool", {}).items()})
    if group_name == "Errors":
        values.update(entry.errors)
    return values


def _merge_legends(fig: mpl.figure.Figure, axes: np.ndarray) -> None:
    """Merge legends from all axes into a single one."""
    handles, labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.02),
    )


# --- End of helpers for plot_test_series_barplots--- #


def generate_barplots_for_test_series(output_dir: Path, test_series_dirs: list[Path], per_commit: bool) -> None:
    """
    Generate Barplots for Test Series
    Each Test Series is displayed in its own color
    Use per_commit = True to generate one barplot per commit
    """
    # create output dir if it does not exist already
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    elif not output_dir.is_dir():
        raise NotADirectoryError(f"{output_dir} exists but is not a directory.")

    # parse all logs
    dataframes: list[TestSeriesInfo] = []
    for test_series in test_series_dirs:
        dataframes += get_test_series_data(test_series)
    commit_series_map = split_series_per_commit(dataframes)

    if per_commit:  # create plot for each commit if requested
        # sort by commit and create plot for each commit including summary
        for commit, sub_series_list in commit_series_map.items():
            output_path = output_dir / f"barplot-{commit}.png"
            plot_testseries_barplots(sub_series_list, output_path)
    else:
        # create summary plot for whole test series
        plot_testseries_barplots(commit_series_map["summary"], output_path=output_dir / "barplot-summary.png")


def main() -> None:
    """
    Get directory of a test series as input
    Produce barplots displaying CRS/LLM performance as output
    """
    # get test series
    parser = argparse.ArgumentParser(description="Analyze CRS log files to create barplots.")
    parser.add_argument("output_path", type=Path, help="Path to save barplots to")
    parser.add_argument("input_dirs", type=Path, nargs="+", help="Path to test series.")
    parser.add_argument(
        "--per-commit", action="store_true", help="Verbose Output: Graph per agent x commit combination"
    )

    args = parser.parse_args()
    output_dir = args.output_path.resolve()
    generate_barplots_for_test_series(output_dir, args.input_dirs, args.per_commit)


if __name__ == "__main__":
    main()
