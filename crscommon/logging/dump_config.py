"""Display the current configuration of the CRS at startup."""

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

from git import Repo

from ..settings.settings import SETTINGS
from .settings import DUMP_ALL_CONFIG


class GitInfo(TypedDict):
    "Information about a Git repository for get_git_info()."

    repo: Path | None
    worktree: Path | None
    commit: str | None
    branch: str | None


def str_or_na(v: object) -> str:
    "Convert a Python value to a string, with a special case for None."
    return "N/A" if v is None else str(v)


def get_git_info() -> GitInfo:
    """
    Identify the Git repository this source file is located in and gather basic information about it.
    """
    result: GitInfo = {"repo": None, "worktree": None, "commit": None, "branch": None}
    try:
        repo = Repo(Path(__file__), search_parent_directories=True)
        result["repo"] = Path(repo.git_dir)
        result["worktree"] = Path(repo.working_tree_dir) if repo.working_tree_dir is not None else None
        result["commit"] = repo.head.commit.hexsha
        result["branch"] = repo.active_branch
    except Exception:  # pylint: disable=W0718
        pass
    return result


def get_pip_info() -> list[str]:
    """
    Get information about the currently installed Python packages.
    """
    return subprocess.run(["pip", "freeze"], check=True, capture_output=True, text=True).stdout.splitlines()


def dump_config(log: logging.Logger) -> None:
    """
    Record the current configration and environment of the CRS.

    Outputs various parameters of the current Python process and all settings.
    Must called after the logging is initialized and settings are loaded.
    All settings containing the word "key" are censored.

    log: Logger to be used.
    """

    log_lines = [
        "Process configuration:",
        f"    Command line: {shlex.join(sys.argv)}",
        f"    Python: {sys.executable}",
        f"    Script: {Path(sys.argv[0]).absolute()}",
        f"    Working dir: {os.getcwd()}",
    ]
    log.info("\n".join(log_lines))

    git_info = get_git_info()
    log_lines = [
        "CRS Git info:",
        f"    Repository: {str_or_na(git_info['repo'])}",
        f"    Work tree: {str_or_na(git_info['worktree'])}",
        f"    Branch: {str_or_na(git_info['branch'])}",
        f"    Commit: {str_or_na(git_info['commit'])}",
    ]
    log.info("\n".join(log_lines))

    if DUMP_ALL_CONFIG.get():
        # This one takes about half a second, which interactive users might dislike.
        pip_info = get_pip_info()
        log.info(f"Installed Python packages: {shlex.join(pip_info)}")

    log_lines = ["Effective settings:"]
    for name, value in sorted(SETTINGS.values.items()):
        if "key" in name.lower() and isinstance(value, str):
            uncensored_part_len = int(len(value) / 4)
            value = value[:uncensored_part_len] + "*" * (len(value) - uncensored_part_len)
        log_lines.append(f"    {name}: {value}")
    log.info("\n".join(log_lines))
