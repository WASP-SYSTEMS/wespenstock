# pylint: disable=line-too-long,too-many-lines
"""
Script to conviently transform OSS-Fuzz projects into Challenge Projects
"""

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from io import BytesIO
from multiprocessing import Pool
from pathlib import Path
from pathlib import PurePosixPath
from subprocess import CalledProcessError
from typing import Any
from typing import Iterable
from urllib.parse import urlsplit

import yaml
from git import GitError
from git import Repo
from git import exc as git_exc
from pydantic import ValidationError

from crscommon.logging.dump_config import dump_config
from crscommon.logging.logging_provider import LOGGING_PROVIDER
from crscommon.settings import load_settings
from oss_fuzz_integration.introspector_api import HarnessSourceAndExecutableResponse
from oss_fuzz_integration.introspector_api import fetch_api_data

log = LOGGING_PROVIDER.new_logger("PROJECT_PREPARATION", hook_exception=True)

PROJECTS_FORCING_INTERACTION = ["aiohttp", "libfido2", "stb", "freeimage"]


class InformationNotFoundError(Exception):
    """Exception raised when a specific information cannot be received in the expected way"""


def verify_api_to_actual_harness_file_mapping(
    cp_dir: Path, out_path: Path, api_harnesses: dict[str, dict[str, str]]
) -> dict[str, dict[str, str]]:
    """
    Check whether there is an executable in the out directory for each harnesses named by the API and vice versa.
    Remove all harnesses from the dict that do not have a matching binary in the out dir.

    After that, verify that all src files mapped to a harness actually exist in the specified path.
    Remove any harnesses that do not have their src file in the expected location.

    Checked again if any harnesses remain
    """
    # get list of harness binaries form /out
    out_harness_names = list_harness_binaries(out_path)
    filtered_harnesses = api_harnesses

    # check if api data and contents of /out match
    out_set = set(out_harness_names)
    log.info(f"Detected the following binaries in {out_path}: {out_harness_names}")
    api_set = set(list(api_harnesses.keys()))
    log.info(f"API returned the following harnesses for CP {cp_dir}: {api_set}")

    no_api_entry = out_set - api_set
    if no_api_entry:
        log.warning(f"The following harness binaries in {out_path} exist but do not have an API entry: {no_api_entry}")

    no_binary_present = api_set - out_set
    if no_binary_present:
        log.warning(
            f"The following harnesses were expected in {out_path} but do not have a matching binary in out: {no_binary_present}. Removing these from the project.yaml."
        )
        filtered_harnesses = {k: h for k, h in api_harnesses.items() if k not in no_binary_present}

    # check if specified src files exist
    to_remove: list[str] = []
    for harness_name, harness_info in filtered_harnesses.items():
        source_file = cp_dir / harness_info["source"]
        if not source_file.exists():
            log.warning(
                f"Source file {source_file} of harness {harness_name} does not exist in {cp_dir}. Removing harness from list."
            )
            to_remove.append(harness_name)

    # remove harnesses with non-existing src file
    for key in to_remove:
        del filtered_harnesses[key]

    # check if any harnesses remain
    if not filtered_harnesses:
        msg = f"No harness remain after comparing API responses to existing files in the CP {cp_dir}"
        log.error(msg)
        raise FileNotFoundError(msg)

    return filtered_harnesses


def list_harness_binaries(out_path: Path) -> list[str]:
    """
    Count the number of harness binaries in the project's output folder
    Returns the list of harnesss names
    """
    out_harnesses = [p for p in out_path.iterdir() if p.is_file() and os.access(p, os.X_OK)]
    out_harness_names = [h.name for h in out_harnesses if h.name != "llvm-symbolizer"]

    log.info(f"Detected the following harnesses in {out_path}: {out_harness_names}")

    if not out_harness_names:  # check if there is at least one harness in /out
        msg = f"No harnesses in {out_path} after building"
        log.error(msg)
        raise FileNotFoundError(msg)
    return out_harness_names


def check_compile_commands(cp_dir: Path) -> bool:
    """
    Check whether the file work/compile_commands.json exists and is non-empty
    """
    file_path = cp_dir / "work" / "compile_commands.json"
    if not file_path.exists() or file_path.stat().st_size == 0:
        return False  # File doesn't exist or is completely empty

    try:
        with file_path.open("r", encoding="utf-8") as f:
            content = json.load(f)
            return bool(content)  # False if {} or []
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"Error checking compilation database {file_path} for content: {e}")
        return False


def get_main_repo(project_yaml: Path) -> str:
    """Parse repo url from project.yaml - used for commit specific CPs"""
    with project_yaml.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    url = data.get("main_repo")

    if url is None:
        raise ValueError(f"'main_repo' not found in {project_yaml}")
    return url


def read_yaml_file(path: Path) -> dict:
    """Read a yaml file"""
    with path.open(encoding="utf8") as file:
        thing = yaml.safe_load(file)

        assert isinstance(thing, dict)

        return thing


def write_yaml_file(data: Any, path: Path) -> None:
    """write a file"""
    with path.open("w", encoding="utf8") as file:
        yaml.dump(data, file, default_flow_style=False, indent=4)


def load_project_list(filename: Path, *, failed: bool) -> set[str]:
    """Load project names from a file into a set."""
    if filename.exists():
        with filename.open(encoding="utf-8") as file:
            if failed:
                # When failed=True, extract project names before the colon
                projects = set()
                for line in file:
                    line = line.strip()
                    no_timestamp = line.split("]", 1)[1].strip()
                    name = no_timestamp.split(":", 1)[0].strip()
                    projects.add(name)
                return projects

            # When failed=False, return project names from each line directly
            return {line.strip() for line in file if line.strip()}

    return set()  # Return an empty set if file doesn't exist


def write_transformation_status_to_file(
    filename: Path, content: str, details: str | None = None, log_error: bool = False
) -> None:
    """Append content to the specified file. Add details if necessary. Log details as error if requested"""
    if log_error and details is not None:
        log.error(details)
    with filename.open("a", encoding="utf-8") as file:
        if details is not None:
            file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {content}: {details}\n")
        else:
            file.write(f"{content}\n")


def remove_depth_1(dockerfile_path: Path, project_name: str) -> None:
    """
    Remove --depth=1 parameter for the project's "git clone" command in its Dockerfile.

    We cannot remove all occurrences of --depth=1 because there are potentially many
    dependencies that we do not need in full (save disk space).
    Try to catch any order of parameters and multi-line clone.
    """
    with dockerfile_path.open(encoding="utf-8") as f:
        lines = f.readlines()

    updated_lines = []
    buffer = []
    in_clone = False

    for line in lines:
        stripped = line.rstrip()

        # Detect start of a git clone command
        if re.search(r"\bgit\s+clone\b", stripped):
            in_clone = True

        if in_clone:
            buffer.append(line)

            # If the command ends (no trailing backslash)
            if not stripped.endswith("\\"):
                full_cmd = "".join(buffer)

                # If it's cloning our project, remove depth flag
                if re.search(rf"https?://\S*{re.escape(project_name)}", full_cmd, re.IGNORECASE):
                    full_cmd = re.sub(r"--depth[ =]1", "", full_cmd)
                    # Collapse extra spaces while keeping newlines
                    full_cmd = re.sub(r"[ \t]+", " ", full_cmd)

                updated_lines.append(full_cmd)
                buffer = []
                in_clone = False
        else:
            updated_lines.append(line)

    with dockerfile_path.open("w", encoding="utf-8") as f:
        f.writelines(updated_lines)


def find_git_path_via_dockerfile(dockerfile: Path, target_url: str) -> str | None:
    """
    Scan Dockerfile lines for a `git clone` command that clones a specific repository
    and return the destination directory.

    Args:
        dockerfile_lines: Iterable of Dockerfile lines (list or file object).
        target_url: The full Git repository URL to search for.

    Returns:
        Destination directory as string, or None if the repo is not cloned.
    """
    with dockerfile.open(encoding="utf-8") as f:
        lines = f.readlines()

    pattern = re.compile(rf"git\s+clone(?:\s+--\S+)*\s+{re.escape(target_url)}(?:\s+(?P<dest>\S+))?")

    for line in lines:
        if target_url not in line:
            continue

        match = pattern.search(line)
        if match:
            if match.group("dest"):
                return match.group("dest")

    # Infer from repo URL as back-up
    repo_name = PurePosixPath(urlsplit(target_url).path).name
    return repo_name.removesuffix(".git")


def prune_git_history(dockerfile: Path, commit_hash: str) -> None:
    """
    Cut off newer commits in commit-specific CPs
    """
    log.info(f"Rewriting Dockerfile {dockerfile} to stop git history at commit {commit_hash}")

    try:
        with dockerfile.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as e:
        log.error(f"Error: Could not read Dockerfile {dockerfile}: {e}")
        raise e

    work_dir = None
    work_pattern = re.compile(r"^\s*WORKDIR\s+(?P<path>\S+)(?:\s+#.*)?\s*$")
    for line in lines:
        match = work_pattern.match(line)
        if match:
            work_dir = match.group("path")
            break

    if work_dir:
        log.info(f"WORKDIR in {dockerfile} has value {work_dir}")
        if work_dir.startswith("$SRC/"):
            work_dir = work_dir.removeprefix("$SRC/")
            log.info(f"WORKDIR in {dockerfile} starts with $SRC/; using {work_dir} instead")
    else:
        log.info(f"No WORKDIR found in {dockerfile} — fallback to inserting git history pruning before 'COPY build.sh'")

    git_prune_cmd = (
        f"RUN cd $SRC/{work_dir or '.'} && "
        f"git fetch --all && "
        f"git reset --hard {commit_hash} && "
        f"git gc --prune=now\n"
    )

    modified = False
    new_lines = []
    copy_pattern = re.compile(r"^\s*COPY\s+build\.sh\b")

    for _, line in enumerate(lines):
        stripped = line.strip()

        # Insert after WORKDIR
        if work_dir and stripped.startswith("WORKDIR "):
            new_lines.append(line)
            new_lines.append(git_prune_cmd)
            modified = True
            continue

        # Fallback: insert before COPY build.sh
        if not work_dir and copy_pattern.match(stripped):
            new_lines.append(git_prune_cmd)
            modified = True

        new_lines.append(line)

    if not modified:
        log.error(f"Neither WORKDIR nor COPY build.sh were found in Dockerfile {dockerfile}")
        raise InformationNotFoundError(
            f"Neither WORKDIR nor COPY build.sh were found in Dockerfile {dockerfile}, git history cannot be pruned."
        )

    # Write back the modified Dockerfile
    with dockerfile.open("w", encoding="utf-8") as f:
        f.writelines(new_lines)


def get_default_branch(project_path: Path) -> str:
    """Get default branch because CRS defaults to main but many oss-fuzz projects use master

    Parameters:
        project_path (Path): git project directory

    Returns:
        str: name of default branch
    """

    # Navigate to the project directory and get the default branch
    branch = Repo(project_path).git.rev_parse("origin/HEAD", abbrev_ref=True)
    # Parse the output, expected result is origin/main or origin/master
    return branch.split("/")[-1]


def fetch_harness_data(project: str, language: str, retries: int = 6) -> dict[str, dict[str, str]]:
    """
    Fetches API data for harnesses (source file - binary pairs) with retries
    """
    url = "https://introspector.oss-fuzz.com/api/harness-source-and-executable"
    params = {"project": project}
    harnesses: dict[str, dict[str, str]] = {}
    try:
        api_data = fetch_api_data(url, params, HarnessSourceAndExecutableResponse.model_validate, log, retries)
        if api_data is None:
            return {}
        if api_data.result != "success":
            return harnesses

        pairs = api_data.pairs
        if not pairs:
            log.error(f"OSS-Fuzz API has no information on harnesses for project {project}")
            raise InformationNotFoundError(f"OSS-Fuzz API has no information on harnesses for project {project}")

        # parsing all harness pairs
        for p in pairs:
            src_path = str(p.source).lstrip("/")
            executable = p.executable

            if src_path == "Did-not-find-sourcefile":
                log.warning(f"Project {project} has no source file for harness {executable}")
                continue

            # the ... case only occure with Python projects atm and should be fixable like this:
            if language == "python":
                if src_path.startswith("..."):
                    src_path = f"src/{src_path.removeprefix('...')}.py"
                src_path = dot_to_path(src_path)  # api might return import strings instead of file paths

            if not src_path.startswith("src/") and not src_path.startswith(str(executable)):
                log.warning(
                    f"Probably invalid src path for harness {executable} of project {project}: {src_path}. Skipping."
                )
                continue

            harnesses[str(executable)] = {
                "name": str(executable),
                "source": str(src_path),
                "binary": f"out/{executable}",
            }

        return harnesses
    except ValidationError as e:
        log.error(
            f"API did not return all necessary information (url: {url}, params: {params}): {e.errors(include_input=True)}"
        )
        return {}
    except Exception as e:  # pylint: disable=broad-exception-caught
        log.error(f"Unexpected error processing response (url: {url}, params: {params}): {e}")
        return {}


def dot_to_path(s: str) -> str:
    """Translate Python imports to file paths"""
    # If there is a dot in the string
    if "." in s:
        # Split into path part and extension (last dot only)
        parts = s.rsplit(".", 1)
        path_part = parts[0].replace(".", "/")
        # If there was an extension, reattach it
        return f"{path_part}.{parts[1]}" if len(parts) > 1 else path_part
    # No dot, return as-is
    return s


def get_all_projects_per_language(language: str, oss_fuzz_path: Path) -> list[str]:
    """Return a list of all oss-fuzz projects of a specific language"""
    filtered_projects = []
    for p in oss_fuzz_path.iterdir():
        if not p.is_dir() or p.name == ".git":
            continue

        project_yaml_path = p / "project.yaml"
        if project_yaml_path.exists():
            data = read_yaml_file(project_yaml_path)
            if data.get("language") == language:
                filtered_projects.append(p.name)
        else:
            log.warning(f"Skipping project {p.name} because yaml file does not exist: {project_yaml_path}")

    return filtered_projects


def get_projects_to_be_transformed(args: argparse.Namespace, oss_fuzz_dir: Path) -> list[str]:
    """
    Create a list of all projects that still need to be transformed into CPs
    This is done by first getting a list of all relevant projects (e.g. all projects of a specific language)
    and then removing those that are already listed in the failed/success lists
    as well as those that are known to force dialogue during the build process
    """
    oss_fuzz_projects_path = oss_fuzz_dir / "projects"
    if args.project_name != "all":
        projects = remaining_projects = [args.project_name]
    elif args.project_name == "all" and args.language:
        projects = get_all_projects_per_language(args.language, oss_fuzz_projects_path)
    else:
        projects = [d.name for d in oss_fuzz_projects_path.iterdir() if d.is_dir()]

    assert not (args.single_cp and len(projects) > 1)

    if args.force:
        log.info(f"Detected {len(projects)} projects to be transformed, processing all due to --force")
        return projects

    # Load previously processed projects
    failed_projects = load_project_list(args.failed_list_path, failed=True)
    successful_projects = load_project_list(args.success_list_path, failed=False)

    # Remove the failed and successful projects from the projects list as well as those that force dialogue
    remaining_projects = [
        project
        for project in projects
        if project not in failed_projects
        and project not in successful_projects
        and project not in PROJECTS_FORCING_INTERACTION
    ]

    log.info(f"Detected {len(projects)} projects to be transformed. Remaining projetcs: {len(remaining_projects)}")
    return remaining_projects


def get_oss_fuzz_dir(args: argparse.Namespace, output_dir: Path) -> Path:
    """
    Determine the path of the local OSS-Fuzz clone
    """
    if args.oss_fuzz:
        oss_fuzz_dir = args.oss_fuzz.resolve()
        log.info(f"Using the OSS-Fuzz clone at {oss_fuzz_dir}")
        if not oss_fuzz_dir.exists() or not oss_fuzz_dir.is_dir():
            raise FileNotFoundError(
                f"No oss-fuzz clone at {oss_fuzz_dir}. Please run git clone https://github.com/google/oss-fuzz.git {oss_fuzz_dir}, then restart the script with the same parameters."
            )
        log.info(f"Using the following path as base OSS-Fuzz clone: {oss_fuzz_dir}")
    else:
        oss_fuzz_dir = output_dir / "oss-fuzz"
        # check if oss-fuzz is already present
        if not oss_fuzz_dir.exists() or not oss_fuzz_dir.is_dir():
            # Not present -> clone
            try:
                Repo.clone_from("https://github.com/google/oss-fuzz.git", oss_fuzz_dir)
                log.info(f"Cloned oss-fuzz into {oss_fuzz_dir}")
            except git_exc.GitError as e:
                log.error(f"Error cloning oss-fuzz into {oss_fuzz_dir}: {e}")
                sys.exit(1)
        else:
            # Present -> checkout master to undo potential time-specific checkouts
            try:
                repo = Repo(oss_fuzz_dir)
                repo.git.checkout("master")
                log.info(f"Checked out 'master' branch in {oss_fuzz_dir}")
            except git_exc.GitError as e:
                log.error(f"Failed to checkout 'master' in {oss_fuzz_dir}: {e}")
                sys.exit(1)
    return oss_fuzz_dir


# pylint: disable=too-many-branches, too-many-statements
def transform_project(
    project_name: str,
    project_repo: Repo | None,
    output_dir: Path,
    oss_fuzz_dir: Path,
    oss_fuzz_repo: Repo,
    script_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Transform one specific oss-fuzz project into a challenge project"""

    def trim_tar_prefix(tar: tarfile.TarFile, prefix: str) -> Iterable[tarfile.TarInfo]:
        "Helper: Trim a prefix from a TarFile's member names."
        for info in tar.getmembers():
            if prefix.startswith(info.name):
                continue
            assert info.name.startswith(prefix), f"{info.name} does not start with {prefix}"
            info.name = info.name[len(prefix) :]
            yield info

    oss_fuzz_checkout = not args.no_oss_fuzz_checkout

    # check if the project exists as subfolder in oss-fuzz
    of_project_dir_rel = Path("projects") / project_name
    of_project_dir = oss_fuzz_dir / of_project_dir_rel
    if not of_project_dir.exists():
        msg = f"Project dir {of_project_dir} for project {project_name} does not exist. Probably invalid project name"
        log.error(msg)
        write_transformation_status_to_file(args.failed_list_path, project_name, details=msg)
        return

    # checkout older oss-fuzz commit if commit-specific build was requested; determine output directory
    if args.commit_hash:
        oss_fuzz_commit = commit_specific_of_preparation(
            args, project_name, project_repo, of_project_dir, oss_fuzz_dir, oss_fuzz_repo, oss_fuzz_checkout
        )
        cp_dir = output_dir / f"cp-{project_name}-{args.commit_hash}"
    else:
        oss_fuzz_commit = "HEAD"
        cp_dir = output_dir / f"cp-{project_name}-HEAD"
    if args.single_cp:
        cp_dir = output_dir
    log.info(f"Creating CP for {project_name} at {cp_dir}")

    # copy all project-specific oss-fuzz files into the CP's src dir
    try:
        if oss_fuzz_checkout:
            copy_details = f"Copying of oss-fuzz data from {of_project_dir}"
            shutil.copytree(of_project_dir, cp_dir, dirs_exist_ok=True)
        else:
            copy_details = (
                f"Extraction of oss-fuzz data from path {of_project_dir_rel} "
                f"in commit {oss_fuzz_commit} of repo {oss_fuzz_repo}"
            )
            buf = BytesIO()
            oss_fuzz_repo.archive(buf, oss_fuzz_commit, format="tar", path=of_project_dir_rel)
            buf.seek(0)
            cp_dir.mkdir(exist_ok=True)
            with tarfile.open(fileobj=buf) as tar:
                tar.extractall(path=cp_dir, members=trim_tar_prefix(tar, f"{of_project_dir_rel}/"))
        log.info(f"{copy_details} to {cp_dir}/src succeeded.")
    except Exception as e:  # pylint: disable=broad-exception-caught
        write_transformation_status_to_file(
            args.failed_list_path,
            project_name,
            details=f"{copy_details} to {cp_dir}/src failed: {e}",
            log_error=True,
        )
        return

    # create cp-dir, src, out, work
    for d in (cp_dir / "src", cp_dir / "out", cp_dir / "work"):
        d.mkdir(exist_ok=True)
    log.info(f"Created src, out, work inside {cp_dir}")

    # create .env file because run.sh expects it to exist
    env_file = cp_dir / ".env.docker"
    env_file.touch(exist_ok=True)
    log.info(f"Created .env file for project {project_name}")

    # remove --depth 1 because CRS needs the commit history
    dockerfile = cp_dir / "Dockerfile"
    try:
        remove_depth_1(dockerfile, project_name)
    except FileNotFoundError as e:
        log.error(f"Error removing depth 1 from Dockerfile {dockerfile}: {e}")
        write_transformation_status_to_file(args.failed_list_path, project_name, details=str(e))
        return

    log.info(f"Removed depth 1 from Dockerfile: {dockerfile}")

    # copy and customize run.sh
    copy_and_customize_run_sh(project_name, args.unique_tag, script_dir, cp_dir)

    # prune git history for commit-specific CPs
    if args.commit_hash:
        try:
            prune_git_history(dockerfile, args.commit_hash)
        except InformationNotFoundError as e:
            write_transformation_status_to_file(args.failed_list_path, project_name, details=str(e))
            return

    # prepare docker conatiner
    shutil.copy2(script_dir / "init_CRS.sh", cp_dir)
    shell_script_log = cp_dir / "run_sh_log.log"  # write build output to file for easier debugging
    with shell_script_log.open("w", encoding="utf-8") as f:
        log.info(f"Starting docker preparation for project {project_name}")
        subprocess.run(
            ["./run.sh", "-v", "make-cpsrc-prepare"], cwd=cp_dir, stdout=f, stderr=subprocess.STDOUT, check=False
        )
        log.info(f"Starting run.sh build for project {project_name}")
        subprocess.run(["./run.sh", "-v", "build"], cwd=cp_dir, stdout=f, stderr=subprocess.STDOUT, check=False)

    # modify project.yaml
    project_yaml_data = modify_project_yaml(cp_dir, args, project_name, dockerfile)

    # check for compile_commands.json if c/c++ project
    if project_yaml_data["language"] == "c" or project_yaml_data["language"] == "c++":
        if not check_compile_commands(cp_dir):
            log.error(f"Project {project_name} does not have an (non-empty) copmilation database after building.")
            write_transformation_status_to_file(
                args.failed_list_path,
                project_name,
                details="Project does not have an (non-empty) compilation database after building",
            )
            return

    # build oss-fuzz version for coverage if requested
    if args.double_build_mode:
        assert oss_fuzz_checkout
        oss_fuzz_coverage_build(project_name, oss_fuzz_dir, args.failed_list_path)

    log.info(f"Build process complete. CP is now available at: {cp_dir}")
    write_transformation_status_to_file(args.success_list_path, project_name)


def copy_and_customize_run_sh(project_name: str, docker_image_tag: str | None, script_dir: Path, cp_dir: Path) -> None:
    """
    Copy run.sh into CP dir and customize it for the project at hand
    """
    with (script_dir / "run_replace.sh").open(encoding="utf-8") as f:
        content = f.read()
    content = content.replace("NAME_TO_BE_REPLACED", project_name)
    docker_image_name = f"gcr.io/oss-fuzz/{project_name}-with-bear"
    if docker_image_tag:
        docker_image_name += f":{docker_image_tag}"
    content = content.replace("DOCKER_IMAGE_TO_BE_REPLACED", docker_image_name)
    cp_run_sh = cp_dir / "run.sh"
    with cp_run_sh.open("w", encoding="utf-8") as f:
        f.write(content)
    cp_run_sh.chmod(cp_run_sh.stat().st_mode | stat.S_IXUSR)
    log.info(f"Wrote customized run.sh to {cp_run_sh}")


def oss_fuzz_coverage_build(project_name: str, oss_fuzz_dir: Path, failed_file: Path) -> None:
    """
    Use the OSS-Fuzz infra/helper.py script to also build the project.
    This OSS-Fuzz build can then be used for coverage evaluation
    """
    log.info(f"Starting build_fuzzers for {project_name}")
    try:
        subprocess.run(
            ["python3", oss_fuzz_dir / "infra" / "helper.py", "build_fuzzers", "--sanitizer", "coverage", project_name],
            check=True,
        )
    except CalledProcessError as e:
        log.error(f"Error building the oss-fuzz version of {project_name}: {e}")
        write_transformation_status_to_file(
            failed_file,
            project_name,
            details=f"Error building the oss-fuzz version of {project_name}: {e}",
        )
        raise e

    log.info(f"Checking build success after build_fuzzers for project {project_name}")
    try:
        list_harness_binaries(oss_fuzz_dir / "build" / "out" / project_name)
    except FileNotFoundError as e:
        write_transformation_status_to_file(failed_file, project_name, details=str(e))
        raise e


def commit_specific_of_preparation(
    args: argparse.Namespace,
    project_name: str,
    project_repo: Repo | None,
    of_project_dir: Path,
    oss_fuzz_dir: Path,
    oss_fuzz_repo: Repo,
    checkout: bool,
) -> str:
    """
    When building for specific commits we also need to checkout the local OSS-Fuzz clone at a matching commit
    or the build.sh might be incompatible with the project.

    The matching commit is determined by date.
    If checkout is true, it is immediately checked out.
    Returns the hash of the OSS-Fuzz commit.
    """

    def resolve_commit_date(repo: Repo) -> tuple[str, str]:
        "Helper: Determine the requested commit's full hash and date."
        try:
            commit = repo.commit(args.commit_hash)
            commit_date = commit.committed_datetime.isoformat()
            log.info(f"Commit date for {commit.hexsha}@{project_name}: {commit_date}")
            return (commit.hexsha, commit_date)
        except GitError as e:
            write_transformation_status_to_file(
                args.failed_list_path,
                project_name,
                details=f"Error: Failed to get commit date for {commit}@{project_name}: {e}",
                log_error=True,
            )
            raise

    log.info(f"Commit-specifc build for {args.commit_hash}@{project_name} was requested.")

    # read repo url from project.yaml because we need the git history to find the date of the requested commit to find the matching oss-fuzz commit
    of_project_yaml_path = of_project_dir / "project.yaml"
    project_repo_url = get_main_repo(of_project_yaml_path)

    if project_repo:
        try:
            repo_origin = project_repo.remotes["origin"]
        except IndexError:
            write_transformation_status_to_file(
                args.failed_list_path,
                project_name,
                details=f'Error: Provided repository {project_repo} does not have the "origin" remote',
                log_error=True,
            )
            raise
        if repo_origin.url != project_repo_url:
            e = RuntimeError(
                f"Error: Provided repository {project_repo} has origin URL {repo_origin.url}, but expected {project_repo_url}"
            )
            write_transformation_status_to_file(
                args.failed_list_path,
                project_name,
                details=str(e),
                log_error=True,
            )
            raise e

        commit_hash, commit_date = resolve_commit_date(project_repo)

    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            log.info(f"Cloning {project_name} ({project_repo_url}) to acquire date of specified commit")

            try:
                repo = Repo.clone_from(project_repo_url, temp_path)
            except GitError as e:
                write_transformation_status_to_file(
                    args.failed_list_path,
                    project_name,
                    details=f"Error: Failed to clone repository {project_repo_url} into {temp_path}: {e}",
                    log_error=True,
                )
                raise

            commit_hash, commit_date = resolve_commit_date(repo)

    log.info(f"Locating oss-fuzz version hopefully compatible with {commit_hash}@{project_name}")

    matching_of_commit = oss_fuzz_repo.git.rev_list("master", before=commit_date, first_parent=True, n=1)
    if not matching_of_commit:
        msg = f"Error: Failed to find matching oss-fuzz commit ({oss_fuzz_dir}) to match {commit_hash}@{project_name}"
        log.error(msg)
        write_transformation_status_to_file(args.failed_list_path, project_name, msg)
        raise InformationNotFoundError(msg)

    if checkout:
        oss_fuzz_repo.git.checkout(matching_of_commit)
        log.info(
            f"Checked out oss-fuzz ({oss_fuzz_dir}) at commit {matching_of_commit} to match {commit_hash}@{project_name}"
        )
    else:
        log.info(
            f"Found oss-fuzz commit {matching_of_commit} (from {oss_fuzz_dir}) to match {commit_hash}@{project_name}"
        )

    return matching_of_commit


def modify_project_yaml(cp_dir: Path, args: argparse.Namespace, project_name: str, dockerfile: Path) -> dict[str, Any]:
    """
    Modify project.yaml fiel to include the following information:
    - harnesses with paths to source file and binary
    - git path
    - relevant sanitizers
    """
    cp_project_yaml_path = cp_dir / "project.yaml"
    project_yaml_data = read_yaml_file(cp_project_yaml_path)
    # get harness data (binary and source file combinations)
    # only ask api if harness field is not already present (looking at you: local test project)
    if (
        "harnesses" in project_yaml_data
        and isinstance(project_yaml_data["harnesses"], dict)
        and project_yaml_data["harnesses"]
    ):
        log.info(f"Project.yaml {cp_project_yaml_path} already contains harness information. Skipping API request.")
    else:  # normal oss-fuzz project -> ask API
        project_yaml_data["harnesses"] = fetch_and_check_harnesses(
            cp_dir, args, project_name, project_yaml_data["language"]
        )

    project_yaml_data["cp_name"] = project_name  # set project name to make crs happy

    # get source
    if "main_repo" in project_yaml_data:
        url = project_yaml_data["main_repo"]
    elif "homepage" in project_yaml_data:
        url = project_yaml_data["homepage"]
    else:
        url = None

    # check for .git folder after building
    cp_sources = project_yaml_data.get("cp_sources", {})
    if cp_sources:
        potential_path = cp_dir / "src" / next(iter(cp_sources)) / ".git"
        if potential_path.exists():
            log.info(f"Git path was already clearified in project.yaml: {potential_path}")
            git_path = potential_path
        else:
            git_path = get_git_path(cp_dir, args.failed_list_path, project_name, dockerfile, url)
    else:
        git_path = get_git_path(cp_dir, args.failed_list_path, project_name, dockerfile, url)
    log.info(f".git path of CP {cp_dir} is {git_path}")

    # save git path so we can re-use it in 0day-search
    parts = Path(git_path).parts
    if "src" in parts:
        src_index = parts.index("src")
        project_yaml_data["src_path"] = str(Path(*parts[src_index:]))
    else:
        msg = f"The .git folder of project {project_name} is not a subfolder of /src"
        log.error(msg)
        write_transformation_status_to_file(
            args.failed_list_path,
            project_name,
            details=msg,
        )
        raise ValueError(msg)

    # get default branch - may be needed to reset state after checking out different commits
    try:
        default_branch = get_default_branch(git_path)
    except subprocess.CalledProcessError as e:
        msg = f"Problem determining the default branch of project {project_name}: {e}"
        log.error(msg)
        write_transformation_status_to_file(
            args.failed_list_path,
            project_name,
            details=msg,
        )
        raise e
    subfolder_name = str(Path(*parts[src_index + 1 :]))
    if default_branch:
        project_yaml_data["cp_sources"] = {str(subfolder_name): {"address": url, "ref": default_branch}}
    else:
        project_yaml_data["cp_sources"] = {str(subfolder_name): {"address": url}}

    # write the changes to file
    write_yaml_file(project_yaml_data, cp_project_yaml_path)
    log.info(f"Writing the following data into {cp_project_yaml_path}: {project_yaml_data}")

    return project_yaml_data


def get_git_path(cp_dir: Path, failed_list: Path, project_name: str, dockerfile: Path, url: str) -> Path:
    """
    Get the path of .git folder inside the CP
    Try with src/project_name first then the repo name or clone destination derived from the docker file
    """
    git_path = cp_dir / "src" / project_name
    if not git_path.exists() or not git_path.is_dir():
        if url is None:
            msg = f"Project {project_name} has no folder at {git_path} (tried: project name) and no url to check Dockerfile for git clone destination"
            log.error(msg)
            raise FileNotFoundError(msg)
        log.warning(f"Project {project_name} has no folder at {git_path} (tried: project name)")
        # try parsing Dockerfile for clone destination dir (fallback: repo name)
        git_sub_path = find_git_path_via_dockerfile(dockerfile, url)
        if git_sub_path is not None:
            git_path = cp_dir / "src" / git_sub_path
            if not git_path.exists() or not git_path.is_dir():
                msg = f"Project {project_name} has no folder at {git_path} (tried: clone destination|repo name, project name)"
                log.error(msg)
                write_transformation_status_to_file(
                    failed_list,
                    project_name,
                    details=msg,
                )
                raise FileNotFoundError(msg)
        else:
            msg = f"Project {project_name} has no folder at {git_path} and no other path can be derived from url {url} or Dockerfile {dockerfile}"
            log.error(msg)
            write_transformation_status_to_file(
                failed_list,
                project_name,
                details=msg,
            )
            raise FileNotFoundError(msg)
    return git_path


def fetch_and_check_harnesses(
    cp_dir: Path, args: argparse.Namespace, project_name: str, language: str
) -> dict[str, dict[str, str]]:
    """
    Fetch harness data from API and verify that they actually exist in our local project clone
    """
    already_logged = False
    try:
        api_harnesses = fetch_harness_data(project_name, language)  # get harness info via API
        if not api_harnesses:
            msg = f"API did not return any harnesses for project {project_name}"
            log.error(msg)
            write_transformation_status_to_file(
                args.failed_list_path,
                project_name,
                details=msg,
            )
            already_logged = True
            raise InformationNotFoundError(msg)
    except InformationNotFoundError as e:
        if not already_logged:
            write_transformation_status_to_file(args.failed_list_path, project_name, details=str(e))
        raise e
    try:
        # remove any harness from list that does not have matching binary in /out or source file does not exist in expected location
        checked_harnesses = verify_api_to_actual_harness_file_mapping(cp_dir, cp_dir / "out", api_harnesses)
    except FileNotFoundError as e:
        write_transformation_status_to_file(args.failed_list_path, project_name, details=str(e))
        raise e
    return checked_harnesses


def main() -> None:
    """
    Transform OSS-Fuzz project into Challenge Project"""
    parser = argparse.ArgumentParser(description="Build utility script for OSS-Fuzz.")
    parser.add_argument("project_name", help="Name of the OSS-Fuzz project or 'all'", metavar="PROJECT_NAME")
    parser.add_argument("-c", "--commit-hash", type=str, help="Build for a specific commit hash", metavar="HASH")
    parser.add_argument(
        "--project-repo",
        type=Path,
        help="Already-cloned Git repository with the project's commit history (with -c, it must already contain the commit)",
        metavar="DIR",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, help="Specify output directory", default=Path.cwd(), metavar="DIR"
    )
    parser.add_argument(
        "--single-cp", action="store_true", help="Use the entire --output-dir for a single CP (incompatible with 'all')"
    )
    parser.add_argument("--oss-fuzz", type=Path, help="Use a specific OSS-Fuzz clone", metavar="DIR")
    parser.add_argument(
        "-p",
        "--parallel-processes",
        type=int,
        help="Number of parallel processes, defaults to 6",
        default=6,
        metavar="#PROCESSES",
    )
    parser.add_argument(
        "-d", "--double-build-mode", action="store_true", help="Enable double build mode for easier coverage evaluation"
    )
    parser.add_argument(
        "-l", "--language", type=str, help="Build all projects of a specific programming language", metavar="LANGUAGE"
    )
    parser.add_argument(
        "-s", "--success-list-path", type=Path, help="Path to successful projects", default=None, metavar="FILE"
    )
    parser.add_argument(
        "-f", "--failed-list-path", type=Path, help="Path to failed projects", default=None, metavar="FILE"
    )
    parser.add_argument(
        "--force", action="store_true", help="Build the project unconditionally, ignoring the success/failed lists"
    )
    parser.add_argument(
        "--unique-tag",
        type=str,
        help="A unique string identifying global resources related to this CP (e.g. the Docker image). Scoped within a project name",
        metavar="STR",
    )
    parser.add_argument(
        "--no-oss-fuzz-checkout",
        action="store_true",
        help="Do not modify the OSS-Fuzz repository. Allows parallel commit-specific builds. Incompatible with --double-build-mode",
    )

    args = parser.parse_args()

    # Determine script directory
    script_dir = Path(__file__).resolve().parent

    # Output directory setup
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True, exist_ok=True)
    elif not args.output_dir.is_dir():
        raise NotADirectoryError(f"{args.output_dir} exists but is not a directory.")
    output_dir = args.output_dir.resolve()

    # set outputpath of success/failed list if necessary
    if args.success_list_path is None:
        args.success_list_path = output_dir / "success.txt"
    if args.failed_list_path is None:
        args.failed_list_path = output_dir / "failed.txt"

    # set up settings and logging
    load_settings()
    LOGGING_PROVIDER.init_logging(output_dir)
    dump_config(log)

    log.info(
        f"Output Directory is: {output_dir} with success list {args.success_list_path} and failed list {args.failed_list_path}"
    )

    # check for incompatible options
    if args.commit_hash:
        args.parallel_processes = 1
        if args.project_name == "all":
            log.error("Commit-specific builds are only possible when script is invoked for one specific project")
            sys.exit(1)
        if args.double_build_mode:
            log.warning("Commit-specific coverage builds are currently not supported. Trying anyways.")
    if args.no_oss_fuzz_checkout and args.double_build_mode:
        log.error("Double build mode is incompatible with --no-oss-fuzz-checkout")
        sys.exit(1)
    if args.project_name == "all" and args.single_cp:
        log.error("--single-cp cannot work with building all projects")
        sys.exit(1)
    if args.project_name == "all" and args.project_repo is not None:
        log.error("--project-repo only works with a specific project")
        sys.exit(1)

    # OSS-Fuzz directory setup
    oss_fuzz_dir = get_oss_fuzz_dir(args, output_dir)
    oss_fuzz_repo = Repo(oss_fuzz_dir)

    log.info(
        f"Double build (Challenge Project + OSS-Fuzz version for easier coverage evaluation) mode is {'active' if args.double_build_mode else 'inactive'}"
    )

    # aquire list of all projects to be transformed
    remaining_projects = get_projects_to_be_transformed(args, oss_fuzz_dir)

    assert len(remaining_projects) == 1 or args.project_repo is None
    if args.project_repo is not None:
        project_repo = Repo(args.project_repo)
    else:
        project_repo = None

    # Create a multiprocessing Pool
    with Pool(processes=args.parallel_processes) as pool:
        # Process each project
        results = [
            (
                project,
                pool.apply_async(
                    transform_project,
                    (project, project_repo, output_dir, oss_fuzz_dir, oss_fuzz_repo, script_dir, args),
                ),
            )
            for project in remaining_projects
        ]

        # Handle results later so we do not hinder multi-processing
        for project_name, result in results:
            try:
                result.get()
            except Exception:  # pylint: disable=broad-exception-caught
                log.error(f"Unexpected error in getting result for project '{project_name}':", exc_info=True)

        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
