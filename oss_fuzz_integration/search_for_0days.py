# pylint: disable=line-too-long,no-else-return,too-many-locals,too-many-statements,too-many-lines

"""
Run CRS on the newest n commits of a list of OSS-Fuzz projects to hopefully detect (previously unknown) vulnerabilities
"""

import argparse
import json
import os
import signal
import subprocess
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from pathlib import PurePath
from types import FrameType
from typing import Any

import yaml
from git import GitError
from git import Repo
from pydantic import BaseModel

from crs.base.constants import ALLOWED_PATCH_EXTENSIONS
from crs.base.settings import INTROSPECTOR_ARTIFACTS_DIR
from crscommon.logging.dump_config import dump_config
from crscommon.logging.logging_provider import LOGGING_PROVIDER
from crscommon.settings import load_settings
from eval.evaluate_crs import get_crs_info
from oss_fuzz_integration.harness import analyze_commit as harnesses_from_changed_lines
from oss_fuzz_integration.harness import get_all_functions

OSS_FUZZ_DATA_DIR = Path(__file__).parent.resolve()

log = LOGGING_PROVIDER.new_logger("0DAY", hook_exception=True)
interrupted = False  # pylint: disable=invalid-name


def get_qualifying_reachable_functions(
    project_name: str, cp_storage: Path, path_to_all_functions_json: Path, args: argparse.Namespace
) -> tuple[list[str], int]:
    """
    Returns list of functions that are
    1. reachable by fuzzers
    2. have less than args.max_harnesses harnesses

    Also returns the number of harness-function pairs of functions that are below the max_harnesses
    threshold.
    """
    yaml_file = (cp_storage / f"cp-{project_name}-HEAD" / "project.yaml").resolve()
    if not yaml_file.exists():
        log.error(f"[Function Mode] project.yaml does not exist for project {project_name}: {yaml_file}")
        return [], 0

    # open the json
    if not path_to_all_functions_json.exists():
        log.error(
            f"[Function Mode] all_functions.json does not exist for project {project_name}: {path_to_all_functions_json}"
        )
        return [], 0
    with path_to_all_functions_json.open(encoding="utf-8") as f:
        data = json.load(f)
    functions = data.get("functions", [])
    if not functions:
        log.warning(f"[Function Mode] No functions found for project {project_name} in all_functions.json")
        return [], 0
    reachable_functions = [
        f for f in functions if f.get("reached_by_fuzzers") if len(f.get("reached_by_fuzzers", [])) < args.max_harnesses
    ]
    harness_function_pairs = sum(
        len(f.get("reached_by_fuzzers", []))
        for f in reachable_functions
        if len(f.get("reached_by_fuzzers", [])) < args.max_harnesses
    )  # count all harness-function pairs that are below the max_harnesses threshold

    return reachable_functions, harness_function_pairs


def validate_harnesses(harnesses: list[str], project_name: str, cp_storage: Path) -> list[str]:
    """
    Determines which harnesses from the list are actually present in the current CP out directory.
    """
    # project.yaml only contains valid harnesses
    yaml_file = (cp_storage / f"cp-{project_name}-HEAD" / "project.yaml").resolve()
    if not yaml_file.exists():
        log.error(f"[Function Mode] project.yaml does not exist for project {project_name}: {yaml_file}")
        return []

    data = yaml.safe_load(yaml_file.open(encoding="utf-8"))

    # Extract harnesses and collect their 'name' fields
    harnesses_from_yaml = data.get("harnesses", {})

    matching_harnesses = []

    for harness in harnesses:
        for key, value in harnesses_from_yaml.items():
            if harness in (key, value):
                matching_harnesses.append(key)

    return matching_harnesses


# pylint: disable=too-many-branches
def process_project_function_mode(args: argparse.Namespace, out_dir: Path, cp_storage: Path, project_name: str) -> None:
    """
    Processes each project and its functions in function-mode.
    """
    yaml_file = (cp_storage / f"cp-{project_name}-HEAD" / "project.yaml").resolve()

    if not yaml_file.exists():
        log.error(f"[Function Mode] project.yaml does not exist for project {project_name}: {yaml_file}")
        return

    # get the "language" field from the yaml file
    with yaml_file.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    language = data.get("language", "").lower()

    if language not in args.supported_languages.split(","):
        log.warning(f"[Function Mode] Unsupported language '{language}' for project {project_name}. Skipping.")
        return

    # Directory that should contain "all_functions.json" for this project
    get_all_functions(project=project_name, outputpath=INTROSPECTOR_ARTIFACTS_DIR.get())
    functions_json_path = INTROSPECTOR_ARTIFACTS_DIR.get() / project_name / "all_functions.json"
    if not functions_json_path.exists():
        log.warning(
            f"[Function Mode] all_functions.json does not exist for project {project_name}: {functions_json_path}"
        )
        return
    else:
        log.info(f"[Function Mode] Found all_functions.json for project {project_name} at {functions_json_path}")

    # check if the number of reachable functions is less than the threshold
    reachable_funcs, harness_func_pairs = get_qualifying_reachable_functions(
        project_name=project_name, cp_storage=cp_storage, path_to_all_functions_json=functions_json_path, args=args
    )
    if len(reachable_funcs) > args.max_reachable_functions:
        log.info(f"[Function Mode] Skipping project {project_name} due to too many reachable functions.")
        return
    log.info(f"[Function Mode] Loading functions from {functions_json_path}")
    log.info(
        f"[Function Mode] Project {project_name} has {len(reachable_funcs)} reachable functions and "
        f"{harness_func_pairs} harness-function pairs to process that meet our criteria."
    )
    with open(functions_json_path, encoding="utf-8") as f:
        data = json.load(f)
    functions = data.get("functions", [])
    if not functions or len(reachable_funcs) == 0:
        log.error(f"[Function Mode] No functions found for project {project_name}")
        return

    if args.dry_run:
        log.info(f"[Function Mode] Dry run enabled, skipping actual processing for {project_name}")

    amount_of_reachable_functions_proccessed = 0
    total_harness_function_pairs_processed = 0
    for func in functions:
        func_name = func.get("function_name")
        source_file: str = func.get("function_filename")

        if language == "python":
            # For Python: preserve leading dots for relative imports, replace other dots with slashes
            if source_file.startswith(".."):
                # Keep the leading dots intact
                source_file = f"src/{project_name}/.." + (source_file.replace(".", "/") + ".py")
            else:
                source_file = f"src/{project_name}/" + (source_file.replace(".", "/") + ".py")

        # construct full path to the source file
        if not Path(cp_storage / f"cp-{project_name}-HEAD" / source_file.lstrip("/")).exists():
            log.error(
                f"[Function Mode] Source file {source_file} for function {func_name} in project {project_name} does not exist, skipping function."
            )
            return

        if not func_name:
            log.error(f"[Function Mode] Function with no 'name' field in project {project_name}: {func}")
            continue

        harnesses = func.get("reached_by_fuzzers", [])
        if not harnesses:
            log.info(f"[Function Mode] No harnesses found for function {func_name} in project {project_name}")
            continue

        matching_harnesses = validate_harnesses(harnesses, project_name, cp_storage)

        log.info(
            f"[Function Mode] Found {len(matching_harnesses)} harnesses for function {func_name} in project {project_name}"
        )

        if not matching_harnesses:
            log.info(
                f"[Function Mode] No valid harnesses found for function {func_name} in project {project_name}, skipping function."
            )
            continue

        total_harness_function_pairs_processed += len(matching_harnesses)
        amount_of_reachable_functions_proccessed += 1

        if args.dry_run:
            continue

        func_dir = out_dir / project_name / func_name
        os.makedirs(func_dir, exist_ok=True)

        for i in range(args.function_sample_size_analysis):
            run_dir = func_dir / f"sample-analysis-{i+1}"
            os.makedirs(run_dir, exist_ok=False)
            log.info(f"[Function Mode] Analyzing {project_name}'s {func_name} run {i+1}")
            analyze_file = run_dir / f"analyze-{datetime.now().isoformat()}.json"

            try:
                subprocess.run(
                    [
                        "run_crs",
                        "--output-dir",
                        run_dir,
                        "--function-to-be-analyzed",
                        func_name,
                        "--ai-model",
                        args.model,
                        "analyze",
                        str(cp_storage / f"cp-{project_name}-HEAD"),
                        "--state-file",
                        analyze_file,
                    ],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                log.error(f"[Function Mode] Analysis failed for {project_name}/{func_name} run {i+1}: {e}")
                continue

            # read the analyze file (json) and get the vulnerability vuln_functions and type
            with open(analyze_file, encoding="utf-8") as f:
                analyze_data = json.load(f)

            vulnerability = analyze_data.get("vulnerability", None)
            if vulnerability is None:
                log.error(
                    f"[Function Mode] No vulnerability report found in analyze file for {project_name}/{func_name} run {i+1}"
                )
                continue

            if not vulnerability["vuln_functions"] or vulnerability["type"] is None:
                log.info(
                    f"[Function Mode] No vulnerable functions found in analyze file for {project_name}/{func_name} run {i+1}"
                )
                continue

            for harness_id in matching_harnesses:
                # Create a directory for the harness
                harness_dir = run_dir / harness_id
                os.makedirs(harness_dir, exist_ok=True)

                for j in range(args.function_sample_size_pov):
                    log.info(
                        f"[Function Mode] Running PoV Builder for {project_name}/{func_name} with harness {harness_id}, analysis sample {i+1}, PoV sample {j+1}"
                    )

                    pov_dir = harness_dir / f"sample-pov-{j+1}"

                    os.makedirs(pov_dir, exist_ok=False)

                    pov_file = pov_dir / f"pov-{datetime.now().isoformat()}.json"

                    # Run the CRS Pov Builder
                    run_pov_cmd = [
                        "run_crs",
                        "--output-dir",
                        pov_dir,
                        "--function-to-be-analyzed",
                        func_name,
                        "--ai-model",
                        args.model,
                        "--pov-skip-build",
                        "True",
                        "pov",
                        str(cp_storage / f"cp-{project_name}-HEAD"),
                        "--harness-id",
                        harness_id,
                        "--load-state",
                        analyze_file,
                        "--state-file",
                        pov_file,
                    ]

                    try:
                        subprocess.run(run_pov_cmd, check=True)
                    except subprocess.CalledProcessError as e:
                        log.error(
                            f"[Function Mode] CRS Pov Builder failed for {project_name}/{func_name} run {i+1}: {e}"
                        )
                        continue

            (run_dir / "done.txt").write_text("done\n", encoding="utf-8")

    log.info(
        f"[Function Mode] Processed {amount_of_reachable_functions_proccessed} reachable functions for project "
        f"{project_name}, total harness-function pairs: {total_harness_function_pairs_processed}"
    )


def run_function_mode(args: argparse.Namespace, out_dir: Path) -> None:
    """
    Run the function mode for all projects in the project file.
    """
    cp_storage: Path = args.cp_dir.resolve()

    # Read the project names from the file
    with args.project_file.open(encoding="utf-8") as f:
        project_names = [line.strip() for line in f.readlines()]

    # One worker per project (up to --parallel-instances)
    parallel_instances = args.parallel_instances
    log.info(f"[Function Mode] Processing {len(project_names)} projects with {parallel_instances} parallel instances")

    with Pool(processes=parallel_instances) as pool:
        results = []
        for project_name in project_names:
            res = pool.apply_async(process_project_function_mode, (args, out_dir, cp_storage, project_name))
            results.append((project_name, res))

        for project_name, res in results:
            try:
                res.get()
            except Exception:  # pylint: disable=broad-exception-caught
                log.error(f"[Function Mode] Unexpected error for project '{project_name}':", exc_info=True)

        pool.close()
        pool.join()


def reset_and_check_for_git_history(git_path: Path, ref: str) -> bool:
    """
    checks whether the project uses git and has more than one commit present in its local clone
    It also resets it to the given reference
    """
    if not git_path.exists():
        log.warning(f"Warning: no .git folder at {git_path}")
        return False

    try:
        repo = Repo(git_path)
        # Checkout the specified ref
        log.info(f"Checking out ref {ref} of project {git_path}")
        repo.git.checkout(ref)

        # Get the number of commits
        log.info(f"Checking for >1 commit in project {git_path}")
        commit_count_str = repo.git.rev_list("HEAD", count=True)
        commit_count = int(commit_count_str.strip())

        return commit_count > 1
    except GitError as e:
        log.error(f"Error executing git checkout or commit counting in path {git_path}: {e}")
        return False


def get_commits(git_path: Path, n: int) -> list[str]:
    """
    Get the latest n commit hashes of a given git project
    """
    # Get a list of the latest n commits
    try:
        log.info(f"Attempting to get the latest {n} commits from {git_path}")
        commits = (
            subprocess.check_output(
                ["git", "-C", git_path, "log", "--format=format:%H", f"HEAD~{n}..HEAD"], stderr=subprocess.STDOUT
            )
            .decode()
            .strip()
            .splitlines()
        )

        if len(commits) < n:
            log.warning(f"Not enough commits in {git_path} ({len(commits)}<{n}). Skipping.")
            return []

        return commits
    except subprocess.CalledProcessError as e:
        log.error(f"Error while getting commits from {git_path}: {e.output.decode()}")
        return []


def parse_project_yaml(yaml_file: Path) -> dict[str, Any]:
    """
    Return content of project.yaml if it has harness information
    """
    # Load the YAML content from the file
    with yaml_file.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)

    # Check if 'harnesses' is present in the YAML structure
    if "harnesses" in data:
        return data
    else:
        log.error(f"No 'harnesses' field found in the YAML {yaml_file.as_posix()}.")
        return {}


def check_harnesses(project_name: str, cp_dir: Path, harnesses: list[str]) -> None:
    """
    Check if the harnesses' source code location in project.yaml is still present
    If the src file is present, we assume that it is still the correct one
    Else: generate a new project.yaml
    """
    generate_new = False
    for harness in harnesses:
        log.info(f"Currently working with harness {harness} of project {project_name}")
        harness_src_path = cp_dir / harness

        # check if source file exists
        if not harness_src_path.exists():
            generate_new = True
            break

        log.info(
            f"Project {project_name}: src path for harness {harness} still exists and is assumed to be correct: {harness_src_path}"
        )

    if generate_new:  # TODO: check if this is a common issue
        log.warning(
            f"At least one harness path of project {project_name} is not available anymore (probably {harness_src_path}). CRS will probably abort."
        )


def build_commit(cp_dir: Path, commit_hash: str, project_name: str, git_path: Path) -> bool:
    """
    Checkout a specific commit and rebuild project
    """
    # add debug symbols to env file because we might need to find fuzz target source files later
    debug_flags = "CFLAGS=-g\nCXXFLAGS=-g\n"
    env_file = cp_dir / ".env.docker"
    log.info(f"Adding debug flags to env file: {env_file} for project {project_name} @ commit {commit_hash}")
    with env_file.open("a") as f:
        f.write(debug_flags)

    log.info(f"removing files from /out directory of project {project_name}")
    try:
        subprocess.check_call(["bash", "-x", "run.sh", "custom", "rm", "-rf", "/out/*"], cwd=cp_dir)
    except subprocess.CalledProcessError as e:
        log.error(
            f"Error while removing /out/* before building commit {commit_hash} for {project_name}: {e}", exc_info=True
        )
        log.warning(f"Skipping commit {commit_hash} of project {project_name}")
        return False

    # Checkout the specific commit
    try:
        subprocess.check_call(["git", "-C", git_path, "checkout", commit_hash])
    except subprocess.CalledProcessError as e:
        log.error(f"Error while checking out commit {commit_hash} for {project_name}: {e}", exc_info=True)
        log.warning(f"Skipping commit {commit_hash} of project {project_name}")
        return False

    log.info(f"Trying to build {project_name} @ {commit_hash}")
    # run.sh always seems to return some non-zero exit code, the actual sanity check is whether fuzz targets spawn in /out
    subprocess.run(["bash", "-x", "run.sh", "build", "-v"], check=False, cwd=cp_dir)

    # remove debug flags again
    # Read all lines and filter out the ones to remove
    log.info(f"Removing debug flags from env file {env_file} in project {project_name}")
    lines = env_file.read_text().splitlines()
    updated_lines = [line for line in lines if line.strip() not in debug_flags]

    # Write back only the lines that should remain
    env_file.write_text("".join(l + "\n" for l in updated_lines))

    # check if there are fuzz targets in out
    out_dir = cp_dir / "out"
    exclude_files = ["llvm-symbolizer"]
    executables = [
        f for f in out_dir.iterdir() if f.is_file() and f.name not in exclude_files and os.access(f, os.X_OK)
    ]
    if len(executables) < 1:
        log.error(f"Error: no fuzz targets detected in /out for project {project_name} @ commit {commit_hash}")
        return False
    else:
        return True


def process_commit(
    project_name: str, commit_hash: str, dir_prefix: Path, harnesses: list[str], result_dir: Path
) -> None:
    """
    Function to process each commit
    build project and check if it was successful (no errors while compiling + >=1 fuzz target in out, excluding llvm-symbolizer)
    check if paths to fuzz targets are still valid
    call CRS analyzer and maybe verifier for every commit-harness-combo
    """
    log.info(f"Processing {project_name} @ commit {commit_hash}")

    cp_dir = dir_prefix / f"cp-{project_name}-HEAD"

    # let analyzer run once per sample because it is not harness-specific
    analyzer_exited_normally = True

    # if there isn't a file that starts with "analyze" in the result directory, we need to run the analyzer
    if any(result_dir.glob("analyze*.json")):
        log.info(f"Skipping CRS Analyzer {project_name} @ commit {commit_hash} as it has already been run")
        analyze_file = next(result_dir.glob("analyze*.json"))
    else:
        analyze_file = result_dir / f"analyze-{datetime.now().isoformat()}.json"
        try:
            log.info(f"Running CRS Analyzer for {project_name} @ commit {commit_hash}")
            subprocess.check_call(
                [
                    "run_crs",
                    "--output-dir",
                    result_dir,
                    "analyze",
                    str(cp_dir),
                    "--commit-hash",
                    commit_hash,
                    "--state-file",
                    analyze_file,
                ]
            )
        except subprocess.CalledProcessError as e:
            analyzer_exited_normally = False
            # this may always happen, not sure, if it happens too often we should investigate it further.
            log.error(f"Error executing CRS Analyzer for {project_name} @ commit {commit_hash}: {e.output.decode()}")

    # call Verifier and hope for the best
    if analyzer_exited_normally:
        # let's check how many harnesses directories are in the result directory
        done_harnesses = len([dir for dir in result_dir.iterdir() if dir.is_dir()])

        harnesses.sort()

        if done_harnesses > 0:
            # remove the last harness result dir because it isn't done
            log.info(
                f"Removing the last harness result dir {harnesses[done_harnesses - 1]} for {project_name} @ commit {commit_hash} because it is not done"
            )
            subprocess.run(["rm", "-rf", harnesses[done_harnesses - 1]], check=True)

        for harness in harnesses[done_harnesses:]:
            # make extra dir for harness so we do not overwrite md files
            harness_dir = result_dir / harness
            os.makedirs(harness_dir, exist_ok=False)
            os.chdir(harness_dir)
            pov_file = harness_dir / f"pov-{datetime.now().isoformat()}.json"
            try:
                log.info(f"Running CRS Verifier for {project_name} @ commit {commit_hash} with harness {harness}")
                subprocess.run(
                    [
                        "run_crs",
                        "--check-pov-at",
                        commit_hash,
                        "--output-dir",
                        harness_dir,
                        "pov",
                        cp_dir,
                        "--commit-hash",
                        commit_hash,
                        "--harness-id",
                        harness,
                        "--load-state",
                        analyze_file,
                        "--state-file",
                        pov_file,
                    ],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                log.error(f"Error executing CRS Verifier: {e.output.decode()}")
            os.chdir(result_dir)


def commit_is_worth_checking_out(commit_hash: str, cp_path_git: Path) -> bool:
    # copy if commit_is_sane
    """
    Applies basic sanity checks to the commit.

    In particular, checks if the commit changed any source files.
    """
    repo = Repo(cp_path_git)
    vuln_commit = repo.commit(commit_hash)
    vuln_parent = vuln_commit.parents[0]
    diff = vuln_commit.diff(vuln_parent)

    for item in diff:
        # TODO: Could renaming a file create a vulnerability? What about
        #       deleting? I don't even know what "changed in the type"
        #       means?..
        if item.change_type not in ("A", "M"):
            continue
        if (name := item.b_path) is None:
            continue
        # TODO: Make this dependent on the project language?
        if PurePath(name).suffix in ALLOWED_PATCH_EXTENSIONS:
            return True

    return False


def get_introspector_harnesses(
    project: str, cp_dir: Path, commit: str, project_result_dir: Path, introspector_path: Path
) -> list[str]:
    """
    Use separate fuzz introspector script to get all relevant harnesses for a specific commit

    Harnesses are relevant when they reach any function that was changed (added/removed lines) in the given commit
    In some cases, changed lines are not attributed to any function and therefore not considered rn -> TODO: fix?
    """
    introspector_data = harnesses_from_changed_lines(project, cp_dir, introspector_path, commit, project_result_dir)
    harnesses = introspector_data["harnesses"]
    log.info(f"Introspector found the following harnesses for {commit} @ {project}: {harnesses}")
    return harnesses


def process_commits(
    project_name: str,
    cp_storage: Path,
    commit_hashes: list[str],
    yaml_file: dict,
    project_result_dir: Path,
    git_path: Path,
    args: argparse.Namespace,
) -> None:
    """Process selected commits."""

    # run CRS on available commits
    for n in range(args.commit_number):
        # check if it is even worth it to build the commit
        cp_dir = cp_storage / f"cp-{project_name}-HEAD"
        if not commit_is_worth_checking_out(commit_hashes[n], git_path):
            log.info(
                f"Commit {commit_hashes[n]} for project {project_name} does not change any source code files. Skipping"
            )
            continue
        # build new commit
        if not build_commit(cp_dir, commit_hashes[n], project_name, git_path):
            log.error(f"Building project {project_name} @ commit {commit_hashes[n]} failed")
            continue

        # iterate over samples for same commit
        commit_dir = project_result_dir / f"{commit_hashes[n]}"
        commit_dir.mkdir(parents=True, exist_ok=True)
        done_sample_dirs = len([i for i in commit_dir.iterdir() if i.is_dir() and (i / "this_sample_is_done").exists()])
        partly_done_sample_dirs = [
            i for i in commit_dir.iterdir() if i.is_dir() and not (i / "this_sample_is_done").exists()
        ]
        for i in range(done_sample_dirs, args.sample_size):
            log.info(
                f"Checking commit {n} of {args.commit_number} in project {project_name} - sample {i + 1} out of {args.sample_size}"
            )
            # create dir for commit/sample
            cur_sample_result_dir = commit_dir / f"{i}_{datetime.now().isoformat()}"
            os.makedirs(cur_sample_result_dir, exist_ok=False)
            log.info(f"args.cp_dir aka dir_prefix for {project_name} is {cp_storage}")

            if not args.dry_run:
                process_commit(
                    project_name=project_name,
                    commit_hash=commit_hashes[n],
                    dir_prefix=cp_storage,
                    harnesses=yaml_file["harnesses"],
                    result_dir=cur_sample_result_dir,
                )
            # write a this_sample_is_done file to the sample directory
            (cur_sample_result_dir / "this_sample_is_done").touch()

        # finish up the partially done samples
        for partly_done_sample_dir in partly_done_sample_dirs:
            cur_sample_result_dir = partly_done_sample_dir
            log.info(
                f"Finishing up the partially done sample {cur_sample_result_dir} for {project_name} @ commit {commit_hashes[n]}"
            )
            if not args.dry_run:
                process_commit(
                    project_name=project_name,
                    commit_hash=commit_hashes[n],
                    dir_prefix=cp_storage,
                    harnesses=yaml_file["harnesses"],
                    result_dir=cur_sample_result_dir,
                )
        # open the done.txt file and write the commit hash to it
        with (project_result_dir / "done.txt").open("a") as f:
            f.write(f"{commit_hashes[n]}\n")


def process_project(args: argparse.Namespace, cp_storage: Path, out_dir: Path, project_name: str) -> None:
    """
    Process a single project
    """
    # check which projects are already done or wip
    checked_projects = [i.name for i in out_dir.iterdir() if i.is_dir()]

    # create result dir for project
    project_result_dir = out_dir / project_name
    os.makedirs(project_result_dir, exist_ok=True)

    # we're working on this project now
    status_file = out_dir / project_name / "STATUS"
    status_file.write_text("BUSY\n", encoding="utf-8")

    already_processed_commits = []

    # check if the project is already done by reading the text file in the logs folder and checking the commit hashes
    if project_name in checked_projects and (out_dir / project_name / "done.txt").exists():
        # read the text file containing the hashes of the commits that have already been completely processed.
        with (out_dir / project_name / "done.txt").open(encoding="utf-8") as f:
            already_processed_commits = [stripped_line for line in f.readlines() if (stripped_line := line.strip())]

            # check if all commits are done
            if len(already_processed_commits) == args.commit_number:
                log.info(f"Project {project_name} is already done. Skipping.")
                status_file.write_text("DONE\n", encoding="utf-8")
                return
    # get current CP path
    cur_project_path = cp_storage / f"cp-{project_name}-HEAD"
    if not cur_project_path.is_dir():
        log.warning(f"No CP {cur_project_path} for {project_name} in {cp_storage}. Skipping")
        return

    # parse yaml file - we need harness info and ref
    yaml_file = parse_project_yaml(cur_project_path / "project.yaml")
    if not yaml_file:
        log.warning(f"project.yaml of project {project_name} does not fulfill requirements. Skipping.")
        status_file.write_text("ERROR\n", encoding="utf-8")
        return
    project_dir = next(iter(yaml_file["cp_sources"]))
    ref = yaml_file["cp_sources"][project_dir]["ref"]

    # check number of harnesses in project.yaml
    num_harnesses = len(yaml_file.get("harnesses", {}))
    log.info(f"Project {project_name} has {num_harnesses} harnesses listed in project.yaml")

    # projects with many harnesses are expensive so we might want to skip them
    if args.max_harnesses is not None and num_harnesses > args.max_harnesses and args.use_introspector is None:
        log.warning(f"Project {project_name} has too many harnesses ({num_harnesses} > {args.max_harnesses}). Skipping")
        status_file.write_text("ERROR\n", encoding="utf-8")
        return

    # check if harness paths still exist
    check_harnesses(project_name, cur_project_path, yaml_file.get("harnesses", {}))

    # nginx is a special little snowflake
    if project_name == "nginx":
        ref = "master"

    # check for git history
    git_path = cur_project_path / "src" / project_dir
    if not reset_and_check_for_git_history(git_path, ref):
        log.warning(f"{project_name} does not fulfill git requirement. Skipping.")
        status_file.write_text("ERROR\n", encoding="utf-8")
        return

    log.info(f"Project {project_name} has a git history.")

    # get latest n commits to CRS
    commit_hashes = get_commits(git_path, args.commit_number)
    if not commit_hashes:
        log.warning(f"Project {project_name} does not have any commits. Skipping")
        status_file.write_text("ERROR\n", encoding="utf-8")
        return

    commit_hashes = [i for i in commit_hashes if i not in already_processed_commits]

    log.info(
        f"Processing {len(commit_hashes)} commits for {project_name}, {len(already_processed_commits)} commits are already done"
    )

    process_commits(project_name, cp_storage, commit_hashes, yaml_file, project_result_dir, git_path, args)

    # change the status of the project to DONE
    status_file.write_text("DONE\n", encoding="utf-8")

    log.info(f"Project {project_name} done")


def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process Git commits for projects.")

    # Add arguments for the project file and N (number of commits to process)
    parser.add_argument(
        "-f",
        "--project_file",
        type=Path,
        default=Path("successful_projects.txt"),
        help="Path to the file containing project names (one per line), defaults to ./successful_projects.txt",
    )

    parser.add_argument(
        "-c",
        "--commit_number",
        type=int,
        default=15,
        help="Number of latest commits to process for each project, defaults to 15",
    )

    parser.add_argument(
        "cp_dir",
        type=Path,
        default=Path("oss-fuzz-all-projects"),
        help="Folder where all successfull OSS-Fuzz CPs are placed",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=Path("zero_day_results"),
        help="Folder for results, defaults to zero_day_results",
    )

    parser.add_argument("model", type=str, help="AI model to be used")

    parser.add_argument(
        "-m",
        "--max_harnesses",
        help="Maximal number of harnesses per project. Projects with more harnesses will be skipped. Defaults to 15",
        type=int,
        default=15,
    )

    parser.add_argument(
        "-i",
        "--parallel_instances",
        help="Number of parallel instances of the CRS to run. Defaults to 1",
        type=int,
        default=1,
    )

    parser.add_argument("-s", "--sample-size", type=int, default=1, help="Number of samples per commit and harness")

    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=-1,
        help="Temperature for the llm - default is -1 / use default of the model",
    )

    parser.add_argument(
        "--use-introspector",
        type=Path,
        help="Path to store/load JSON Fuzz Introspector API data from. Must be set to use Introspector for harness selection. Defaults to None",
        default=None,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Disable building and CRS invocation, e.g. to test how many commits, harnesses etc are accessible",
    )

    parser.add_argument(
        "--function-mode",
        action="store_true",
        help="Run in function-by-function mode analyzing all functions in all_functions.json for each project. This disables all commit/harness/project building logic and is a completely separate logic branch.",
    )
    parser.add_argument(
        "--function-sample-size-analysis",
        type=int,
        default=3,
        help="[Function Mode] Number of runs per function",
    )
    parser.add_argument(
        "--function-sample-size-pov",
        type=int,
        default=5,
        help="[Function Mode] Number of runs per function and harness",
    )

    parser.add_argument(
        "--supported-languages",
        type=str,
        default="c",
        help="Comma-separated list of supported languages for the projects. Defaults to 'c'.",
    )

    parser.add_argument(
        "--max_reachable_functions",
        type=int,
        default=999999,
        help="[Function Mode] Maximum number of reachable functions to process per project.",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default="/data/CRSLOG",
        help="Logfile to store information about the test series, defaults to /data/CRSLOG",
    )

    # Parse arguments
    return parser.parse_args()


def handle_sigint(_signum: int, _frame: FrameType | None) -> None:
    """Handles the SIGINT singal on the main thread"""
    global interrupted  # pylint: disable=global-statement
    interrupted = True
    raise KeyboardInterrupt


def main() -> None:
    """
    Iterate over list of OSS-Fuzz projects and run Analyzer and Verifier on the newest n commits if possible
    """
    global interrupted  # pylint: disable=global-statement
    signal.signal(signal.SIGINT, handle_sigint)

    args = get_args()
    cp_storage: Path = args.cp_dir.resolve()

    # create directory for results
    os.makedirs(args.output_dir, exist_ok=True)
    out_dir: Path = args.output_dir.resolve()

    # set up settings and logging
    load_settings()
    LOGGING_PROVIDER.init_logging(out_dir)
    dump_config(log)

    # export AI model and other settings
    os.environ["CRS_AI_MODEL_NAME"] = args.model
    os.environ["CRS_AI_MODEL_TEMP"] = str(args.temperature)
    os.environ["CRS_DUMP_ALL_CONFIG"] = "yes"

    if args.function_mode:
        log.info("[Function Mode] Running in function-by-function independent mode")
        run_function_mode(args, out_dir)
        log.info("[Function Mode] Done")
        return

    # bad projects that start conversations mid build process
    bad_projects = ["libfido2", "stb"]

    # Read the project names from the file
    with args.project_file.open(encoding="utf-8") as f:
        project_names = [line.strip() for line in f.readlines()]

    # check which projects are already done or wip
    checked_projects = [i.name for i in out_dir.iterdir() if i.is_dir() and i not in bad_projects]
    for p in checked_projects:
        # to make it clear which projects are being worked on currently we'll change the contents of the
        # STATUS file from 'BUSY' to 'INCOMPLETE'
        status_file = out_dir / p / "STATUS"
        if status_file.read_text(encoding="utf-8").strip() == "BUSY":
            status_file.write_text("INCOMPLETE\n", encoding="utf-8")

    parallel_instances = args.parallel_instances

    log_info = ZerodayInfo(
        sample_size=args.sample_size,
        model_name=args.model,
        model_temp=args.temperature,
        result_dir=args.output_dir,
        cp_pool=args.cp_dir,
        project_list=args.project_file,
        logfile=args.log_file.resolve(),
        description="0day-search",
    )

    # log eval run
    log_eval_run(log_info, "started")

    log.info(f"Processing {len(project_names)} projects with {parallel_instances} parallel instances")

    # Create a multiprocessing Pool
    try:
        with Pool(processes=parallel_instances) as pool:
            # Process each project
            results = []
            # for project_name in project_names:
            project_args = [(args, cp_storage, out_dir, project_name) for project_name in project_names]

            # we need to wrap the call so we can parse potential exceptions later
            for project_name, arg_tuple in zip(project_names, project_args):
                result = pool.apply_async(process_project, arg_tuple)
                results.append((project_name, result))

            # Handle results later so we do not hinder multi-processing
            for project_name, result in results:
                try:
                    result.get()
                except Exception:  # pylint: disable=broad-exception-caught
                    log.error(f"Unexpected error in getting result for project '{project_name}':", exc_info=True)

            pool.close()
            pool.join()
    except KeyboardInterrupt:
        interrupted = True
        log_eval_run(log_info, "aborted")

    except Exception as e:  # pylint: disable=broad-exception-caught
        log.error(e)
        interrupted = True
        log_eval_run(log_info, "failed")

    finally:
        if not args.dry_run:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGINT, handle_sigint)

    if not interrupted:
        log_eval_run(log_info, "completed")

    log.info("Done")


class ZerodayInfo(BaseModel):
    """
    Class for storing infos for global logging
    """

    sample_size: int
    model_name: str
    model_temp: float
    result_dir: Path
    cp_pool: Path
    project_list: Path
    logfile: Path
    description: str


def log_eval_run(
    log_info: ZerodayInfo,
    status: str,
) -> None:
    """Logs the evaluation to the evaluation log file"""
    git_info = get_crs_info()
    commit, branch = git_info
    log_entry = (
        f"timestamp={datetime.now().isoformat()} crs_branch={branch} crs_commit={commit} "
        f"type=0days llm_model={log_info.model_name}"
        f"sample_size={log_info.sample_size} temperature={log_info.model_temp} "
        f"eval_results_path={log_info.result_dir} user={os.getlogin()} "
        f"cp_pool={log_info.cp_pool} project_list={log_info.project_list}"
        f'run_description="{log_info.description}" status={status}\n'
    )
    print(log_entry)
    if log_info.logfile:
        with log_info.logfile.open("a", encoding="UTF-8") as f:
            f.write(log_entry)


if __name__ == "__main__":
    main()
