"""
Build with coverage
"""

import os
import subprocess
from pathlib import Path

from crs.base.context import CrsContext
from crs.base.settings import OSS_FUZZ_LOCATION
from crs.logger import CRS_LOGGER

log = CRS_LOGGER.getChild(__name__)

# languages supported by coverage
SUPPORTED_LANGUAGES = ["c"]


def produce_coverage_artifacts(ctx: CrsContext, reproducer_path: Path) -> Path | None:
    """
    Run the helper script to build the project with coverage and produce coverage information
    This function does *NOT* return coverage info. It only produces the artifacts
    """
    # check if the project has already been built with coverage
    # project_name = ctx.proj_yaml_model.cp_name
    cov_build_location: Path = OSS_FUZZ_LOCATION.get() / "build" / "out" / ctx.proj_yaml_model.cp_name

    # if it has been built with coverage, a folder oss-fuzz/build/work/" + project_name + "/" exists
    if cov_build_location.exists():
        log.debug("Project has already been built with coverage")
    else:  # build the project with coverage
        # _build_project_with_coverage(ctx, cov_dir_name)
        log.warning(f"No coverage build for '{cov_build_location}', returning None")
        return None

    # we can now use the helper script to execute the input on the project
    _exec_coverage_container(ctx, reproducer_path, cov_build_location)

    project_dir = (OSS_FUZZ_LOCATION.get() / "build" / "out" / ctx.proj_yaml_model.cp_name).resolve().as_posix()
    report_file: Path = Path(project_dir) / "textcov_reports" / (ctx.harness_id + ".covreport")

    if report_file.exists():
        return report_file
    log.warning(f"Coverage file not found. Expected at path: '{report_file.absolute()}'")
    return None


def _exec_coverage_container(ctx: CrsContext, reproducer_path: Path, cov_build_location: Path) -> None:
    """
    Gather coverage information for the run (cleans the blob directory beforehand)
    """
    oss_dir = OSS_FUZZ_LOCATION.get()
    blob_dir = (cov_build_location / "blobs-for-coverage/").resolve()
    curr_dir = os.getcwd()
    try:
        # clear the blobs directory
        log.debug("Clearing blobs directory")
        command = ["rm", "-rf", blob_dir.as_posix()]
        subprocess.run(command, check=True, text=True, capture_output=True)
        # create blobs directory if it does not exist
        log.debug("Creating blobs directory")
        command = ["mkdir", "-p", blob_dir.as_posix()]
        subprocess.run(command, check=True, text=True, capture_output=True)
        # copy the pov file to the blobs directory
        log.debug("Copying pov file to blobs directory")
        command = ["cp", reproducer_path.as_posix(), blob_dir.as_posix()]
        subprocess.run(command, check=True, text=True, capture_output=True)

        # we can now gather coverage information for the run
        log.info("Trying to gather coverage information...")

        # TODO: maybe change this to a more general path lol
        os.chdir(oss_dir.as_posix())
        log.debug(f"Changed working directory to '{oss_dir.as_posix()}'")
        _run_coverage_command(ctx, oss_dir, blob_dir)
        log.info(f"Running coverage command: {' '.join(command)}")
        subprocess.run(command, check=True, text=True, capture_output=True)
        log.info("Coverage information gathered")
    except subprocess.CalledProcessError as e:
        log.warning(
            f"Error gathering coverage information.\n"
            f"Please make sure you've got a coverage build present. "
            f"See: https://github.com/google/oss-fuzz/blob/master/docs/advanced-topics/code_coverage.md\n"
            f"{'stdout':-^40}\n{e.output}"
            f"\n"
            f"{'stderr':-^40}\n{e.stderr}"
            f"\n{'-'*46}"
        )
    finally:
        os.chdir(curr_dir)
        log.debug(f"Changed working directory back to '{curr_dir}'")


def _run_coverage_command(ctx: CrsContext, oss_dir: Path, blob_dir: Path) -> None:

    log.debug(f"Changed working directory to '{oss_dir.as_posix()}'")
    command = [
        "python3",
        "infra/helper.py",
        "coverage",
        "--no-serve",
        "--fuzz-target=" + ctx.proj_yaml_model.harnesses[ctx.harness_id].name,
        "--corpus-dir=" + Path(blob_dir).resolve().as_posix(),
        ctx.proj_yaml_model.cp_name,
    ]
    log.info(f"Running coverage command: {' '.join(command)}")
    subprocess.run(command, check=True, text=True, capture_output=True)
