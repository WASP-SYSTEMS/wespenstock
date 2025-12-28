"Interaction with the AIxCC competition environment."

from functools import cached_property
from pathlib import Path
from typing import overload

from git import Repo
from pydantic import BaseModel
from pydantic import Field

from crs.aixcc.project_yaml import ProjectYaml
from crs.aixcc.scripts import RunResult
from crs.aixcc.scripts import invoke_run_sh
from crs.logger import CRS_LOGGER

log = CRS_LOGGER.getChild(__name__)


class SubprojectCommit(BaseModel):
    "Parameters for checking out a CP."

    src_id: str  # Name of directory inside the CP's src/
    ref: str  # Git ref to check out in src_id


class BuildWithPatch(BaseModel):
    "Parameters for applying a patch while building a CP."

    patch_file: Path  # Should be absolute
    src_id: str  # Name of directory inside the CP's src/


class CompetitionEnvironment(BaseModel):
    "Environment provided to the CRS with a particular challenge project already selected"

    cp_path: Path = Field(description="Location of the CP to process (may be outside cp_root).")

    @cached_property
    def project_yaml(self) -> ProjectYaml:
        "The project.yaml file of this challenge project"
        return ProjectYaml.from_cp_path(self.cp_path)

    def repo(self, src_id: str | None) -> Repo:
        "Return a Repo object denoting the CP or a source subrepository"

        # TODO: Is the CP required to be a Git repository?
        if src_id is None:
            return Repo(self.cp_path)
        if src_id not in self.project_yaml.cp_sources.keys():
            raise RuntimeError(f"Source subrepository {src_id} does not exist")
        return Repo(self.cp_path / "src" / src_id)

    @overload
    def checkout(self, src_commit: SubprojectCommit, also_clean: bool = ...) -> Repo: ...
    @overload
    def checkout(self, src_commit: None = ..., also_clean: bool = ...) -> None: ...

    def checkout(self, src_commit: SubprojectCommit | None = None, also_clean: bool = False) -> Repo | None:
        "Checkout the target refs of the source subprojects, with one subproject commit overridable"

        return_repo: Repo | None = None

        for src_id, src_desc in self.project_yaml.cp_sources.items():
            repo = self.repo(src_id)
            ref = src_desc.ref
            if src_commit is not None and src_id == src_commit.src_id:
                ref = src_commit.ref
                return_repo = repo

            repo.git.checkout(ref, force=True)
            log.info(f"Checked out {src_id} at {ref}")
            if also_clean:
                repo.git.clean("-dfx")

        if src_commit is not None and return_repo is None:
            raise RuntimeError(f"Source subrepository {src_commit.src_id} does not exist")

        return return_repo

    def build(self, patch: BuildWithPatch | None = None) -> RunResult:
        "Build the CP"

        if patch:
            log.info(f"Building {self.cp_path} with patch {patch.patch_file} applied to {patch.src_id}")
            return invoke_run_sh(["build", patch.patch_file.as_posix(), patch.src_id], self.cp_path)

        log.info(f"Building {self.cp_path}")
        return invoke_run_sh(["build"], self.cp_path)

    def run_pov(self, input_path: Path, harness_name: str) -> RunResult:
        "Run a pov blob against a harness"

        log.info(f"Running PoV in {self.cp_path}")
        return invoke_run_sh(["run_pov", input_path.as_posix(), harness_name], self.cp_path)

    def run_tests(self) -> RunResult:
        "Run the functionality tests"

        log.info(f"Running tests in {self.cp_path}")
        return invoke_run_sh(["run_tests"], self.cp_path)

    def run_custom_command(self, cmd: list[str]) -> RunResult:
        "Run a custom command"

        log.info(f"Running custom command in {self.cp_path}")
        return invoke_run_sh(["custom"] + cmd, self.cp_path)


def aixcc_local_env(cp_path: Path) -> CompetitionEnvironment:
    "Build a competition environment for non-competition mode."
    return CompetitionEnvironment(cp_path=cp_path)
