"""Context of CRS which is passed to subagents."""

from __future__ import annotations

from pathlib import Path

from pydantic import Base64Bytes
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from crs.aixcc.env import CompetitionEnvironment
from crs.aixcc.env import SubprojectCommit
from crs.aixcc.project_yaml import ProjectYaml


class VulnerabilityReport(BaseModel):
    """Structured vulnerability description produced by the Analyzer."""

    description: str
    """Description of the vulnerability."""

    vuln_functions: list[str]
    """Functions causing the vulnerability."""

    type: str
    """Type of the vulnerability."""


class ReproducerDescription(BaseModel):
    """
    Output of the reproducer (pov) builder agent.
    """

    blob: Base64Bytes  # the reproducer (pov) input blob (CAUTION: must be base64 when passed to the constructor)
    crash_report: str  # address sanitizer report/stacktrace etc.
    commit_hash: str  # hash of the source project commit introducing this reproducer (pov)
    sanitizer_name: str  # name of a sanitizer from project.yaml that is triggered by this reproducer (pov)
    harness_id: str  # id of harness from project.yaml that this reproducer (pov) works against
    path: Path  # path where the reproducer (pov) was written to


class PatchDescription(BaseModel):
    """
    Output of patcher agent.

    Needs a sibling PoV description for the ID of the vulnerability being patched.
    """

    patch_diff: str  # the patch git diff


class Vulnerability(BaseModel):
    """
    Describes a vulnerability using a report, a reproducer and a patch.
    """

    report: VulnerabilityReport | None = None
    reproducer: ReproducerDescription | None = None
    patch: PatchDescription | None = None


class CrsContextBase(BaseModel):
    """
    Subset of CrsContext holding the nonvolatile fields.

    This must not contain any absolute paths.
    """

    # NOTE: For the analyzer and the PoV builder agents, the commit hash and harness ID are those that the agent is
    #       supposed to analyze. For the patcher agent, they must match the PoV description. That is, update them
    #       before starting the patcher.

    # TODO: We only look at one commit of one source subproject -- all others are at their target refs.
    #       We cannot even express analyzing a combination of different commits of different source subprojects.

    cp_id: str  # name of challenge project directory (relative to the CP root)
    cp_name: str  # name of challenge project
    src_path_rel: Path  # path to selected source repo inside the CP (relative to CrsContext.cp_path)
    commit_hash: str  # hash of commit inside src_path (see note above)
    harness_id: str  # harness ID in project.yaml (see note above)
    target_ref: str  # tag/branch/ref inside src_path at which the vulnerability must still work (e.g. main)

    vulnerabilities: list[Vulnerability] = []

    viewed_files: list[Path]  # List of viewed files across all agents

    save_to: Path | None

    @property
    def subproj_commit(self) -> SubprojectCommit:
        "Bundle up src_path_rel and commit_hash into a SubprojectCommit object."
        return SubprojectCommit(src_id=self.src_path_rel.relative_to("src").as_posix(), ref=self.commit_hash)

    @property
    def subproj_target_ref(self) -> SubprojectCommit:
        "Bundle up src_path_rel and target_ref into a SubprojectCommit object."
        return SubprojectCommit(src_id=self.src_path_rel.relative_to("src").as_posix(), ref=self.target_ref)

    def nonvolatile_dict(self) -> dict:
        "Dump the non-volatile parts of this context into a JSON-compatible dictionary."
        fields: dict[str, FieldInfo] = CrsContextBase.model_fields  # type: ignore  # That's a *property*, MyPy!
        return self.model_dump(include=set(fields), mode="json")

    def save(self) -> None:
        """Save this context to the file given on construction."""
        if self.save_to is not None:
            self.save_to.write_text(self.model_dump_json())


class CrsContext(CrsContextBase):
    """
    Main context of CRS which holds all information gathered by agents.
    """

    comp_env: CompetitionEnvironment  # Simulacrum of competition environment (TODO: yeet competition reference)
    cp_path_abs: Path  # absolute path to challenge project
    proj_yaml_model: ProjectYaml  # the same as the project yaml, but as base model

    @property
    def source_path_abs(self) -> Path:
        """Get the absolute source path"""
        return self.cp_path_abs / self.src_path_rel
