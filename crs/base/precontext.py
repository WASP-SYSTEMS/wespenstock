"""Context used while building the CrsContext."""

import enum
import json
from pathlib import Path

from git import Repo
from pydantic import BaseModel

from crs.aixcc.env import CompetitionEnvironment
from crs.aixcc.env import aixcc_local_env
from crs.aixcc.project_yaml import ProjectYaml
from crs.base.context import CrsContext
from crs.logger import CRS_LOGGER

log = CRS_LOGGER.getChild(__name__)


class VulnerabilityStateFields(enum.Enum):
    "Fields of the type `Vulnerability` which can be injected into a CrsContext."

    REPORT = "report"
    REPRODUCER = "reproducer"


class CrsPreContext(BaseModel):
    """
    The context used while building the CrsContext!
    """

    cp_path: Path  # absolute path to challenge project
    src_path_in_cp: Path | None = None  # path to source repo inside the CP's src/ (or absolute)
    commit_hash: str | None = None  # commit hash to analyze
    harness_id: str | None = None  # harness ID to use
    project_yaml: ProjectYaml | None = None  # may be set if the file is already loaded
    comp_env: CompetitionEnvironment | None = None  # set when running in competition mode
    state_data: dict | None = None  # for non-first steps: get CRS state from here (overrides state_file_load)
    state_file_load: Path | None = None  # for individual steps: read CRS state from this file
    state_file_save: Path | None = None  # for individual steps: save CRS state into this file

    def to_context(self, apply_state: set[VulnerabilityStateFields] | None = None) -> CrsContext:
        "Build a full CrsContext from this pre-context."

        # pylint: disable=R0912,R0915

        if apply_state is None:
            apply_state = set()

        cp_path = self.cp_path.absolute()
        project_yaml = self.project_yaml
        if project_yaml is None:
            project_yaml = ProjectYaml.from_cp_path(cp_path)

        src_path_in_cp = self.src_path_in_cp
        if src_path_in_cp is None:
            if len(project_yaml.cp_sources) != 1:
                raise RuntimeError("Challenge project does not have a unique source repository; specify it explicitly")
            src_path_in_cp = Path(next(iter(project_yaml.cp_sources)))

        cp_src_dir = cp_path / "src"
        src_path_abs = cp_src_dir / src_path_in_cp
        if not src_path_abs.is_relative_to(cp_src_dir):
            raise RuntimeError("Source subproject must be inside the challenge project")
        src_path_rel = src_path_abs.relative_to(cp_path)

        # Validate the source repo and find its target_ref.
        for subpath, desc in project_yaml.cp_sources.items():
            if (cp_src_dir / subpath) != src_path_abs:
                continue
            target_ref = desc.ref
            break
        else:
            raise RuntimeError(f"Could not find source subproject {src_path_abs.relative_to(cp_src_dir)}")

        commit_hash = self.commit_hash
        if commit_hash is None:
            repo = Repo(src_path_abs)
            commit_hash = repo.commit(target_ref).hexsha

        harness_id = self.harness_id
        if harness_id is None:
            # Usually, this will end up picking "id_1".
            harness_id = min(project_yaml.harnesses)
        elif harness_id not in project_yaml.harnesses:
            raise RuntimeError(
                f"Invalid harness ID {harness_id}; must be one of {', '.join(project_yaml.harnesses)} for this CP"
            )

        comp_env = self.comp_env
        if comp_env is None:
            comp_env = aixcc_local_env(cp_path)
        elif comp_env.cp_path != cp_path:
            raise RuntimeError(
                f"CP path mismatch: CrsPreContext has {cp_path}, CompetitionEnvironment has {comp_env.cp_path}"
            )

        result = CrsContext(
            cp_id=cp_path.name,
            cp_name=project_yaml.cp_name,
            cp_path_abs=cp_path,
            src_path_rel=src_path_rel,
            commit_hash=commit_hash,
            harness_id=harness_id,
            target_ref=target_ref,
            proj_yaml_model=project_yaml,
            comp_env=comp_env,
            save_to=self.state_file_save,
        )

        state_data = self.state_data
        if state_data is None and self.state_file_load:
            try:
                state_data = json.loads(self.state_file_load.read_text())
            except FileNotFoundError:
                pass
        if state_data is not None:
            vuln_update: list[dict[str, object]] = []
            for vuln in state_data["vulnerabilities"]:
                vuln_update.append({k.value: vuln[k.value] for k in apply_state if vuln.get(k.value)})

            result = CrsContext.model_validate(result.model_dump() | {"vulnerabilities": vuln_update})

        log.info(f"Working on context:\n{result.model_dump_json(indent=2)}")

        return result
