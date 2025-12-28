"""Modelling the project_yaml as pydantic object"""

from pathlib import Path
from typing import Any
from typing import Callable

import yaml
from pydantic import BaseModel
from pydantic import Field


class SourcesValue(BaseModel):
    """Value of an entry in the sources dict"""

    address: str  # URL/string for "git clone"-ing the source code.
    ref: str = Field(default="main")  # Git ref of the repository to check out (optional in YAML).
    artifacts: list[Path] = Field(default_factory=list)  # Build outputs (relative to CP directory).


class HarnessValue(BaseModel):
    """Value of an entry in the harness dict"""

    name: str  # Name (not to be confused with the ID). Pass this to "run.sh run_pov".
    source: Path  # Source file (relative to CP directory)
    binary: Path | None = None  # Compiled binary (relative to CP directory)


class ProjectYaml(BaseModel):
    """Modelling the project.yaml from a challenge project"""

    cp_name: str  # Name of the challenge project.
    language: str  # Programming language (for AIxCC semifinals, either "c" or "java").
    cp_sources: dict[str, SourcesValue]  # Source code subrepositories. Keys are subdirectories of "src/".
    docker_image: str | None = Field(default=None)  # Identifier of Docker image for run.sh operations.
    harnesses: dict[str, HarnessValue]  # Mapping from harness IDs ("id_NNN") to harness descriptions.

    @staticmethod
    def from_parsed_yaml(read_yaml: dict) -> "ProjectYaml":
        """initialize from parsed dict of a project yaml"""
        return ProjectYaml.model_validate(read_yaml)

    @staticmethod
    def from_cp_path(cp_path: Path) -> "ProjectYaml":
        """load project yaml from project path"""
        read_yaml = yaml.safe_load((cp_path / "project.yaml").read_text())
        return ProjectYaml.model_validate(read_yaml)

    @property
    def all_harness_ids(self) -> list[str]:
        """all harness keys as list"""
        return list(self.harnesses.keys())

    def harness_id_by_name(self, name: str) -> str:
        """get harness id by knowing the name of the harness"""
        return self.__search_dict_for_value(self.harnesses, name, lambda val, query: val.name == query)

    @property
    def all_cp_source_keys(self) -> list[str]:
        """all cp source keys as list"""
        return list(self.cp_sources.keys())

    @staticmethod
    def __search_dict_for_value(
        d: dict, to_search: Any, comparator: Callable[[Any, Any], bool] = lambda val, query: val == query
    ) -> Any:
        """get the key of a value in a dict, allows for a custom comparator, default is equality"""
        for k, v in d.items():
            if comparator(v, to_search):
                return k

        raise LookupError(f"No key found for value {to_search!r}")
