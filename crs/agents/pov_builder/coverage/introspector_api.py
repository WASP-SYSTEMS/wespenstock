"""Introspector API."""

from __future__ import annotations

import pickle
from pathlib import Path

import lxml.html
import requests
import yaml
from pydantic import BaseModel
from pydantic import Field

from crs.logger import CRS_LOGGER

log = CRS_LOGGER.getChild(__name__)


class ProjectWithIntrospectorArtifacts(BaseModel):
    """Project with valid introspector artifacts available."""

    name: str
    introspector_url: str
    harness_names: list[str] = Field(default_factory=list)


class ValidProjectFile(BaseModel):
    """List of valid projects"""

    projects: list[ProjectWithIntrospectorArtifacts]


class IntrospectorApi:
    """Class managing interaction with the introspector API."""

    INTROSPECTOR_PROJECT_LIST_URL = "https://oss-fuzz-introspector.storage.googleapis.com/index.html"
    CORRELATION_FILENAME = "exe_to_fuzz_introspector_logs.yaml"

    def __init__(self, projects: list[ProjectWithIntrospectorArtifacts]) -> None:
        self.projects = projects

    def download(self, dst_path: Path) -> None:
        """Download all artifacts."""

        for project in self.projects:
            proj_path = dst_path / project.name

            # check that artifact is not empty
            if proj_path.exists() and any(proj_path.iterdir()):
                log.info(f"Artifacts found for project {project.name}")
                continue

            self._download_fuzz_report(project, proj_path)

    def _download_fuzz_report(self, project: ProjectWithIntrospectorArtifacts, dst_path: Path) -> None:
        """
        Download the fuzz report for a project and save it at dst_path.
        """

        dst_path.mkdir(parents=True, exist_ok=True)

        fuzz_report_url = project.introspector_url + "fuzz_report.html"
        r = requests.get(fuzz_report_url, timeout=16)
        if not r.ok:
            raise requests.exceptions.RequestException(fuzz_report_url)

        html_doc: lxml.html.HtmlElement = lxml.html.document_fromstring(r.text)
        metadata_section = html_doc.get_element_by_id("Metadata-section")
        table = metadata_section.find_class("cell-border")

        if not len(table) > 0:
            raise RuntimeError("Failed to download artifacts: Empty table")

        for link in table[-1].iterlinks():
            url = project.introspector_url + link[2]

            r = requests.get(url, timeout=180)
            if not r.ok:
                raise requests.exceptions.RequestException(f"Failed to download file: {url}")

            dts_file: Path = dst_path / link[2]

            dts_file.write_text(r.text)

            log.info(f"Downloaded file: {url}")

            # The yaml files can be above 30mb and parsing is very time consuming,
            # so we parse once and store the pickled python object.
            if dts_file.suffix == ".yaml":
                with open(dts_file, encoding="utf-8") as file:
                    log.info(f"Parsing YAML file {dts_file}")
                    all_functions_yaml = yaml.safe_load(file)

                    log.info("Dumping file as pickle object")
                    with open(str(dts_file) + ".pickle", "wb") as pfile:
                        pickle.dump(all_functions_yaml, pfile, protocol=pickle.HIGHEST_PROTOCOL)

        # download correlation file
        correlation_url = project.introspector_url + self.CORRELATION_FILENAME
        r = requests.get(correlation_url, timeout=180)
        if not r.ok:
            raise requests.exceptions.RequestException(f"No correlation file found: {correlation_url}")

        (dst_path / self.CORRELATION_FILENAME).write_text(r.text)
