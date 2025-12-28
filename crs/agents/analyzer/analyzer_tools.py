"""Tools for Analyzer."""

from typing import ClassVar

from pydantic import BaseModel
from pydantic import Field

from crs.agents.constants import DISPOSITION_TASK_DONE
from crs.agents.tools.named_base_tool import NamedBaseTool
from crs.base.context import CrsContext
from crs.base.context import Vulnerability
from crs.base.context import VulnerabilityReport
from crs.logger import CRS_LOGGER

log = CRS_LOGGER.getChild(__name__)


class SubmitVulnerabilityReport(NamedBaseTool):

    # pylint: disable=missing-class-docstring
    class Args(BaseModel):
        description: str = Field(description="A brief explanation of the identified vulnerability.")
        vuln_functions: list[str] = Field(
            description=("A list of function names that contribute to the vulnerability.")
        )
        vuln_type: str = Field(description="Specifies the category of the vulnerability.")

    # (...) attributes annotated with typing.ClassVar will be automatically excluded from the model. ~pydantic docs
    NAME: ClassVar[str] = "SubmitVulnerabilityReport"
    """Same as .name but as a class variable, can be used to access the tool name without having an instance of it."""

    name: str = NAME
    description: str = (
        "This tool facilitates the submission of vulnerability descriptions found during source code analysis. "
        "It enables efficient documentation of security flaws, including details like vulnerability type and "
        "affected code."
    )

    ctx: CrsContext = Field(exclude=True)

    args_schema: type[BaseModel] = Args

    # pylint: disable=arguments-differ
    def _run(self, description: str, vuln_functions: list[str], vuln_type: str) -> str | dict[str, str]:

        # pylint: disable=no-member
        self.ctx.vulnerabilities.append(
            Vulnerability(
                report=VulnerabilityReport(
                    description=description,
                    vuln_functions=vuln_functions,
                    type=vuln_type,
                )
            )
        )

        self.ctx.save()

        log.info(f"Analyzer identified a potential vulnerability of type '{vuln_type}'")

        return "The vulnerability was successfully reported."


class EndAnalysis(NamedBaseTool):

    # pylint: disable=missing-class-docstring
    class Args(BaseModel):
        """Args"""

    # (...) attributes annotated with typing.ClassVar will be automatically excluded from the model. ~pydantic docs
    NAME: ClassVar[str] = "EndAnalysis"
    """Same as .name but as a class variable, can be used to access the tool name without having an instance of it."""

    name: str = NAME
    description: str = (
        "End the vulnerability analysis. This tool should only be called after all vulnerabilities"
        "have been reported."
    )

    args_schema: type[BaseModel] = Args

    # pylint: disable=arguments-differ
    def _run(self) -> str | dict[str, str]:
        return {
            "content": DISPOSITION_TASK_DONE,
            "disposition": DISPOSITION_TASK_DONE,
        }
