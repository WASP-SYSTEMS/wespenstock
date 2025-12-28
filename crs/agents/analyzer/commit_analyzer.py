"""POV builder agent."""

from pathlib import PurePath

from git import Repo
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

from crs.agents.analyzer.analyzer_agent import AnalyzerAgent
from crs.agents.analyzer.prompt import AnalyzerPrompt
from crs.base.constants import ALLOWED_PATCH_EXTENSIONS
from crs.base.context import CrsContext
from crs.base.settings import ANALYZER_CHECK_SANITY
from crs.logger import CRS_LOGGER

log = CRS_LOGGER.getChild(__name__)


class CommitAnalyzer(AnalyzerAgent):
    """
    Analyze a specific commit in the codebase.
    """

    additional_prompt = """Use the followings steps to analyze the commit:

Step 1 - Gather context about the code snippets modified by the commit. For example, if a method was changed, determine its role within the codebase. Consider how the change impacts the overall behavior of the system.
         Obtain all necessary context about the referenced symbols within the modified code. You must fully understand what the code did before the change, how its behavior changes by the commit, and how that affects the overall behavior of the codebase.

Step 2 - Now that you fully understand the modified code snippets, assess whether the changes could introduce any vulnerabilities. Could they lead to unintended behavior or cause a crash?
         Reflect on the behavioral changes introduced by the commit and identify potential points where things could go wrong.

Step 3 - Describe your findings clearly and thoroughly. Remember, your response will be given to another researcher as a starting point to investigate and potentially fix any introduced vulnerabilities. Try to help him out as much as possible.
         Explain the modified code snippets, how their behavior has changed, and how these changes may or may not introduce vulnerabilities. Provide the researcher with all the necessary context to understand your analysis.
"""

    def commit_is_sane(self, repository: Repo) -> None:
        """
        Applies basic sanity checks to the commit.

        In particular, checks if the commit changed any source files.
        """
        vuln_commit = repository.commit(self.ctx.commit_hash)
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
                return

        log.error("Skipping analyzer due to failed sanity checks ~.~")
        raise ValueError("Commit sanity check failed")

    def __init__(self, ctx: CrsContext):
        """
        Inits the function analyzer.
        """
        super().__init__(ctx=ctx)
        # checkout commit to analyze
        repository = self.ctx.comp_env.checkout(ctx.subproj_commit)
        if ANALYZER_CHECK_SANITY.get():
            self.commit_is_sane(repository)

    def _get_initial_messages(self) -> list[BaseMessage]:
        msgs = [
            SystemMessage(content=AnalyzerPrompt.get_system_prompt()),
            HumanMessage(
                content=AnalyzerPrompt.get_prompt(
                    self.ctx.proj_yaml_model.language,
                    self.ctx.cp_name,
                    self.target_code,
                    "commit",
                    additional_prompt=self.additional_prompt,
                )
            ),
        ]
        return msgs
