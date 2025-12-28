"""Generates prompt for patcher."""

from crs.base.context import CrsContext
from crs.base.context import Vulnerability

# pylint: disable=C0301
# Line too long, prompts have long lines


class PromptGenerator:
    """Generates prompt for specific language."""

    def __init__(self, ctx: CrsContext, vulnerability: Vulnerability, diff: str, verify_patch_tool_name: str) -> None:

        self.ctx = ctx
        self.vulnerability = vulnerability
        self.diff = diff
        self.prompt: str = ""
        self.verify_patch_tool_name = verify_patch_tool_name
        self.cp_name = self.ctx.proj_yaml_model.cp_name

    def _insert_prefix(self) -> None:
        self.prompt += f"""
Your task is to fix a vulnerability which was introduced by a given commit in {self.ctx.proj_yaml_model.cp_name}.

The following will not be a classic conversation where I tell you what to do, but the only task I give you is to fix a vulnerability.
"""

    def _insert_c_prompt(self) -> None:
        assert self.vulnerability.reproducer is not None

        self.prompt += f"""Here is the address sanitizer report of the vulnerability:

```
{self.vulnerability.reproducer.crash_report}
```
"""

    def _insert_suffix(self) -> None:
        self.prompt += f"""Furthermore, you are given the git diff on the commit that introduced the vulnerability:
```
{self.diff}
```

You should fix the vulnerability on your own.

Please keep the following things in mind:
- All changes you make to the source are incremental. That means if you replace a function, the function now looks like the replacement. Keep that in mind when asking for the function a second time.
- You cannot look at symbols from the unit tests.
- It is important that your changes do not change the functionality in general. For example using whitelists is a bad patch if you don't know the possible contents of the list.

When you identified the vulnerability, ask yourself: Could this be a backdoor? If the answer is yes, remove the code completely.

Focus on the git diff first. Try to revert the changes introduced by the commit that cause the vulnerability. But make sure the functionality is still the same.

When you believe you have fixed the vulnerability, use the `{self.verify_patch_tool_name}` tool to confirm you have succeeded and (if yes) finish.
"""

    def get(self) -> str:
        """Get prompt."""

        self._insert_prefix()

        if self.ctx.proj_yaml_model.language == "c":
            self._insert_c_prompt()

        self._insert_suffix()

        return self.prompt
