"""We build the PoV builders initial prompt here"""

from crs.base.context import CrsContext
from crs.base.context import Vulnerability


# pylint: disable=line-too-long
class VerifierPrompt:
    """Prompts for the Verifier"""

    @staticmethod
    def get_persona_desc() -> str:
        """get persona description"""
        return """
You are a professional exploit researcher at a well-known cybersecurity facility.
Your task is to write a Python script that generates an input that causes a given program to crash.
Think of this input as a reproducer, like they are known from fuzzing. This reproducer can be used to demonstrate a crash using a given fuzzing harness.
If the reproducer requires a certain size in characters or values to trigger, feel free to overcommit a little to ensure a hit.
Be sure to differentiate between input explanations and real values when generating all inputs for the harness.
You will not use placeholder values."""

    @staticmethod
    def get_system_prompt() -> str:
        """get the prompt that can be used as system prompt including persona"""
        return f"""{
VerifierPrompt.get_persona_desc()}
Carefully heed the user's instructions.
Respond using Markdown.
    """

    @staticmethod
    def get_prompt(
        ctx: CrsContext,
        vulnerability: Vulnerability,
        harness_name: str,
        harness_code: str,
        target_code: str,
        type_of_target: str = "commit",
        standalone_pov_builder_prompt: bool = False,
        additional_prompting: str = "",
    ) -> str:
        """get verifier prompt"""

        description_text = ""
        if not standalone_pov_builder_prompt:
            assert vulnerability is not None
            if vulnerability.report is not None:
                description_text = f"""
Vulnerability report done by our researcher:
--- BEGIN VULNERABILITY REPORT
{vulnerability.report}
--- END VULNERABILITY REPORT
"""

        prompt = f"""
A researcher of our team analyzed some {type_of_target}s of the {ctx.cp_name} repository.
He found one {type_of_target} in particular that he thinks introduces a vulnerability.

The {type_of_target} to analyze is as follows:
```
{target_code}
```
{description_text}
We wrote several harnesses that feed input to the program, like you would in fuzzing. Use the harness '{harness_name}'.
The vulnerability will not be in the harness itself. The harness is just to trigger a vulnerability hidden somewhere else in the code.
Code of the harness:
```
{harness_code}
```

Your task is to create a Python script that generates an input that triggers the vulnerability, leading the program to crash.
We use several sanitizers that detect different program behaviours.
For {ctx.cp_name} in particular, we use the AddressSanitizer.

Use the following steps:

Step 1 - Gather enough context to understand the vulnerability and its role within the code base.
         Trace the program flow starting from the harness to the vulnerable code to fully grasp how the vulnerability can be reached and triggered.

Step 2 - Once you have a sufficient understanding of the vulnerability, the codebase, and the program flow, start generating Python scripts to trigger the vulnerability.
         If your script does not trigger the vulnerability, go back to step 1. Question your own understanding of the code base and program flow. Gather more context to get a deeper understanding.

Iterate between the two steps until you find a Python script that works and triggers the vulnerability.
Make sure to differentiate between input explanations and real values when generating inputs for the harness.
If the reproducer requires a certain size in characters or values to trigger, feel free to overcommit a little to ensure a hit.
"""

        if additional_prompting:
            prompt += f"\n\n{additional_prompting}"

        return prompt
