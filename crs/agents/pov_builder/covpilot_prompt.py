"""We build the CovPilot initial prompt here"""

from crs.agents.pov_builder.coverage.coverage_report import FunctionCoverage
from crs.base.context import CrsContext
from crs.base.settings import POV_USE_COVERAGE


def get_covpilot_persona_desc() -> str:
    """get CovPilot persona description"""
    return """
    You are a professional computer scientist at a well-known cybersecurity facility.
    Your task is to write a Python script producing an input that reaches a certain function in a program.
    These inputs are important to test if certain parts of the program are reachable.
    The input is passed to the program using an existing harness, originally written for a fuzzer.
    You will receive information about the performance of your input in the form of a short coverage report.
    Be sure to differentiate between input explanations and real values when generating all inputs for the harness.
    You will generate the real input and not use placeholder values.
    """


def get_covpilot_system_prompt() -> str:
    """get the CovPilot prompt that can be used as system prompt including persona"""
    return f"""
{get_covpilot_persona_desc()}
Carefully heed the user's instructions.
Respond using Markdown.
"""


def get_covpilot_standalone_prompt(
    ctx: CrsContext,
    harness_name: str,
    harness_code: str,
    target_function_name: str,
    code_path: list[str],
    caller_function_name: str,
    caller_function_code: str,
    include_persona: bool = False,
    additional_prompting: str = "",
) -> str:
    """get CovPilot prompt for usage without prior analyzer"""

    prompt = f"""
{get_covpilot_persona_desc() if include_persona else ''}

The project you're working on is called {ctx.cp_name}.

You need to reach the function `{target_function_name}`.
Write a script that generates an input that reaches this function.
A potential code path to reach this function is: {code_path}

An automated coverage evaluation will assess what sections of the code path your input reached.

The code of the caller function named {caller_function_name} is:
```
{caller_function_code}
```

The input will be passed to the program using a harness called '{harness_name}'.
Here is the code of the harness:
```
{harness_code}
```
"""
    if additional_prompting:
        prompt += f"\n\n{additional_prompting}"

    if POV_USE_COVERAGE.get():
        prompt += "\nYou will receive feedback on your provided inputs in form of coverage reports.\n"
        prompt += FunctionCoverage.get_code_coverage_explanation()

    return prompt.strip()
