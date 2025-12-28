"""We build the PoV builders initial prompt here"""

from crs.agents.tools.lsp_tools import GetSymbolDefinition


def get_coverage_prompt(target_function_coverage: str | None = None, searched_symbols_cov: str | None = None) -> str:
    """get a prompt for the pov builder, decide if you wanna use analyzer inputs or not"""
    # pylint: disable=line-too-long
    prompt = """
Coverage reports for the provided reproducer have been generated.

Think about how the coverage info relates to the inputs that you generate. \
Consider how you need to change your approach to reach the line that you need to reach \
in order to exploit the vulnerability before generating new inputs."""

    if target_function_coverage is not None:
        prompt += f"""

The coverage of the target function is:\n
{target_function_coverage}
"""

    if searched_symbols_cov is not None:
        prompt += f"""
You also used the tool '{GetSymbolDefinition.NAME}' to search for these symbols:\n
{searched_symbols_cov if searched_symbols_cov else '<no function covered>'}
"""

    return prompt
