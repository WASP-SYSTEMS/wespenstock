"""a few specific exceptions for our CRS"""


class ContextWindowExceededError(Exception):
    """Raised when context of LLM is full"""


class ToolCallNotAnsweredError(Exception):
    """To be raised when we get the message that we didn't answer all our tool calls"""


class RunScriptExecutionError(Exception):
    """To be raised when the run.sh script fails"""


class TooManyInvalidToolCalls(Exception):
    """To be raised when the LLM produces invalid tool calls and we want to end the agent"""


class TooManyErrorsError(RuntimeError):
    """
    Sometimes it just be like that.
    Mainly used in the verifier working on multiple reports.
    """
