"""
Try to evaluate arbitrary Python code from an LLM without crashing and burning everything.

As time progresses and we try using more and more LLMs, the code snippets they come up with become less and less
well-behaved. Correspondingly, we keep developing more and more sophisticated barriers between the LLM code and the
rest of the system.
"""

from __future__ import annotations

import io
import multiprocessing
import queue
import resource
import sys
import traceback
import types
import typing
from multiprocessing.queues import Queue

from crs.base.util import format_text_list

DEFAULT_TIMEOUT = 10  # seconds
DEFAULT_MEM_LIMIT = 4 * 2**30  # bytes

TERMINATE_TIMEOUT = 1  # second

T = typing.TypeVar("T")

ResultType = type[T] | types.UnionType
ResultDict = dict[str, typing.Any]


def describe_type(ty: ResultType) -> str:
    "Return a textual description of the given type."
    if typing.get_origin(ty) is types.UnionType:
        return format_text_list(map(describe_type, typing.get_args(ty)), joiner="or", if_empty="(empty union?!)")
    assert not isinstance(ty, types.UnionType)
    try:
        return repr(ty.__name__)
    except AttributeError:
        return repr(ty)


def _sandbox_main(
    code: str, result_var: str, require_type: ResultType | None, mem_limit: int, result_queue: multiprocessing.Queue
) -> None:
    "Main function inside the sandbox."
    resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))

    result_dict: ResultDict = {}

    # We don't really use these, but at least they don't pollute our consoles this way.
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    try:
        scope = {result_var: None}
        exec(code, scope)  # pylint: disable=W0122

        # This should immediately produce the right kind of exception. :3
        result = eval(result_var, scope)  # pylint: disable=W0123

        if require_type is not None:
            if not isinstance(result, require_type):
                raise TypeError(f"Invalid {result_var} type {type(result)}; expected {describe_type(require_type)}")

    except:  # pylint: disable=W0702  # Yes, this is the one place where we really want this.
        exc_info = sys.exc_info()
        result_dict.update(error=exc_info[1], traceback=traceback.format_exc())

    else:
        result_dict.update(result=result)

    finally:
        result_dict.update(stdout=sys.stdout.getvalue(), stderr=sys.stderr.getvalue())
        result_queue.put(result_dict)


def run_in_sandbox_ex(
    code: str,
    result_var: str,
    require_type: ResultType | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    mem_limit: int = DEFAULT_MEM_LIMIT,
) -> ResultDict:
    "Run Python code in a somewhat isolated environment and return details about how it went."
    result_queue: Queue[ResultDict] = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_sandbox_main, args=(code, result_var, require_type, mem_limit, result_queue), daemon=True
    )
    proc.start()

    try:
        result_dict = result_queue.get(timeout=timeout)
    except queue.Empty as exc:
        raise TimeoutError("Sandboxed code execution timed out") from exc
    else:
        return result_dict
    finally:
        if proc.exitcode is None:
            proc.terminate()
            proc.join(TERMINATE_TIMEOUT)
        if proc.exitcode is None:
            proc.kill()
            proc.join(TERMINATE_TIMEOUT)


def run_in_sandbox(
    code: str,
    result_var: str,
    require_type: ResultType[T] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    mem_limit: int = DEFAULT_MEM_LIMIT,
) -> T:
    "Run Python code in a somewhat isolated environment and return the result or raise an error."
    result_dict = run_in_sandbox_ex(code, result_var, require_type, timeout, mem_limit)

    if result_dict.get("error"):
        exc_type = type(result_dict["error"])
        exc_value = result_dict["error"]
        exc_traceback = result_dict["traceback"]
        raise exc_type(f"{exc_value}\n{exc_traceback}")

    return result_dict["result"]
