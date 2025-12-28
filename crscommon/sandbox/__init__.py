"""Try to evaluate arbitrary Python code from an LLM without crashing and burning everything."""

from crscommon.sandbox.sandbox import run_in_sandbox
from crscommon.sandbox.sandbox import run_in_sandbox_ex

__all__ = ["run_in_sandbox", "run_in_sandbox_ex"]
