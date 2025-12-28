"""
Interface to communicate that our BaseTools knows its name as a class var.
This allows mentioning of tool name without requiring access to an instance.
"""

from abc import ABC
from typing import ClassVar

from langchain_core.tools import BaseTool


class NamedBaseTool(BaseTool, ABC):
    """BaseTool but requiring a .NAME class var."""

    NAME: ClassVar[str]
    """The name the tool is called especially when communicating with the LLM."""
