"""Base state for chat and tool calling agents."""

from __future__ import annotations

from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from pydantic import BaseModel
from pydantic import Field
from pydantic import SerializeAsAny

from crs.base.model_list import get_temperature_for_model
from crs.base.model_list import get_top_p_for_model
from crs.base.settings import AI_MODEL_NAME
from crs.base.settings import AI_MODEL_TEMP
from crs.base.settings import AI_MODEL_TOP_P


class BaseChatState(BaseModel):
    """
    Base state used by other classes in this file. If your agent uses classes from this file
    your state should be derived from this class.
    """

    messages: list[SerializeAsAny[BaseMessage]]
    processed_tools: set[str] = Field(default_factory=set)  # str is tool call id

    temperature: float | None
    top_p: float | None

    @classmethod
    def from_messages(cls, messages: list[BaseMessage]) -> BaseChatState:
        """Create state from message list."""
        temperature = AI_MODEL_TEMP.get()
        if temperature is None:
            temperature = get_temperature_for_model(AI_MODEL_NAME.get())

        top_p = AI_MODEL_TOP_P.get()
        if top_p is None:
            top_p = get_top_p_for_model(AI_MODEL_NAME.get())

        return cls(temperature=temperature, top_p=top_p, messages=messages)

    @property
    def latest_tool_messages(self) -> list[ToolMessage]:
        """Get the latest message(s) that represent concluded tool calls."""
        result: list[ToolMessage] = []

        for msg in reversed(self.messages):
            if isinstance(msg, ToolMessage):
                result.append(msg)
            else:
                break

        return result
