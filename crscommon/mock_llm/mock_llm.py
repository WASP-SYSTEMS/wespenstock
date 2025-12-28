"""mocking an LLM model to 'play' a predefined chat w/o actually calling a llm"""

import copy
import json
from pathlib import Path
from typing import Any
from typing import Iterator

from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import Input


class MockLLM:
    """Larping as LLM"""

    name: str = "MockLLM"

    def __init__(self, agent_json: dict[str, list[dict[str, Any]]]):
        """takes a recorded conversation"""

        self.message_objects: list[BaseMessage] = []
        self.ai_messages: list[AIMessage] = []

        # we do in place replacements but the original should stay the original :)
        agent_json_copy = copy.deepcopy(agent_json)
        for m in agent_json_copy["messages"]:

            new_message: SystemMessage | HumanMessage | AIMessage | ToolMessage
            match (message_type := m["type"]):
                case "system":
                    new_message = SystemMessage.model_validate(m)
                case "human":
                    new_message = HumanMessage.model_validate(m)
                case "ai":
                    metadata = m["response_metadata"]
                    metadata["model_group"] = f"{self.name}_{metadata['model_group']}"
                    new_message = AIMessage.model_validate(m)
                    self.ai_messages.append(new_message)
                case "tool":
                    new_message = ToolMessage.model_validate(m)
                case _:
                    raise ValueError(f"Can't load type '{message_type}'")

            self.message_objects.append(new_message)

        self.ai_response_generator: Iterator[AIMessage] = iter(self.ai_messages)

    # the signature of langgraph.Base.Runnable.invoke() - except the underscore
    # pylint: disable=W0613
    def invoke(self, _input: Input, config: RunnableConfig | None = None, **kwargs: Any) -> AIMessage:
        """
        function to call the "llm" it's only a wrapper tp return the next predefined AIMessage
        doesn't do any sanity checks for now. it just accepts everything and returns the next message
        """
        return next(self.ai_response_generator)

    @staticmethod
    def from_json_file(file: Path) -> "MockLLM":
        """secondary constructor from file"""

        content = json.loads(file.read_text())
        return MockLLM(content)
