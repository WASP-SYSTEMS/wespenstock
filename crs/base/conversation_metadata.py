"""Metadata of the conversation with the LLM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolCall
from langchain_core.messages import ToolMessage
from pydantic import BaseModel

from crs.base.base_state import BaseChatState


class AIMessageMetadata(BaseModel):
    """Metadata of AI message."""

    model: str

    perplexity: float | None
    joint_logprob: float | None
    mean_prob: float | None

    total_tokens: int
    prompt_tokens: int
    completion_tokens: int

    eval_duration: float | None

    content: str
    tool_calls: list[ToolCall]


class ConversationMetadata(BaseModel):
    """
    Metadata of the conversation with the LLM.

    This class can be used for evaluation purposes. A log created with to_json_file() can be
    loaded again using from_json_file() and used in evaluation scripts.
    """

    temperature: float | None
    top_p: float | None

    messages: list[AIMessageMetadata | BaseMessage]

    @staticmethod
    def from_state(state: BaseChatState) -> ConversationMetadata:
        """Create this object from a chat state."""

        metadata = ConversationMetadata(temperature=state.temperature, top_p=state.top_p, messages=[])

        for msg in state.messages:
            if isinstance(msg, AIMessage):
                confidences = ConversationMetadata.extract_logprob_metrics(msg)

                meta_msg = AIMessageMetadata(
                    model=ConversationMetadata.extract_model_name(msg),
                    perplexity=confidences["perplexity"],
                    joint_logprob=confidences["joint_logprob"],
                    mean_prob=confidences["mean_prob"],
                    total_tokens=msg.response_metadata["token_usage"]["total_tokens"],
                    prompt_tokens=msg.response_metadata["token_usage"]["prompt_tokens"],
                    completion_tokens=msg.response_metadata["token_usage"]["completion_tokens"],
                    eval_duration=ConversationMetadata.extract_eval_duration(msg),
                    content=str(msg.content),
                    tool_calls=msg.tool_calls,
                )

                metadata.messages.append(meta_msg)

            else:
                metadata.messages.append(msg)

        return metadata

    @staticmethod
    def extract_logprob_metrics(msg: AIMessage) -> dict[str, float | None]:
        """
        Calculate and return confidence scores of an AIMessage using token log-probabilities.

        The following confidence scores are reported:
            perplexity: Perplexity (exp(-mean(token_logprobs)))
            joint_logprob: Joint log-probability (sum(token_logprobs))
            mean_prob: Mean token probability (mean(exp(token_logprobs)))

        Note: Only processes an output from OpenAI; Anthropic doesn't provide the needed data; Google: not researched.
        If no logprobs are available, the confidence scores are None.
        """
        if not (logprobs := msg.response_metadata.get("logprobs")) or logprobs["content"] is None:
            return {"perplexity": None, "joint_logprob": None, "mean_prob": None}

        token_logprobs: list[float] = [token_elm["logprob"] for token_elm in logprobs["content"]]

        # The perplexity calculation is drawn from an OpenAI cookbook:
        # https://cookbook.openai.com/examples/using_logprobs#5-calculating-perplexity
        perplexity = np.exp(-np.mean(token_logprobs))
        # Log-probabilities are there for exactly this calculation. :)
        joint_logprob = sum(token_logprobs)
        # This might be interesting?..
        mean_prob = np.mean(np.exp(token_logprobs))

        return {"perplexity": perplexity, "joint_logprob": joint_logprob, "mean_prob": mean_prob}

    @staticmethod
    def extract_model_name(msg: AIMessage) -> str:
        """
        Extract the name of the model from which the AIMessage came from.

        Note: In some cases this information may be missing and "<Not available>" is returned.
        """

        try:
            model_name = msg.response_metadata["hidden_params"]["additional_headers"]["x-litellm-model-group"]
        except KeyError:
            # in some cases (with unknown steps to reproduce) messages from qwen do not have "hidden_params"s.
            model_name = "<Not available>"

        return model_name

    @staticmethod
    def extract_eval_duration(msg: AIMessage) -> float | None:
        """
        Extract the evaluation duration from the AIMessage.

        Note: In some cases this information may be missing and None is returned.
        """

        try:
            response_ms = msg.response_metadata["hidden_params"]["_response_ms"]
        except KeyError:
            response_ms = None

        return response_ms

    @staticmethod
    def from_json_file(file: Path) -> ConversationMetadata:
        """
        Create this model from json.
        """

        with file.open() as fp:
            return ConversationMetadata.model_validate(json.load(fp))

    def to_json_file(self, file_path: Path) -> None:
        """Dump the conversation to a machine-readable file"""

        with open(file_path, mode="w", encoding="utf-8") as file:
            json.dump(self.model_dump(), file, indent=2, allow_nan=False)

    def to_markdown_file(self, file_path: Path) -> None:
        """Return human readable conversation as markdown."""

        with open(file_path, mode="w", encoding="utf-8") as file:
            for msg in self.messages:
                file.write(f"#### {msg.__class__.__name__} \n")

                if isinstance(msg, AIMessageMetadata):
                    file.write(f"*Model:* {msg.model}\n\n")
                    file.write(f"{msg.content}\n\n")
                    if len(msg.tool_calls) > 0:
                        file.write("*Tool calls*\n")
                        file.write(f"```json\n{json.dumps(msg.tool_calls, indent=4)}\n```\n")
                elif isinstance(msg, ToolMessage):
                    file.write(f"*Tool name:* `{msg.name}`\n\n")
                    file.write(f"Result:\n{msg.content}\n")
                else:
                    file.write(f"{msg.content}\n\n")

                file.write("\n---\n\n")
