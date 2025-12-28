"""Baseclass for all agents managed by CRS."""

from __future__ import annotations

import time
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any
from typing import Generic
from typing import Iterator
from typing import Literal
from typing import Sequence
from typing import TypeVar

import litellm
import openai
from git import Repo
from langchain_community.chat_models import ChatLiteLLMRouter
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as create_tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.state import StateGraph
from langgraph.prebuilt.tool_node import InjectedState
from pydantic import BaseModel
from pydantic import ValidationError

from crs.agents.constants import DISPOSITION_TASK_DONE
from crs.agents.tools.lsp_tools import LSPContext
from crs.agents.tools.named_base_tool import NamedBaseTool
from crs.aixcc.env import SubprojectCommit
from crs.base.base_state import BaseChatState
from crs.base.context import CrsContext
from crs.base.conversation_metadata import ConversationMetadata
from crs.base.exceptions import ContextWindowExceededError
from crs.base.exceptions import ToolCallNotAnsweredError
from crs.base.exceptions import TooManyInvalidToolCalls
from crs.base.model_list import get_available_models
from crs.base.settings import AI_MODEL_NAME
from crs.base.settings import AI_MODEL_RETRIES_ON_INVALID_TOOL_CALL
from crs.base.settings import AI_MODEL_TOOL_CHOICE
from crs.base.settings import MOCK_CHAT
from crs.base.util import to_instance_or_none
from crs.logger import CRS_LOGGER
from crscommon.logging.logging_provider import LOGGING_PROVIDER
from crscommon.logging.settings import OUTPUT_DIR
from crscommon.mock_llm.mock_llm import MockLLM
from crscommon.settings import Setting

log = CRS_LOGGER.getChild(__name__)

Context = TypeVar("Context")


class BaseAgent(Generic[Context], ABC):
    """
    Baseclass for all agents.
    """

    state_type: type[BaseChatState] = BaseChatState

    NAME = "BaseAgent"

    def __init__(self, ctx: CrsContext):
        """init"""
        self.ctx = ctx
        self.lsp_ctx: LSPContext
        self.target_code: str  # code or git diff which is examined by the agent

        self.graph: CompiledStateGraph  # call build_and_compile_default_graph to populate

        self.state: BaseChatState
        self.tools: list[NamedBaseTool]

    @abstractmethod
    def run(self) -> Context:
        """
        Run the agent.
        """

    @abstractmethod
    def _get_initial_messages(self) -> list[BaseMessage]:
        """The messages the LLm shall initially be prompted with."""
        raise NotImplementedError("Please define how the initial messages shall look for you agent.")

    def get_base_chat_state(self) -> BaseChatState:
        """Get the state on which the LLM is initialized."""
        return self.state_type.from_messages(self._get_initial_messages())

    def init_lsp(self, checkout_repo_at: None | SubprojectCommit | Repo) -> tuple[LSPContext, str]:
        """
        Inits the LSP and returns a newly created LSP context and the target code to be examined.
        The default implementation returns the git diff of the last commit as target.
        """
        lsp_ctx = LSPContext(self.ctx)
        lsp_ctx.add_known_symbols()

        if checkout_repo_at and isinstance(checkout_repo_at, SubprojectCommit):
            repo = self.ctx.comp_env.checkout(checkout_repo_at)

        elif checkout_repo_at and isinstance(checkout_repo_at, Repo):
            repo = checkout_repo_at

        # this case shouldn't happen, but we need to handle it since the type used further up the chain allowed for None
        elif checkout_repo_at is None:
            raise ValueError(f"Cannot initialize on None repo: {checkout_repo_at=}")

        else:
            # well. just assign it and see if it works...
            repo = checkout_repo_at  # type: ignore
            log.warning(
                f"Received type '{type(checkout_repo_at)}' as input for {checkout_repo_at=}. "
                f"Proceeding with it. Let's see if it'll fail :sparkles:"
            )

        diff = lsp_ctx.add_changed_symbols(repo)

        return lsp_ctx, diff

    def exec_graph(
        self,
        config: RunnableConfig | None = None,
    ) -> BaseMessage:
        """
        Run the compiled graph and reports output to stout / JSON / Markdown as configured.

        Args:
            config: config dict that will be passed to compiled_graph.stream()

        Returns:
            The last AIMessage which should contain the answer to our problem
        """

        log.info(f"Executing graph for agent '{self.NAME}'")

        seen_message_count = 0

        for output in self.stream_graph(config or RunnableConfig()):
            self.state = BaseChatState.model_validate(output)

            for msg in self.state.messages[seen_message_count:]:
                log.info(f"New {type(msg)=}")
                msg.pretty_print()
                seen_message_count = len(self.state.messages)

            self._write_logs()

        last_message: BaseMessage = self.state.messages[-1]

        log.info(f"Finished executing graph for agent '{self.NAME}'")
        return last_message

    def stream_graph(self, config: RunnableConfig) -> Iterator[dict[str, Any] | Any]:
        """Generate wrapper for CompiledStateGraph.stream to catch validation errors."""
        for _ in range(AI_MODEL_RETRIES_ON_INVALID_TOOL_CALL.get()):
            try:
                # yield from ends with a StopIteration -> we implicitly catch that with this syntax
                return (yield from self.graph.stream(self.state, config, stream_mode="values"))
            except ValidationError:
                log.warning("Failed to validate message")

        raise TooManyInvalidToolCalls()

    def build_and_compile_default_graph(self, ai_model_name: Setting[str]) -> None:
        """Initialize default agent graph."""

        def from_call_model_to(state: BaseChatState) -> Literal["run_tool", "motivate", "__end__"]:
            """decide if we call a tool or end the loop"""
            last_message = state.messages[-1]
            if isinstance(last_message, AIMessage):
                if last_message.tool_calls:
                    return "run_tool"
                if MOTIVATOR_NODE.matches_give_up(last_message):
                    return "__end__"
                return "motivate"
            return "__end__"

        def from_run_tool_to(state: BaseChatState) -> Literal["call_model", "__end__"]:
            """decide if this tool concludes the loop"""
            for tool_msg in state.latest_tool_messages:
                if tool_message_disposition(tool_msg) == DISPOSITION_TASK_DONE:
                    return "__end__"
            return "call_model"

        call_model = CallModelNode(self.tools, model_name=ai_model_name.get())
        run_tool = SequentialToolNode(self.tools)

        # define graph
        graph = StateGraph(BaseChatState)

        graph.add_node("call_model", call_model)
        graph.add_node("run_tool", run_tool)
        graph.add_node("motivate", MOTIVATOR_NODE)

        graph.add_conditional_edges("call_model", from_call_model_to)
        graph.add_conditional_edges("run_tool", from_run_tool_to)
        graph.add_edge("motivate", "call_model")

        graph.set_entry_point("call_model")

        self.graph = graph.compile()

    def _write_logs(self) -> None:
        """Write message logs to log direcytory."""

        metadata = ConversationMetadata.from_state(self.state)

        markdown_file = Path(OUTPUT_DIR.get()) / f"{self.NAME}_{LOGGING_PROVIDER.file_name_tag}.md"
        json_file = Path(OUTPUT_DIR.get()) / f"{self.NAME}_{LOGGING_PROVIDER.file_name_tag}.json"
        state_dump_json_file = Path(OUTPUT_DIR.get()) / f"{self.NAME}_state_dump_{LOGGING_PROVIDER.file_name_tag}.json"

        metadata.to_json_file(json_file)
        metadata.to_markdown_file(markdown_file)

        state_dump_json_file.write_text(
            self.state.model_dump_json(indent=2)
        )  # other dump for successful/failed tool call parsing


# anthropic does not support logprobs, so we allow litellm to drop unsupport params
litellm.drop_params = True


class MotivatorNode:
    """A graph node that tries to motivate the LLM to do more tool calls."""

    def __init__(self) -> None:
        self.motivate_msg = (
            "You have not performed any tool calls. To make progress, you must call tools. "
            "If you really want to abandon your task, reply with 'I GIVE UP'."
        )
        self.confirm_msg = (
            "Your message contains the text 'I GIVE UP', but does not solely consist of it. "
            "If you are sure you want to abandon your task, reply only and exactly with 'I GIVE UP'."
        )
        self.error_msg = "There has been an internal error. Please try doing something in a different manner."
        self.give_up_string = "I GIVE UP"

    def contains_give_up(self, msg: BaseMessage) -> bool:
        """Returns whether the given message could indicate abandonment but not with certainty."""
        return isinstance(msg, AIMessage) and isinstance(msg.content, str) and self.give_up_string in msg.content

    def matches_give_up(self, msg: BaseMessage) -> bool:
        """Returns whether the given message matches this node's give-up string."""
        if not isinstance(msg, AIMessage) or not isinstance(msg.content, str):
            return False
        return msg.content.strip() in (self.give_up_string, self.give_up_string + ".")

    def __call__(self, state: BaseChatState) -> BaseChatState:
        """Appends this node's response to the chat state."""

        last_message = state.messages[-1]
        if not isinstance(last_message, AIMessage) or self.matches_give_up(last_message):
            # The router should not have activated this node in the first place.
            reply = self.error_msg
        elif self.contains_give_up(last_message):
            reply = self.confirm_msg
        else:
            reply = self.motivate_msg

        state.messages.append(HumanMessage(content=reply))
        return state


MOTIVATOR_NODE = MotivatorNode()


class SequentialToolNode:
    """
    Executes tools in the same order as they are received from the LLM.

    The langgraph implementation runs all tools in parallel.
    """

    def __init__(self, tools: Sequence[BaseTool]) -> None:
        self.tools = [tool if isinstance(tool, BaseTool) else create_tool(tool) for tool in tools]
        self.tool_map = {t.name: t for t in self.tools}

        self.invalid_tool_calls = 0

        if len(self.tools) != len(self.tool_map):
            raise ValueError(f"Received at least 2 tools with the same name: {[t.name for t in self.tools]}")

    def __call__(self, state: BaseChatState) -> BaseChatState:
        last_message = state.messages[-1]

        if not isinstance(last_message, AIMessage):
            raise TypeError(f"Cannot call tools on message of type {last_message.__class__.__name__}")

        log.info(f"Processing {len(last_message.tool_calls)} tool calls...")

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]
            invocation_string = f"{tool_name}({', '.join(f'{k}={v!r}' for k, v in tool_args.items())})"
            log.info(f"Executing tool {invocation_string} [{tool_call_id=}]")

            try:
                tool = self.tool_map[tool_name]
            except KeyError as e:
                log.warning(
                    f"The tool '{tool_name}' could not be found in the registered tools {self.tool_map.keys()}. "
                    "This can happen if the existence of a tool was mentioned in a prompt but not registered.",
                )

                self.invalid_tool_calls += 1

                if self.invalid_tool_calls > AI_MODEL_RETRIES_ON_INVALID_TOOL_CALL.get():
                    raise TooManyInvalidToolCalls() from e

                # return unaltered state and try again
                return state

            tool_args = self._inject_extra_args(tool.get_input_schema(), tool_args, state)
            output = tool.invoke(tool_args)

            log.info(f"Finished executing tool {invocation_string} [{tool_call_id=}]")

            self.invalid_tool_calls = 0

            content: str
            artifact: Any
            if isinstance(output, dict):
                content = str(output["content"])
                artifact = output
            else:
                content = artifact = str(output)

            state.messages.append(
                ToolMessage(
                    content=content,
                    artifact=artifact,
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )

        return state

    @staticmethod
    def _inject_extra_args(
        tool_schema: type[BaseModel], tool_args: dict[str, Any], state: BaseChatState
    ) -> dict[str, Any]:
        """
        Inject runtime dependencies (like state) into the arguments of a tool call.

        This is useful to pass context that is not present in a tool but is required for certain operations.
        Langgraph has this feature built-in (when using decorators), but we don't use decorators for tools.
        https://langchain-ai.github.io/langgraph/reference/agents/#langgraph.prebuilt.tool_node.InjectedState

        Currently, processes the InjectedState annotation.
        """
        found_injects: dict[tuple[str, int], InjectedState] = {}
        for name, info in tool_schema.model_fields.items():
            for i, annotation in enumerate(info.metadata):
                if inject := to_instance_or_none(annotation, InjectedState):
                    found_injects[name, i] = inject

        # Mutating tool_args in-place causes issues with serializing the chat state later, so we copy tool_args
        # below (but only if necessary).
        result = tool_args

        if found_injects:
            if len(found_injects) > 1:
                raise TypeError(
                    f"There may be at most one InjectedState argument in a tool's schema, please fix {tool_schema}"
                )
            (name, _), inject = found_injects.popitem()
            if inject.field:
                raise TypeError("InjectedState(field=...) is not implemented, sorry")
            result = {**result, name: state}

        return result


def tool_message_disposition(msg: ToolMessage) -> str | None:
    """
    Return the tool message's disposition, if any.

    The tool message must have an artifact, that artifact must be a dict, and its "disposition" entry must
    exist and be a string.

    When using the ToolExecutorNode, this can be achieved by letting the tool's _run() method return a dictionary
    instead of a bare string.
    """
    if not isinstance(msg.artifact, dict):
        return None
    result = msg.artifact.get("disposition")
    if not isinstance(result, str):
        return None
    return result


def get_model_with_tools(
    model: str,
    tools: Sequence[BaseTool],
) -> Runnable[LanguageModelInput, BaseMessage]:
    """Get llm model with tools bound to it."""

    litellm_router = litellm.Router(
        model_list=get_available_models(), num_retries=3, timeout=1800, default_max_parallel_requests=10000
    )

    # if ollama is used, this makes sure the model is loaded
    # TODO: make this work with self build models
    # pull_model(model)

    chat = ChatLiteLLMRouter(
        router=litellm_router,
        model_name=model,
    ).bind_tools(tools)

    # For gpt-5 family, do not set the logprobs parameter at all.
    if not model.startswith("gpt-5"):
        chat = chat.bind(logprobs=True)
    # else: leave chat unbound for logprobs so the router/LiteLLM can decide defaults

    return chat


class CallModelNode:
    """
    Node which calls the llm on the current state.
    """

    def __init__(
        self,
        tools: Sequence[BaseTool],
        model_name: str | None = None,
    ) -> None:

        if model_name is None:
            model_name = AI_MODEL_NAME.get()

        self.tools = tools
        self.model_name = model_name
        # mock llm is for testing purposes it doesn't need any of these parameters
        # and it just a replay of a given agent sample
        self.model: Runnable[LanguageModelInput, BaseMessage] | MockLLM
        if model_name == "MockLLM":
            # it's a debug measure, so I'm fine with ignoring the type in mypy
            self.model = MockLLM.from_json_file(MOCK_CHAT.get())  # type: ignore
        # this is the default branch
        else:
            self.model = get_model_with_tools(model_name, tools)

    def __call__(self, state: BaseChatState) -> BaseChatState:
        """Call model."""
        # pylint: disable=R0912
        # (we branch as much as we want here. because there are just many cases to cover)

        # TODO: that for-loop decide if there are more "savable" cases where we also can try again instead of raising
        invalid_tool_call_messages: int = 0
        for _ in range(0, 10000):
            try:
                log.info(f"Invoking {self.model_name} with {len(state.messages)} messages")
                response = self.model.invoke(
                    state.messages,
                    temperature=state.temperature,
                    top_p=state.top_p,
                    tool_choice=AI_MODEL_TOOL_CHOICE.get(),
                )

                # LLMs can produce invalid tool calls in rare occasions, if we don't catch this the agent dies
                if len(invalid_tcs := response.lc_attributes["invalid_tool_calls"]) > 0:
                    # if we would not safeguard against this, we could potentially run in an "infinite" loop
                    # burning cash without noticing
                    if invalid_tool_call_messages > AI_MODEL_RETRIES_ON_INVALID_TOOL_CALL.get():
                        raise TooManyInvalidToolCalls(
                            f"Got {invalid_tool_call_messages} LLM responses that contained invalid tool calls"
                        )

                    # TODO: maybe append a Human Message telling the LLM that it did a faulty tool call?
                    log.warning(
                        f"Got invalid tool call(s) from LLM- Not appending message to state. "
                        f"Trying again. The invalid tool calls dict: {invalid_tcs}"
                    )
                    invalid_tool_call_messages += 1
                    continue

                # okay we're good.
                state.messages.append(response)
                log.info(f"{self.model_name} successfully answered")
                break
            except openai.RateLimitError as e:
                if e.code == 429:
                    # ignore "No deployments available for selected model"
                    log.warning('Ignoring LiteLLM API error "No deployments available for selected model"')
                    time.sleep(10)  # backoff time
                else:
                    raise
            except openai.AuthenticationError as e:
                if e.code == 401:
                    timeout = 20

                    log.error("Max API budget exceeded")
                    log.info(f"Going to sleep for {timeout} seconds")

                    # we do this to keep the logs clean, because container will restart
                    # forever and logs will be polluted with this error
                    time.sleep(timeout)
                else:
                    raise

            except openai.BadRequestError as e:
                # these Exceptions contain a lot of text and aren't easily parsable, so a basic check must be enough
                # in general: this is just for better understanding what caused the error
                searchable_e = str(e)
                log.info(
                    f"Got an openai.BadRequestError (Message: {e}), "
                    f"trying to determine the specific reason before re-raising..."
                )
                # works at least for anthropic, let's hope the oai message is the same
                if "ContextWindowExceededError" in searchable_e:
                    raise ContextWindowExceededError(e) from e
                if "'tool_calls' must be followed" in searchable_e:
                    raise ToolCallNotAnsweredError(e) from e
                raise  # well. let's throw the original thing and let the user figure it out.
        else:
            raise RuntimeError("Could not call model! Too many rate limit errors!")

        return state
