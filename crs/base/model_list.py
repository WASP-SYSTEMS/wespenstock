"""
This file contains the list of models that are available for use in the CRS.
"""

# environment variables need to be set before running the CRS:
# OPENAI_API_KEY=<your-openai-api-key> (if you want to use OpenAI models)
# ANTHROPIC_API_KEY=<your-anthropic-api-key> (if you want to use Anthropic models)

from typing import Any
from typing import cast

from ollama import Client

from crs.logger import CRS_LOGGER

from .settings import ANTHROPIC_API_KEY
from .settings import OLLAMA_SERVER
from .settings import OPENAI_API_KEY
from .settings import VLLM_SERVER

log = CRS_LOGGER.getChild(__name__)


def get_available_models() -> list[dict[str, Any]]:
    """Get model list with available models for the LiteLLM router."""

    # we often switch between "ollama_chat" and "openai" API endpoints
    # -> allows us to switch between those without changing all model entries
    ollama_api_type = "openai"
    base_server = OLLAMA_SERVER.get().rstrip("/")
    api_base = f"{base_server}/v1" if ollama_api_type == "openai" else base_server

    # Common default params for ollama-hosted models
    default_ollama: dict[str, Any] = {
        "api_base": api_base,
        "api_key": "no_key",
    }

    # Add ollama-chat specific runtime options only when in ollama_chat mode
    if ollama_api_type == "ollama_chat":
        default_ollama.update(
            {
                "num_ctx": 1024 * 32,
                "keep_alive": "60m",
            }
        )

    # This is the model list with all models available for the litellm router.
    # Please use the actual name of the model (without the provider prefix) as "model_name"
    # as done for the other models in this list, when adding new models.
    model_list = [
        {
            "model_name": "gpt-5-nano-2025-08-07-flex",
            "litellm_params": {
                "model": "openai/gpt-5-nano-2025-08-07",
                "api_key": OPENAI_API_KEY.get() or "<not set>",
                "service_tier": "flex",
            },
        },
        {
            "model_name": "gpt-5-mini-2025-08-07-flex",
            "litellm_params": {
                "model": "openai/gpt-5-mini-2025-08-07",
                "api_key": OPENAI_API_KEY.get() or "<not set>",
                "service_tier": "flex",
            },
        },
        {
            "model_name": "gpt-5-2025-08-07-flex",
            "litellm_params": {
                "model": "openai/gpt-5-2025-08-07",
                "api_key": OPENAI_API_KEY.get() or "<not set>",
                "service_tier": "flex",
            },
        },
        {
            "model_name": "gpt-5-nano-2025-08-07",
            "litellm_params": {
                "model": "openai/gpt-5-nano-2025-08-07",
                "api_key": OPENAI_API_KEY.get() or "<not set>",
                "default_temperature": 1.0,
            },
        },
        {
            "model_name": "gpt-5-mini-2025-08-07",
            "litellm_params": {
                "model": "openai/gpt-5-mini-2025-08-07",
                "api_key": OPENAI_API_KEY.get() or "<not set>",
                "default_temperature": 1.0,
            },
        },
        {
            "model_name": "gpt-5-2025-08-07",
            "litellm_params": {
                "model": "openai/gpt-5-2025-08-07",
                "api_key": OPENAI_API_KEY.get() or "<not set>",
                "default_temperature": 1.0,
            },
        },
        {
            "model_name": "gpt-4.1-nano-2025-04-14",
            "litellm_params": {
                "model": "openai/gpt-4.1-nano-2025-04-14",
                "api_key": OPENAI_API_KEY.get() or "<not set>",
                "default_temperature": 0.0,
            },
        },
        {
            "model_name": "gpt-4.1-mini-2025-04-14",
            "litellm_params": {
                "model": "openai/gpt-4.1-mini-2025-04-14",
                "api_key": OPENAI_API_KEY.get() or "<not set>",
                "default_temperature": 0.0,
            },
        },
        {
            "model_name": "gpt-4.1-2025-04-14",
            "litellm_params": {
                "model": "openai/gpt-4.1-2025-04-14",
                "api_key": OPENAI_API_KEY.get() or "<not set>",
                "default_temperature": 0.0,
            },
        },
        {
            "model_name": "gpt-4o-2024-05-13",
            "litellm_params": {
                "model": "openai/gpt-4o-2024-05-13",
                "api_key": OPENAI_API_KEY.get() or "<not set>",
                "default_temperature": 0.0,
            },
        },
        {
            "model_name": "gpt-4o-2024-11-20",
            "litellm_params": {
                "model": "openai/gpt-4o-2024-11-20",
                "api_key": OPENAI_API_KEY.get() or "<not set>",
                "default_temperature": 0.0,
            },
        },
        {
            "model_name": "claude-3-5-sonnet-20241022",
            "litellm_params": {
                "model": "anthropic/claude-3-5-sonnet-20241022",
                "api_key": ANTHROPIC_API_KEY.get() or "<not set>",
                "default_temperature": 0.0,
            },
        },
        {
            "model_name": "claude-3-7-sonnet-20250219",
            "litellm_params": {
                "model": "anthropic/claude-3-7-sonnet-20250219",
                "api_key": ANTHROPIC_API_KEY.get() or "<not set>",
                "default_temperature": 0.0,
            },
        },
        {
            "model_name": "claude-sonnet-4-20250514",
            "litellm_params": {
                "model": "anthropic/claude-sonnet-4-20250514",
                "api_key": ANTHROPIC_API_KEY.get() or "<not set>",
            },
        },
        # Ollama-hosted models
        {
            "model_name": "gpt-oss_120b",
            "litellm_params": {
                **default_ollama,
                "model": f"{ollama_api_type}/gpt-oss:120b_131072",
                "default_temperature": 1.0,
                "default_top_p": 1.0,
            },
        },
        {
            "model_name": "llama3.3",
            "litellm_params": {
                **default_ollama,
                "model": f"{ollama_api_type}/llama3.3",
            },
        },
        {
            "model_name": "llama4-maverick",
            "litellm_params": {
                **default_ollama,
                "model": f"{ollama_api_type}/llama4:maverick",
            },
        },
        {
            "model_name": "deepseek-r1_671b-0528-q4_K_M",
            "litellm_params": {
                **default_ollama,
                "model": f"{ollama_api_type}/deepseek-r1:671b-0528-q4_K_M_8192",
                "default_temperature": 0.0,  # https://api-docs.deepseek.com/quick_start/parameter_settings
            },
        },
        {
            "model_name": "deepseek-v3.1_671b",
            "litellm_params": {
                **default_ollama,
                "model": f"{ollama_api_type}/deepseek-v3.1:671b_131072",
                "default_temperature": 0.0,  # https://api-docs.deepseek.com/quick_start/parameter_settings
            },
        },
        {
            "model_name": "deepseek-v3.1_openrouter",
            "litellm_params": {
                "model": "openai/deepseek/deepseek-chat-v3.1",
                "default_temperature": 0.0,
                "api_base": "https://openrouter.ai/api/v1",
                "api_key": OPENAI_API_KEY.get() or "<not set>",
            },
        },
        {
            "model_name": "qwen2.5-coder_32b-instruct-q8_0",
            "litellm_params": {
                **default_ollama,
                "model": f"{ollama_api_type}/qwen2.5-coder:32b-instruct-q8_0",
                "default_temperature": 0.7,
                "default_top_p": 1.0,
            },
        },
        {
            "model_name": "qwen3_32b",
            "litellm_params": {
                **default_ollama,
                "model": f"{ollama_api_type}/qwen3:32b_131072",
                "default_temperature": 0.7,
                "default_top_p": 1.0,
            },
        },
        {
            "model_name": "qwen3-coder_30b-a3b-q4_K_M",
            "litellm_params": {
                **default_ollama,
                "model": f"{ollama_api_type}/qwen3-coder:30b-a3b-q4_K_M",
                "default_temperature": 0.7,
                "default_top_p": 1.0,
            },
        },
        {
            "model_name": "qwen3-coder_480b",
            "litellm_params": {
                **default_ollama,
                "model": f"{ollama_api_type}/qwen3-coder:480b_131072",
                "default_temperature": 0.7,
                "default_top_p": 1.0,
            },
        },
        {
            "model_name": "mistral_7b",
            "litellm_params": {
                **default_ollama,
                "model": f"{ollama_api_type}/mistral:7b_32768",
            },
        },
        {
            "model_name": "mistral-nemo_12b",
            "litellm_params": {
                **default_ollama,
                "model": f"{ollama_api_type}/mistral-nemo:12b_131072",
            },
        },
        {
            "model_name": "mixtral_8x22b",
            "litellm_params": {
                **default_ollama,
                "model": f"{ollama_api_type}/mixtral:8x22b_65536",
            },
        },
        # Models hosted on a vLLM server
        # Note: only one model will be hosted at a time
        {
            "model_name": "qwen3_32b_fp8",
            "litellm_params": {
                "model": "hosted_vllm/Qwen/Qwen3-32B-FP8",
                "default_temperature": 0.7,
                "default_top_p": 1.0,
                "api_base": VLLM_SERVER.get().rstrip("/") + "/v1",
                "api_key": "no_key",
            },
        },
        {
            "model_name": "Qwen3-Coder-30B-A3B-Instruct-FP8",
            "litellm_params": {
                "model": "hosted_vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
                "default_temperature": 0.7,
                "default_top_p": 1.0,
                "api_base": VLLM_SERVER.get().rstrip("/") + "/v1",
                "api_key": "no_key",
            },
        },
        {
            "model_name": "gpt-oss_20b_vllm",
            "litellm_params": {
                "model": "hosted_vllm/openai/gpt-oss-20b",
                "default_temperature": 1.0,
                "default_top_p": 1.0,
                "api_base": VLLM_SERVER.get().rstrip("/") + "/v1",
                "api_key": "no_key",
            },
        },
        {
            "model_name": "gpt-oss_120b_vllm_it_sec",
            "litellm_params": {
                "model": "hosted_vllm/openai/gpt-oss-120b",
                "default_temperature": 1.0,
                "default_top_p": 1.0,
                "api_base": "https://ai.wasp.systems/v1",
                "api_key": "no_key",
            },
        },
        {
            "model_name": "gpt-oss_120b_vllm",
            "litellm_params": {
                "model": "hosted_vllm/openai/gpt-oss-120b",
                "default_temperature": 1.0,
                "default_top_p": 1.0,
                "api_base": VLLM_SERVER.get().rstrip("/") + "/v1",  # 8001 -> our custom load balancing solution
                "api_key": "no_key",
            },
        },
        # MockLLM
        {
            "model_name": "MockLLM",
            "litellm_params": {
                "model": "hosted_vllm/WASPSLLM",
                "default_temperature": 0.0,
                "api_base": VLLM_SERVER.get().rstrip("/") + "/v1",
                "api_key": "no_key",
            },
        },
    ]

    return model_list


def _get_model_params(model_name: str) -> dict[str, Any] | None:
    """
    Helper: Fetch the named model's litellm_params, if any.
    """

    model = next((model for model in get_available_models() if model["model_name"] == model_name), None)

    if model is None:
        raise ValueError(f"Model {model_name} not found in model_list")

    return cast(dict[str, Any], model.get("litellm_params"))


def get_temperature_for_model(model_name: str) -> float | None:
    """
    Get the default temperature for a model from the model list.
    """

    if (litellm_params := _get_model_params(model_name)) is None:
        return None

    return litellm_params.get("default_temperature")


def get_top_p_for_model(model_name: str) -> float | None:
    """
    Get the default top-p for a model from the model list.
    """

    if (litellm_params := _get_model_params(model_name)) is None:
        return None

    return litellm_params.get("default_top_p")


def _get_model_full_name(model_name: str) -> str:
    """
    Get the full model name from the model list.
    """

    if (litellm_params := _get_model_params(model_name)) is None:
        raise ValueError(f"Model {model_name} does not have litellm_params")

    return litellm_params["model"]


def _check_if_ollama(model_name: str) -> bool:
    """
    Check if the model is a Ollama model.
    """

    model = _get_model_full_name(model_name)

    if model.startswith("ollama/"):
        return True

    return False


def _check_if_model_exists(client: Client, model_name: str) -> bool:
    """
    Check if the Ollama server has the model.
    """

    if _get_model_full_name(model_name).split("/")[-1] in client.list():
        return True

    return False


def pull_model(model_name: str) -> None:
    """
    Checks if the model is a Ollama model.
    If it is, make sure, the model is available on the Ollama server.
    """

    client = Client(OLLAMA_SERVER.get())

    if not _check_if_ollama(model_name):
        # not an Ollama model
        return

    if _check_if_model_exists(client, model_name):
        # if model exists, no need to pull
        log.info(f"{model_name} already pulled")
        return

    log.info(f"Pulling {model_name}")

    # pull the model
    # TODO: add some kind of logging if we run into this case
    # Might take same time -> user should be informed
    client.pull(_get_model_full_name(model_name).split("/")[-1])

    log.info(f"Pulled {model_name}")
