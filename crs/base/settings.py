"""
Settings for CRS agents.











If you're just a READ-ONLY visitor, this bot doesn't apply to you.
Yet you're invited to do a quick check if these settings align with the readme :)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!                       NOTE!                         !!
!!           The names you see/ define                 !!
!!       are NOT the names of the ENV-Variables!       !!
!! These have a CRS_ as prefix to their defined *name* !!
!!  Name is the first parameter. It will be converted  !!
!!     to upper and replace dashes with underscores    !!
!!        This is also documented in the readme.       !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!                   WARNING! STOP!                    !!
!!       Think twice before changing defaults!         !!
!! Other people might actively rely on these defaults  !!
!!     staying consistent as soon as they hit main!    !!
!!      When adding settings or changing settings,     !!
!!        make sure to document it in the README!      !!
!!  Also keep the CRS_ prefix in mind, mentioned above !!
!!                   WARNING! STOP!                    !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

And yes these boxes are completely typed by hand.
...if you really thought about it, feel free continue.

"""

# note: there are more settings defined in various modules
# e.g. in crscommon/logging/settings.py

from pathlib import Path

from crscommon.settings import setting
from crscommon.settings.settings import Setting

IGNORE_INTERNAL_ONLY = setting(
    "ignore-internal-only",
    bool,
    description="Ignore the .internal_only folder of the challenge project (e.g. private tests)",
    default=False,
)

OLLAMA_SERVER = setting(
    "ollama-server",
    str,
    description="Ollama server managing local LLMs",
    default="http://localhost:11434",
    cli_option="--ollama-server",
)
VLLM_SERVER = setting(
    "vllm-server",
    str,
    description="vLLM server providing a single local LLM",
    default="http://localhost:8000",
    cli_option="--vllm-server",
)
OPENAI_API_KEY: Setting[str | None] = setting(
    "openai-api-key",
    str | None,
    description="OpenAI API key",
    cli_option="--openai-api-key",
    env_name="OPENAI_API_KEY",
)
ANTHROPIC_API_KEY: Setting[str | None] = setting(
    "anthropic-api-key",
    str | None,
    description="Anthropic API key",
    cli_option="--anthropic-api-key",
    env_name="ANTHROPIC_API_KEY",
)

AI_MODEL_NAME = setting(
    "ai-model-name",
    str,
    description="LiteLLM model name to use for AI agents",
    cli_option="--ai-model",
)
AI_MODEL_NAME_ANALYZER = setting(
    "ai-model-name-analyzer",
    str,
    description="LiteLLM model name to use for the analyzer agent",
    parent=AI_MODEL_NAME,
    cli_option="--ai-model-analyzer",
)
AI_MODEL_NAME_POV = setting(
    "ai-model-name-pov",
    str,
    description="LiteLLM model name to use for the PoV builder agent",
    parent=AI_MODEL_NAME,
    cli_option="--ai-model-pov",
)

AI_MODEL_NAME_PATCHER = setting(
    "ai-model-name-patcher",
    str,
    description="LiteLLM model name to use for the patcher agent",
    parent=AI_MODEL_NAME,
    cli_option="--ai-model-patcher",
)
AI_MODEL_TEMP: Setting[float | None] = setting(
    "ai-model-temp",
    float | None,
    description="LLM temperature parameter to use for AI agents",
    cli_option="--ai-model-temp",
)
AI_MODEL_TOP_P: Setting[float | None] = setting(
    "ai-model-top-p",
    float | None,
    description="LLM top_p parameter to use for AI agents",
    cli_option="--ai-model-top-p",
)

AI_MODEL_TOOL_CHOICE = setting(
    "ai-model-tool-choice",
    str,
    description="Tool choice parameter for LLM API. Can be 'auto' or 'required'",
    cli_option="--ai-model-tool-choice",
    default="auto",
)

RECURSION_ANALYZER = setting(
    "recursion-analyzer",
    int,
    description="Agent recursion parameter to set maximum iterations of agent loop for Analyzer agent",
    default=50,
    cli_option="--recursion-analyzer",
)
RECURSION_POV = setting(
    "recursion-pov",
    int,
    description="Agent recursion parameter to set maximum iterations of agent loop for PoV agent",
    default=50,
    cli_option="--recursion-pov",
)

RECURSION_PATCHER = setting(
    "recursion-patcher",
    int,
    description="Agent recursion parameter to set maximum iterations of agent loop for Patcher agent",
    default=50,
    cli_option="--recursion-patcher",
)
AI_MODEL_RETRIES_ON_INVALID_TOOL_CALL = setting(
    "ai-model-retries-on-invalid-tool-call",
    int,
    description="Tries to get a valid tool call from LLM when returned Toolcall is invalid.",
    default=3,
    cli_option="--ai-model-retries-on-invalid-tool-call",
)

MOCK_CHAT: Setting[Path | None] = setting(
    "mock-chat",
    Path | None,
    default=None,
    description="With MockLLM, replay AI messages from this JSON chat recording",
    cli_option="--mock-chat",
)

ANALYZER_CHECK_SANITY = setting(
    "analyzer-check-sanity",
    bool,
    description="Refuse analyzing commits that do not change source files",
    default=True,
    cli_option="--analyzer-check-sanity",
)

POV_SKIP_BUILD = setting(
    "pov-skip-build",
    bool,
    description="PoV builder: Skip building the project (assuming that it is already built)",
    default=False,
    cli_option="--pov-skip-build",
)
CHECK_POV_AT: Setting[str | None] = setting(
    "check-pov-at",
    str | None,
    description="PoV builder: Use this reference instead of HEAD to check whether a PoV crashes",
    cli_option="--check-pov-at",
)
POV_MAX_LEN_BYTES = setting(
    "pov-max-len-bytes",
    int,
    description="The maximum length of a PoV in bytes",
    default=2 * 1024**2,
    cli_option="--pov-max-len-bytes",
)
POV_REPR_MAX_LEN = setting(
    "pov-repr-max-len",
    int,
    description="The max length a PoV repr may have before modifying the representation the LLM gets to see",
    default=1000,
    cli_option="--pov-repr-max-len",
)
POV_REPR_CUT_ON_MAX_LEN_TO = setting(
    "pov-repr-cut-on-max-len",
    int,
    description="The approximate length to cut the PoV to when max len is exceeded (should be less than max_len/2)",
    default=200,
    cli_option="--pov-repr-cut-on-max-len-to",
)
POV_STANDALONE_MODE = setting(
    "pov-standalone-mode",
    bool,
    description="If true, the PoV Builder will use a prompt not including any inputs from an analyzer run",
    default=False,
    cli_option="--pov-standalone-mode",
)
POV_USE_COVERAGE = setting(
    "pov-use-coverage",
    bool,
    description="Whether to use coverage for PoV generation",
    default=False,
    cli_option="--pov-use-coverage",
)

POV_COV_INCLUDE_SEARCHES = setting(
    "pov-cov-include-searches",
    bool,
    description="Whether to include searches in PoV generation",
    default=True,
    cli_option="--pov-include-searches",
)

OSS_FUZZ_LOCATION = setting(
    "oss-fuzz-location",
    Path,
    default=Path(__file__).resolve().parent.parent.parent / "oss_fuzz_integration" / "oss-fuzz",
    description="The path of the official OSS Fuzz repository.",
    cli_option="--oss-fuzz-location",
)

POV_TARGET_FUNCTION: Setting[str | None] = setting(
    "pov-target-function",
    str | None,
    description="Target function to analyze for call tree analysis. If set, call tree analysis will be performed.",
    default=None,
    cli_option="--pov-target-function",
)

CALL_TREE_ANALYSIS_DIR: Setting[Path] = setting(
    "call-tree-analysis-dir",
    Path,
    description="Directory to store call tree analysis results in.",
    default=(Path(__file__).resolve().parent.parent.parent / "oss_fuzz_integration" / "introspector-artifacts"),
    cli_option="--call-tree-analysis-dir",
)

INTROSPECTOR_ARTIFACTS_DIR: Setting[Path] = setting(
    "introspector-artifacts",
    Path,
    description="Directory to store introspector artifacts in. By default, the same as call-tree-analysis-dir.",
    parent=CALL_TREE_ANALYSIS_DIR,
    cli_option="--introspector-artifacts",
)

POV_USE_CALL_TREE_PROMPTING = setting(
    "pov-use-call-tree-prompting",
    bool,
    description="Whether to use call tree analysis for PoV generation",
    default=False,
    cli_option="--pov-use-call-tree-prompting",
)

COV_PILOT_MODE = setting(
    "cov-pilot-mode",
    bool,
    description="Activates CovPilot mode (mode aimed at 'breaking' fuzz blockers)",
    default=False,
    cli_option="--covpilot",
)

TARGET_FUNCTION: Setting[str | None] = setting(
    "target-function",
    str | None,
    description="The function to be analyzed by the analyzer/verifier agent.",
    default=None,
    cli_option="--target-function",
)

FUNCTION_MODE = setting(
    "function-mode",
    bool,
    description="If true, the analyzer/verifier agent will focus on a specific function.",
    default=False,
    cli_option="--function-mode",
)

MIXED_MODE = setting(
    "mixed-mode",
    bool,
    description="If true, the functions to analyze will be extracted from the commit.",
    default=False,
    cli_option="--mixed-mode",
)
