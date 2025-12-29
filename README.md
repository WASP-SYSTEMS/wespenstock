# Wespenstock

The WASPS Cyber Reasoning System — AI-driven bug discovery and fixing since 2024!

## Running

> [!CAUTION]
> When running the verifier, the LLM-generated code will be executed on your system! This may have catastrophic side effects.
> Pull requests welcome.

The challenge project (CP) to operate upon is specified using command-line arguments.
Various other [configuration knobs](#settings) can be adjusted using environment variables and CLI options.

**TL;DR:** Create a [venv](#venv), set environment variables as explained [below](#environment), and run:

```
run_crs (analyze|pov|patch) <CP-PATH> --src-path <PATH-IN-CP-src> --commit-hash <HASH> --state-file <NAME>.json
```

`--src-path` is optional if there is only one source subrepository in the CP.
`--commit-hash` defaults to the latest commit in `--src-path`
`--harness-id` is the challenge project harness to execute.
`--state-file` provides required previous-step information for `pov` and `patch`.
`--load-state` can be used instead of `--state-file` to avoid overwriting the state file.

### venv

Working on the project requires a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e '.[dev]'
# run_crs as above
```

### Minimal execution example
To run the CRS against 3 vulnerable nginx functions, you could use the CRS evaluation script as follows:
```
export ANTHROPIC_API_KEY=<YOUR API KEY>
eval_crs -f --model claude-sonnet-4-20250514 --json nginx-top3.json
```
You can use any model defined in `crs/base/model_list.py`. However, make sure that all necessary exports are set.

### CLI

Once the CRS is installed into a venv, invoke it using the `run_crs` command with that venv active.

The first CLI argument is the mode of operation. It can be one of:

- `analyze`: Run the analyzer agent only.
- `pov`: Run the PoV builder agent only.
- `patch`: Run the patching agent only.

The following options can/should/must be passed:

- `<CP-PATH>`: The challenge project's root directory (as a positional argument).

- `--src-path <PATH-IN-CP-src>`: A source code subrepository inside `<CP-PATH>/src`.
  If there is more than one subrepository, this is mandatory.

- `--commit-hash <HASH>`: A commit in the subrepository to investigate.
  If it is not specified, the latest commit is used as a default.
  This may or may not be useful.

- `--harness-id <NAME>`: A challenge project harness to concentrate upon.
  If not specified, the CRS will pick a harness in an arbitrary but deterministic manner.

- `--state-file <NAME>.json`: A JSON file that stores the result of the analyzer / PoV builder for consumption by
  later CRS steps.
  Once a step is done, it adds its result to this file.
  For the single-agent modes of operation, this and/or `--load-file` must be given.

- `--load-state <NAME>.json`: If given, the previous-run state is read from this file instead of `--state-file`.
  This allows easily restarting an agent using the same state multiple times.

### Choosing a project to test
You can run the CRS either against the AIxCC nginx challenge (use `eval/setup_aixcc_nginx.py` to prepare it) or against any OSS-Fuzz project written in C,C++ or Python (use `oss_fuzz_integration/prepare_project.py`).

### Settings

Settings can be listed with  `run_crs --help`.
#### LLM backends

Depending on the LLMs you are using, you must/should set one or more of the following variables:

- `OPENAI_API_KEY`: API key for OpenAI.
  Required if OpenAI LLMs are used.

- `ANTHROPIC_API_KEY`: API key for Anthropic.
  Required if Anthropic LLMs are used.

- `CRS_OLLAMA_SERVER`: Ollama server managing local LLMs.
  Required if local LLMs using Ollama are used.
  Defaults to `http://localhost:11434`.
  CLI option: `--ollama-server`.

- `CRS_VLLM_SERVER`: vLLM server managing a local LLM.
  Required if local LLMs using vLLM are used.
  Defaults to `http://localhost:8080`.
  CLI option: `--vllm-server`.

#### General settings

- `CRS_OUTPUT_DIR`: Filesystem directory to store the CRS' outputs. Like the reproducer, agent logs, logs, coverage data etc.
  Defaults to `.` (the current working directory).
  CLI option: `--output-dir`.

- `CRS_AI_MODEL_NAME`: Name of LLM to use in AI agents, as defined in `crs/base/model_list.py`.
  Defaults to `gpt-4o-2024-05-13`.
  CLI option: `--ai-model`.


## Structure

This CRS consists of three agents:

- `AnalyzerAgent`: Analyzes one commit or function and outputs a description of the vulnerability if it finds one.
- `PovBuilderAgent`: Constructs a _verified_ reproducer based on the description from the analyzer.
- `PatcherAgent`: Produces a patch.

All agents get the `CrsContext` as input and must output the updated context so that the next agent can proceed.

## Development

To prepare, run this inside the [venv](#venv):

```
pip install -e '.[dev]'
```

### Pre-commit

You can install the pre-commit hooks using `pre-commit install` in the base directory of the project.
This will run code formatting and other things before you commit your file, ensuring a base quality of your code before committing.

```sh
# install pre-commit hooks that run before each commit
pre-commit install

# run pre-commit checks manually (on staged files)
pre-commit run

# run pre-commit checks on all files
pre-commit run --all
```

The recommended VSCode PyLint and MyPy extensions can be found in `.vscode/extensions.json`.

### Source structure

`crs/agents/`: Directory for implementations of the agents. All code for the agent and its specific tools are located in the corresponding subfolder.

`crs/base/`: Here lies the agent base class, the context definition and some type definitions.

All LangGraph agent nodes should be listed in `base.types.NodeNames` as shown below:

```python
AnalyzerAgent: str = AnalyzerAgent.__name__
PovBuilderAgent: str = PovBuilderAgent.__name__
PatcherAgent: str = PatcherAgent.__name__
```

Thereby, we make sure the naming of the nodes is uniform and there are no typos in the names. Furthermore, we have a neat overview over all nodes in our agents.

### Defining settings

To declare a new tunable parameter:
1. Create a `settings.py` file in a suitable package if it does not exist yet.
   It is suggested to put the file adjacent to its users; e.g., the module `crs.base.chat_agent_base` uses settings declared in `crs.base.settings`.
2. Call `crscommon.settings.setting()` in that file at module scope, supplying the setting's name, type, and description.
3. Store the call's return value in a module-level variable.
4. Where you need the setting, import the setting's object from the `settings.py` file and call `get()` on it.
5. At runtime, you can define a correspondingly-named environment variable (e.g., the setting `pet-the-kittens` maps to `CRS_PET_THE_KITTENS`) with the setting's value.

See `crs.base.settings` and its uses in `crs.base.chat_agent_base` for examples.
See the `pydoc crscommon.settings` for more details.
