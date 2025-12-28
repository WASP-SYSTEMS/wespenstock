# Evaluation Tools

This repository provides two primary tools for evaluation:

* **`eval_crs`** – Launches an evaluation run.
* **`inspect_eval`** – Inspects and displays the results of an evaluation run.

---

## `eval_crs`

This tool initiates an evaluation run on a challenge project using a specified LLM.

### Required Arguments

* `--cp-path`
  Path to the challenge project. The directory will be accessed as read-only and copied for each iteration.

* `--json-file`
  JSON file containing the commits to analyze. The search order for the file is:

  1. Absolute path
  2. Relative to the current working directory
  3. Inside `eval/templates/`

* `--model`
  The LLM model to use.

### Templates
You can specify a template as the only positional argument. For example:

```bash
eval_crs nginx-qwen-quick -j 3
```

This command uses the `nginx-qwen-quick` template to perform a quick test run. The template evaluates three commits that have been automatically selected based on high likelihood of triggering relevant behavior. The `-j 3` flag runs the evaluations in parallel for faster execution.

Templates are a convenient way to run predefined evaluations without manually specifying all parameters.

### Optional Arguments

* `--temp`
  Model temperature (e.g., `--temp 0.7`).

* `--dry`
  Dry run mode. No actual evaluation will be performed.

* `-j`
  Number of parallel jobs to run.

For a full list of options, run:

```bash
eval_crs --help
```

---

## `inspect_eval`

This tool processes and displays the results of a completed evaluation run.

### Usage

```bash
inspect_eval <path-to-results>
```

Where `<path-to-results>` is the directory containing the results from `eval_crs`.

### Optional Arguments

* `--last`
  Fetches your last result directory from `/data/CRSLOG` and displays the results.

---
## setup_aixcc_nginx.py
This script is just a quick way to create a fresh copy of the AIxCC nginx challenge project in a location of your choice.
The CP is then ready to use and already includes a compile_commands.json (in the directory work).
The script requires a specific yq version that can be installed into your virtual environment via the following command (run in the parent directory of your venv):
```
curl -L https://github.com/mikefarah/yq/releases/download/v4.44.1/yq_linux_amd64 -o venv/bin/yq && chmod +x venv/bin/yq
```
---
## graphs.py
Use this script to compare multiple test series of the CRS in regards to tokens, tool calls, PoVs, LLM invocations and errors.
It creates colorful barplots for quick visual interpretation.

It currently only supports the Analyzer and the Verifier agent and will need updates if tool names change or new frequent error types occur. The current tools and known errors can be found at the top of the file.

You might need to update your venv to run this script:
```
pip install -e .[dev]
```

---
## webview.py
For an even better overview of test series results, run the webviewer. It displays:

- All barplots for the given test series.
- A results table for AIxCC Nginx test series, showing how many (successful) samples per CPV were achieved.

This allows you to quickly compare test results across multiple series.

### Usage

First, choose a port and connect to fuzzy with port forwarding:
```
ssh -L [port]:localhost:[port] fuzzy
```

Then run the webviewer

```
usage: webview.py [-h] [--non-nginx] [--plots-already-built]
                  output_path input_dirs [input_dirs ...]

Locally hosted website for pretty evaluation visualization. *happy REDACTED noise*

positional arguments:
  output_path           Path to save barplots to
  input_dirs            Path to test series.

options:
  -h, --help            show this help message and exit
  --non-nginx           At least one of the provided test series is not based in the AIxCC
                        Nginx CP
  --plots-already-built
                        Barplots to be displayed are already build and inside the
                        output_path
```
and visit http://localhost:[port]/.
