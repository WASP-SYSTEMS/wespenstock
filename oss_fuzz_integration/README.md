# Using OSS-Fuzz Projects as Input for the CRS

## prepare_project.py
Choose a OSS-Fuzz project and run `prepare_project.sh <project_name>`. It creates an [AIxCC style project directory](#challenge-project-structure) `cp-[project_name]-[commit-hash | HEAD]`. Refer to **successful_projects.txt** for a list of C/C++ projects expected to build successfully.

```
prepare_project.py --help
```

- the script will check whether there is already a clone of the OSS-Fuzz repository in your output path (either the one specified by -o, or if you do not specify one, the script's directory). If this is not the case, it will create a new clone.
- If there is already content in the successful/failed list, the script will continute its evaluation with the next project that is not yet listed in one of these two files. If you want to reevaluate a project you must delete its entry in the corresponding file, or provide new empty files, or use --force
- Criteria for compatible projects:
    - uses git and does not remove .git directory
    - compiles without error
    - has at least one fuzz target in /out that can be mapped to a source file
- The script creates run_sh_log.log capturing stderr and stdout during execution of project preparation steps via run.sh in every CP for easier debugging. Additionally, the failed projects file saves a timestamp and a short error description for a quick overview.

### Sanitizers
The project.yaml must contain a list of sanitizers for our CRS to check against to validate crashes. The sanitizers are language-specific.
- c-cpp-sanitizers.yaml: All sanitizers used in public AIxCC C projects
- python-sanitizers.yaml: The string `Traceback (most recent call last):`, which is indicative of an uncaught exception

## search_for_0days.py
Use this script to start CRS's *Analyzer* and *Verifier* on the newest n commits of OSS-Fuzz C/C++ projects. Every commit is started for every available harness. You probably want to specify the **maximal number of harnesses** (via -m) so that projects with huge amounts of fuzz harnesses are skipped (e.g. ffmpeg with 991 harnesses).

- project_file should contain the list of projects you want to work on
- cp_folder must contain a CP in the format cp-project_name-HEAD for each project listed in project_file
- the script expects you to have a running venv with **pyelftools** installed
- The `output_dir` contains a folder for each project that the script has already processed or is currently working on. If you want the script to process the same project again, you must delete or rename its existing folder.
- Each project folder contains directories for every commit, following the format `[commit-id]-[timestamp]`. Within these commit directories, there is a separate directory for each available harness, where the CRS files for each sample are stored.

```
search_for_0days.py --help
```


## harness.py
Input a project and a commit to get information which harnesses can reach the changed lines of code. Information is collected via the Fuzz Introspector API
```
harness.py  --help
```

## Challenge Project Structure
Each (OSS-Fuzz) Challenge Project has the following directories and scripts:
- **run.sh**:
```
A helper script for CP interactions.

Usage:  [OPTIONS] build|run_pov|custom|make-cpsrc-prepare

OPTIONS:
  -h    Print this help menu
  -v    Turn on verbose debug messages

Subcommands:
  build [<patch_file> <source>]       Build the CP
  run_pov <blob_file> <harness_name>  Run the binary data blob against specified harness
  custom <arbitrary cmd ...>          Run an arbitrary command in the docker container
  make-cpsrc-prepare                  Prepare docker container
```
- src: contains multiple subdirectories with source code (including source code of fuzzers and dependencies). Usually, the intersting source code is in a subfolder with the same name as the project itself.
- work: intermediate artifacts, entrypoint for *run.sh custom*
- out: Mostly fuzzing binaries but sometimes also seed generators or corpora
- project.yaml: various information about the CP
- build.sh: build script
- Dockerfile
- init_CRS.sh: helper script to enable compilation database generation with bear

# Crawl OSS-Fuzz issue

## crawl_oss_issue.py

This script receives a link to an OSS-Fuzz issue and then crawls the issue page to create a database entry containing:

- `name`: Name of the project
- `oss_fuzz_issue`: Link to OSS-Fuzz issue
- `harness_name`: Name of the harness used to trigger issue
- `vuln_commit`: Vulnerable commit
- `introspector_url`: Link to introspector report

The only thing missing for a complete dataset entry as in `dataset_c.json` and `dataset_python.json` is the vulnerable function.
```
usage:
python crawl_oss_issue.py <link to OSS-Fuzz issue>
```

# Datasets

- Entries for projecy `miniz` removed because of auf code amalgamation.
- Remove entry for `libucl` udn function `kh_get_ucl_hash_node`, because the function is a lib function which is created through a macro.
- remove project `flex` because of harness bug
