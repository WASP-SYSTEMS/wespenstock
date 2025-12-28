#!/usr/bin/env python3
"""Script to evaluate the cyber reasoning system"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from types import FrameType
from typing import List
from typing import Optional
from typing import Tuple

from git import Repo

from crs.aixcc.env import aixcc_local_env
from eval.inspect_eval import evaluate_test_series
from eval.inspect_eval import generate_results_table
from eval.setup_aixcc_nginx import clone_and_prep

PRESETS = {
    "nginx-qwen-full": {
        "temp": 0.7,
        "json_file": "nginx-test.json",
        "model": "Qwen3-Coder-30B-A3B-Instruct-FP8",
        "r": 20,
        "d": "Testing with nginx-qwen-full",
    },
    "nginx-qwen-one": {
        "temp": 0.7,
        "json_file": "nginx-test.json",
        "model": "Qwen3-Coder-30B-A3B-Instruct-FP8",
        "r": 1,
        "d": "Testing with nginx-qwen-one",
    },
    "nginx-qwen-quick": {
        "temp": 0.7,
        "json_file": "nginx-top3.json",
        "model": "Qwen3-Coder-30B-A3B-Instruct-FP8",
        "r": 3,
        "d": "Testing with nginx-qwen-quick",
    },
    "nginx-bug": {
        "temp": 0.7,
        "json_file": "nginx-bug.json",
        "model": "Qwen3-Coder-30B-A3B-Instruct-FP8",
        "r": 5,
        "d": "Testing with nginx-qwen-quick",
    },
}

interrupted = False  # pylint: disable=invalid-name


# pylint: disable=too-many-instance-attributes
class Evaluator:
    """Class for evaluating the CRS"""

    def __init__(
        self,
        model_temp: float,
        model_name: str,
        json_file: Path,
        eval_dir: Path,
        tmp_dir: Path,
        dry_run: bool = False,
        function_mode: bool = False,
        mixed: bool = False,
    ):
        self.model_temp = model_temp
        self.model_name = model_name
        self.json_file = json_file
        self.eval_dir = eval_dir
        self.dry_run = dry_run
        self.cp_temp: Path | None = None
        self.tmp_dir: Path = tmp_dir
        self.function_mode = function_mode
        self.mixed = mixed

    def abort(self, _signum: int, _frame: Optional[FrameType]) -> None:
        """Function to handle aborts"""
        sys.exit(0)

    def setup_cp(self, commit: str) -> None:
        """copies the cp"""
        if self.dry_run:
            print("[DRY RUN] Creating project in temporary dir")
            self.cp_temp = Path("/tmp/ONLYDRYRUN")
        else:
            self.cp_temp = Path(tempfile.mkdtemp(prefix=f"nginx-{commit}-", dir=self.tmp_dir))
            clone_and_prep(self.cp_temp, build_project=False)
            comp_env = aixcc_local_env(self.cp_temp / "challenge-004-nginx-cp")
            comp_env.build()

    def create_output_directory(self, i: int, commit: str) -> Path:
        """creates the output directory for a commit run combination"""
        timestamp = datetime.now().isoformat()
        output_directory = self.eval_dir / commit / f"{i}_{timestamp}"
        if self.dry_run:
            print(f"[DRY RUN] Would create output dir: {output_directory}")
        else:
            output_directory.mkdir(parents=True, exist_ok=True)
        return output_directory

    def run_analyzer(self, output_directory: Path, target: str) -> None:
        """runs the analyzer"""
        run_crs_cmd = [
            "run_crs",
            "--output-dir",
            str(output_directory),
            "--ai-model",
            self.model_name,
            "--ai-model-temp",
            str(self.model_temp),
        ]

        if self.mixed:
            run_crs_cmd.extend(["--mixed-mode", "true"])
        elif self.function_mode:
            run_crs_cmd.extend(["--target-function", target, "--function-mode", "true"])

        run_crs_cmd.extend(
            ["analyze", str(self.cp_temp / "challenge-004-nginx-cp") if self.cp_temp else "challenge-004-nginx-cp"]
        )

        if not self.function_mode:
            run_crs_cmd.extend(["--commit-hash", target])

        run_crs_cmd.extend(["--state-file", str(output_directory / "analyze.json")])

        if self.dry_run:
            print(f"[DRY RUN] Would run: {' '.join(run_crs_cmd)}")
        else:
            try:
                subprocess.run(run_crs_cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Analyzer failed: {e}", file=sys.stderr)

    def run_verifier(self, output_directory: Path, target: str, harness_id: str) -> None:
        """runs the verifier"""
        run_pov_cmd = [
            "run_crs",
            "--output-dir",
            str(output_directory),
            "--ai-model",
            self.model_name,
            "--ai-model-temp",
            str(self.model_temp),
        ]

        if self.mixed or self.function_mode:
            run_pov_cmd.extend(["--function-mode", "true"])
        else:
            run_pov_cmd.extend(["--check-pov-at", target])

        run_pov_cmd.extend(
            [
                "--pov-skip-build",
                "true",
                "pov",
                str(self.cp_temp / "challenge-004-nginx-cp") if self.cp_temp else "challenge-004-nginx-cp",
            ]
        )

        if not self.function_mode and not self.mixed:
            run_pov_cmd.extend(["--commit-hash", target])

        run_pov_cmd.extend(
            [
                "--harness-id",
                harness_id,
                "--load-state",
                str(output_directory / "analyze.json"),
                "--state-file",
                str(output_directory / "pov.json"),
            ]
        )

        if self.dry_run:
            print(f"[DRY RUN] Would run: {' '.join(run_pov_cmd)}")
        else:
            try:
                subprocess.run(run_pov_cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Verifier failed: {e}", file=sys.stderr)

    def evaluate(self, args: Tuple[str, str, str, int, bool]) -> None:
        """evaluate a target harness combination"""
        signal.signal(signal.SIGINT, self.abort)
        target, harness_id, harness_name, runs, execute_verifier = args
        print(f"Evaluating {target} / {harness_id} {runs} times -- started")

        self.setup_cp(target)

        for i in range(1, runs + 1):
            print(f"Run {i} for {target}/{harness_name}")
            output_directory = self.create_output_directory(i, target)

            self.run_analyzer(output_directory, target)
            if execute_verifier:
                self.run_verifier(output_directory, target, harness_id)

        print(f"Evaluating {target} / {harness_id} {runs} times -- finished")

    def run(self, jobs: int = 1, runs: int = 1, run_verifier: bool = True) -> None:
        """Runs all evaluations with parallel jobs"""
        with open(self.json_file, encoding="UTF-8") as f:
            data = json.load(f)

        work_items: List[Tuple[str, str, str, int, bool]]
        if self.function_mode:
            work_items = [
                (func, entry["harness_id"], entry["harness_name"], runs, run_verifier)
                for entry in data.values()
                for func in entry.get("functions", [])
            ]
        else:
            work_items = [
                (entry["commit_introduced"], entry["harness_id"], entry["harness_name"], runs, run_verifier)
                for entry in data.values()
            ]

        with Pool(processes=jobs) as pool:
            try:
                pool.map(self.evaluate, work_items)
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                raise
            except Exception as e:
                print(f"Error during evaluation: {e}", file=sys.stderr)
                traceback.print_exception(type(e), e, e.__traceback__)
                raise


def get_crs_info() -> Tuple[str, str]:
    """resolves the run_crs command to aquire git information"""
    run_crs_path = shutil.which("run_crs")
    if run_crs_path is None:
        raise RuntimeError("Could not find 'run_crs' in PATH")

    try:
        repo = Repo(Path(run_crs_path).resolve().parents[2])
        return repo.head.commit.hexsha, repo.active_branch.name
    except Exception:  # pylint: disable=broad-exception-caught
        return "unknown", "unknown"


def log_eval_run(
    e: Evaluator, runs: int, status: str, logfile: Path | None, git_info: Tuple[str, str], description: str
) -> None:
    """Logs the evaluation to the evaluation log file"""
    commit, branch = git_info
    log_entry = (
        f"timestamp={datetime.now().isoformat()} crs_branch={branch} crs_commit={commit} "
        f"type=nginx llm_model={e.model_name}"
        f"sample_size={runs} temperature={e.model_temp} "
        f"eval_results_path={e.eval_dir} user={os.getlogin()} "
        f'run_description="{description}" status={status}\n'
    )
    print(log_entry)
    if logfile:
        with open(logfile, "a", encoding="UTF-8") as f:
            f.write(log_entry)


def resolve_template_path(provided_path: str) -> Path:
    """resolves the location of a template.
    absolut path first, relative second and finaly looks up if the tempate
    exists in the template directory"""
    input_path = Path(provided_path)

    if input_path.is_absolute() and input_path.exists():
        return input_path
    if input_path.expanduser().exists():
        return input_path.expanduser().resolve()

    script_dir = Path(__file__).parent.resolve()
    fallback_path = script_dir / "templates" / input_path.name

    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(f"File '{provided_path}' not found, including fallback in 'eval/templates'.")


def parse_args() -> argparse.Namespace:
    """parses the command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", action="store_true", help="Only run Analyzer")
    parser.add_argument("-t", type=Path, help="Test series path aka output path")
    parser.add_argument("-j", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("-r", type=int, default=1, help="Number of crs executions per target")
    parser.add_argument("-f", action="store_true", help="Run function based evaluation")
    parser.add_argument("--mixed", action="store_true", help="Run mixed function and commit based evaluation")
    parser.add_argument("-d", type=str, default="user did not provide description", help="Description for the log")
    parser.add_argument("--temp", type=float, default=0.0, help="LLM temperature")
    parser.add_argument("--log-file", default=Path("/data/CRSLOG"), type=Path, help="Path to the evaluation log file")
    parser.add_argument("--dry-run", action="store_true", help="Print actions instead of executing them")
    parser.add_argument("--json-file", type=Path, help="Json file to load tasks from")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("~/.bench_tmp"),
        help="Location where the temporary cp's will be placed. Default is ~/.bench_tmp.",
    )
    parser.add_argument("--model", type=str, help="LLM model to use")
    template_list = "" + ", ".join(f'"{key}"' for key in list(PRESETS)[:-1]) + f' and "{list(PRESETS)[-1]}"'
    parser.add_argument("preset", type=str, nargs="?", help=f"Configuration preset. Available are: {template_list}")

    args = parser.parse_args()
    if args.preset:
        print(f"Loading preset {args.preset}")
        try:
            for k, v in PRESETS[args.preset].items():
                print(f"Setting '{k}' to '{v}'")
                setattr(args, k, v)
        except KeyError:
            print("Preset not found")
            sys.exit(1)
    return args


def handle_sigint(_signum: int, _frame: Optional[FrameType]) -> None:
    """Handels the sigint singal in the main thread"""
    global interrupted  # pylint: disable=global-statement
    interrupted = True
    raise KeyboardInterrupt


def main() -> None:
    """main function"""
    global interrupted  # pylint: disable=global-statement
    signal.signal(signal.SIGINT, handle_sigint)
    args = parse_args()

    json_file = resolve_template_path(args.json_file)
    work_dir = args.work_dir.expanduser().resolve()
    if not args.dry_run:
        work_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(dir=work_dir))
    else:
        tmp_dir = Path("/does/not/exist/dry/run/only")

    if args.t:
        test_series_dir = Path(args.t)
    else:
        test_series_dir = (
            Path("test_series") / f"{args.json_file}_{datetime.now().isoformat()}_{args.model}-t{args.temp}"
        )

    test_series_dir = test_series_dir.expanduser().resolve()
    evaluator = Evaluator(
        model_temp=args.temp,
        model_name=args.model,
        json_file=json_file,
        eval_dir=test_series_dir,
        dry_run=args.dry_run,
        tmp_dir=tmp_dir,
        function_mode=args.f,
        mixed=args.mixed,
    )
    git_info = get_crs_info()
    log_file = Path(args.log_file).expanduser().resolve()

    log_eval_run(evaluator, args.r, "started", log_file, git_info, args.d)
    try:
        evaluator.run(jobs=args.j, runs=args.r, run_verifier=not args.a)
    except KeyboardInterrupt:
        log_eval_run(evaluator, args.r, "aborted", log_file, git_info, args.d)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(e)
        interrupted = True
        log_eval_run(evaluator, args.r, "failed", log_file, git_info, args.d)
    finally:
        if not args.dry_run:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            subprocess.run(["rm", "-rf", tmp_dir], check=True)
            signal.signal(signal.SIGINT, handle_sigint)

    if not interrupted:
        log_eval_run(evaluator, args.r, "completed", log_file, git_info, args.d)
        if not args.dry_run:
            print(generate_results_table(evaluate_test_series(test_series_dir)))


if __name__ == "__main__":
    main()
