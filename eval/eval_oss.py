"""Evaluate oss datasets."""

import argparse
import logging
import subprocess
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

from pydantic import BaseModel

from oss_fuzz_integration.introspector_api import AllFunctionsResponse
from oss_fuzz_integration.introspector_api import fetch_and_cache_api_data

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class Entry(BaseModel):
    """Dataset entry."""

    name: str
    oss_fuzz_issue: str
    harness_name: str
    vuln_function: str
    vuln_commit: str
    fixing_commit: str | None = None
    introspector_url: str


class Dataset(BaseModel):
    """Oss-Fuzz dataset."""

    projects: list[Entry]


def cp_dir_from_entry(entry: Entry) -> str:
    """Get CP dir name."""
    return f"cp-{entry.name}-{entry.vuln_commit}"


class EvaluatorOss:
    """Evaluator for OSS."""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        model_temp: float,
        model_name: str,
        eval_dir: Path,
        dry_run: bool = False,
    ):
        self.model_temp = model_temp
        self.model_name = model_name
        self.eval_dir = eval_dir.absolute()
        self.results_dir = self.eval_dir / Path("results") / f"{model_name}-t{model_temp}_{datetime.now().isoformat()}"
        self.project_dir = self.eval_dir / Path("projects")
        self.dry_run = dry_run

        self.prepare_only = False
        self.runs = 1

        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.project_dir.mkdir(exist_ok=True, parents=True)

    def new_out_dir(self, entry: Entry) -> Path:
        """Create new output directory for one run."""

        output_directory = self.results_dir / f"{entry.name}-{entry.vuln_function}" / datetime.now().isoformat()
        if self.dry_run:
            logging.info(f"[{entry.name}][{entry.vuln_function}] [DRY RUN] Create output dir {output_directory}")
        else:
            logging.info(f"[{entry.name}][{entry.vuln_function}] Create output dir {output_directory}")
            output_directory.mkdir(parents=True, exist_ok=True)

        return output_directory

    def analyze(self, entry: Entry, out_dir: Path) -> bool:
        """Run the analyzer agent."""
        cp_dir = self.project_dir / cp_dir_from_entry(entry)

        cmd = [
            "run_crs",
            "--output-dir",
            out_dir.as_posix(),
            "--ai-model",
            self.model_name,
            "--ai-model-temp",
            str(self.model_temp),
            "--target-function",
            entry.vuln_function,
            "--function-mode",
            "true",
            "analyze",
            cp_dir.as_posix(),
            "--commit-hash",
            entry.vuln_commit,
            "--state-file",
            (out_dir / "analyze.json").as_posix(),
        ]

        return self._exec_cmd(cmd, entry, cwd=self.eval_dir)

    def verify(self, entry: Entry, out_dir: Path) -> None:
        """Run the verifier agent."""
        cp_dir = self.project_dir / cp_dir_from_entry(entry)

        cmd = [
            "run_crs",
            "--output-dir",
            out_dir.as_posix(),
            "--ai-model",
            self.model_name,
            "--ai-model-temp",
            str(self.model_temp),
            "--target-function",
            entry.vuln_function,
            "--function-mode",
            "true",
            "--check-pov-at",
            entry.vuln_commit,
            "--pov-skip-build",
            "true",
            "pov",
            cp_dir.as_posix(),
            "--harness-name",
            entry.harness_name,
            "--load-state",
            (out_dir / "analyze.json").as_posix(),
            "--state-file",
            (out_dir / "pov.json").as_posix(),
        ]

        self._exec_cmd(cmd, entry, cwd=self.eval_dir)

    def evaluate(self, dataset: Dataset, jobs: int = 1, runs: int = 1, prepare_only: bool = False) -> None:
        """Evaluate on dataset."""

        self.prepare_only = prepare_only
        self.runs = runs

        logging.info(f"Evaluating using {jobs} jobs")

        with Pool(processes=jobs) as pool:
            try:
                pool.map(self.evaluate_one, dataset.projects)
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                raise
            except Exception as e:
                logging.error(f"Error during evaluation of one entry: {e}")
                raise

    def evaluate_one(self, entry: Entry) -> None:
        """Evaluate one dataset entry."""
        if not self._prepare_project(entry):
            logging.error(f"Failed to prepare project {entry.name}")
            return

        if self.prepare_only:
            logging.info(f"[{entry.name}][{entry.vuln_function}] Skipping evaluation (project preparation only)")
            return

        for _ in range(0, self.runs):
            logging.info(f"[{entry.name}][{entry.vuln_function}] Start evaluation")
            out_dir = self.new_out_dir(entry)

            if self.analyze(entry, out_dir):
                self.verify(entry, out_dir)

            logging.info(f"[{entry.name}][{entry.vuln_function}] Evaluation finished")

    def _prepare_project(self, entry: Entry) -> bool:
        """Prepare oss fuzz project."""

        cp_dir = (self.project_dir / cp_dir_from_entry(entry)).absolute()

        if cp_dir.exists():
            logging.info(f"[{entry.name}][{entry.vuln_function}] Project found at {cp_dir.as_posix()}")
            return True

        logging.info(f"[{entry.name}][{entry.vuln_function}] Preparing project {entry.name}")

        cmd = [
            "prepare_project",
            "--force",
            "--commit-hash",
            entry.vuln_commit,
            "--output-dir",
            self.project_dir.as_posix(),
            "--no-oss-fuzz-checkout",
            "--success-list-path",
            (self.project_dir / "preparation-success.txt").as_posix(),
            "--failed-list-path",
            (self.project_dir / "preparation-failed.txt").as_posix(),
            "--unique-tag",
            f"oss-eval-{entry.name}-{entry.vuln_commit}",
            entry.name,
        ]

        if not self._exec_cmd(cmd, entry, cwd=self.eval_dir):
            return False

        # download all_functions.json

        url = "https://introspector.oss-fuzz.com/api/all-functions"
        params = {"project": entry.name}
        filepath = (
            Path(__file__).resolve().parent.parent
            / "oss_fuzz_integration"
            / "introspector-artifacts"
            / entry.name
            / "all_functions.json"
        )

        result = fetch_and_cache_api_data(
            url=url,
            params=params,
            filepath=filepath,
            data_extractor=AllFunctionsResponse.model_validate,
            retries=5,
            log=logging.getLogger(),
        )

        if result is None:
            return False

        return True

    def _exec_cmd(self, cmd: list[str], entry: Entry, cwd: Path) -> bool:
        """Execute a command as subprocess."""

        if self.dry_run:
            logging.info(f"[{entry.name}][{entry.vuln_function}] [DRY RUN] Executing {' '.join(cmd)}")
        else:
            logging.info(f"[{entry.name}][{entry.vuln_function}] Executing {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                logging.error(f"[{entry.name}][{entry.vuln_function}] Command failed")
                return False

        return True


def main() -> None:
    """Main."""

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--eval-dir", type=Path, help="Output directory for evaluation data")
    parser.add_argument("--temp", type=float, default=0.0, help="LLM temperature")
    parser.add_argument("--model", type=str, help="LLM model")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--runs", type=int, default=1, help="Number of executions per dataset entry")
    parser.add_argument("--dry-run", action="store_true", help="Print actions instead of executing them")
    parser.add_argument("--prepare-projects-only", action="store_true", help="Prepare projects without evaluation")

    args = parser.parse_args()

    dataset_file = Path(args.dataset)
    dataset = Dataset.model_validate_json(dataset_file.read_text(encoding="utf-8"))

    evaluator = EvaluatorOss(
        model_temp=args.temp,
        model_name=args.model,
        eval_dir=args.eval_dir,
        dry_run=args.dry_run,
    )

    evaluator.evaluate(dataset, jobs=args.jobs, runs=args.runs, prepare_only=args.prepare_projects_only)


if __name__ == "__main__":
    main()
