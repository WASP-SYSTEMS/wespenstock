"""Main module."""

import argparse
import os
from pathlib import Path

from crs.agents.analyzer.commit_analyzer import CommitAnalyzer
from crs.agents.analyzer.function_analyzer import FunctionAnalyzer
from crs.agents.analyzer.mixed_analyzer import MixedAnalyzer
from crs.agents.patcher.patcher_agent import PatcherAgent
from crs.agents.pov_builder.pov_builder_agent import PovBuilderAgent
from crs.agents.pov_builder.verifier_code_path import VerifierAgentCodePath
from crs.agents.pov_builder.verifier_cov_pilot import CovPilot
from crs.agents.pov_builder.verifier_function import FunctionVerifier
from crs.aixcc.project_yaml import ProjectYaml
from crs.base.agent_base import BaseAgent
from crs.base.precontext import CrsPreContext
from crs.base.precontext import VulnerabilityStateFields
from crs.base.settings import COV_PILOT_MODE
from crs.base.settings import FUNCTION_MODE
from crs.base.settings import MIXED_MODE
from crs.base.settings import POV_USE_CALL_TREE_PROMPTING
from crs.logger import CRS_LOGGER
from crscommon.logging.dump_config import dump_config
from crscommon.logging.logging_provider import LOGGING_PROVIDER
from crscommon.settings import init_settings
from crscommon.settings import load_settings

log = CRS_LOGGER.getChild(__name__)


# pylint: disable=too-many-branches
def main() -> None:  # pylint: disable=R0915
    "The main function!"

    def add_local_args(parser: argparse.ArgumentParser) -> None:
        "Declare CLI arguments for the local execution modes."
        parser.add_argument(
            "--state-file",
            type=Path,
            help="JSON file where to write analysis/PoV state for subsequent steps (required for pov and patch)",
        )
        parser.add_argument(
            "--load-state",
            type=Path,
            help="JSON file where to load analysis/PoV state from (default: same as --state-file)",
        )
        parser.add_argument(
            "--src-path",
            type=Path,
            help=(
                "path to source code subrepository (within PROJECT_PATH/src; "
                "must be given if there is more than one)"
            ),
        )
        parser.add_argument("--commit-hash", help="Git commit hash to concentrate on (default: latest)")
        parser.add_argument("--harness-id", help="Harness to concentrate upon (default: arbitrary but consistent)")
        parser.add_argument(
            "--harness-name",
            help="Harness to concentrate upon (default: arbitrary but consistent) (overwrites --harness-id)",
        )
        parser.add_argument("project_path", type=Path, help="path of the challenge project to inspect")

    def local_context(args: argparse.Namespace) -> CrsPreContext:
        "Prepare a context for the local execution modes."
        state_file_load = args.load_state or args.state_file
        state_file_save = args.state_file

        if args.harness_name:
            project_yaml = ProjectYaml.from_cp_path(args.project_path)
            harness_id = project_yaml.harness_id_by_name(args.harness_name)
        else:
            harness_id = args.harness_id

        return CrsPreContext(
            cp_path=args.project_path,
            src_path_in_cp=args.src_path,
            commit_hash=args.commit_hash,
            harness_id=harness_id,
            state_file_load=state_file_load,
            state_file_save=state_file_save,
        )

    parser = argparse.ArgumentParser(description="the WASPS CRS")

    init_settings(parser)

    # For those among us who use PyCharm, which cannot do this on its own.
    parser.add_argument(
        "--add-to-path",
        type=str,
        help="String to prepend to $PATH (trailing colon not necessary)",
        default="",
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="mode of operation")

    analyze_parser = subparsers.add_parser("analyze", help="run the analyzer agent only")
    add_local_args(analyze_parser)

    pov_parser = subparsers.add_parser("pov", help="run the PoV builder agent only")
    add_local_args(pov_parser)

    patch_parser = subparsers.add_parser("patch", help="run the patcher agent only")
    add_local_args(patch_parser)

    args = parser.parse_args()

    # load settings before initializing logging, because the logger uses the settings
    load_settings(args)

    LOGGING_PROVIDER.init_logging()

    # log settings after initializing logging
    dump_config(log)

    precontext: CrsPreContext | None = None
    pipeline: BaseAgent

    log.info(f"CRS arguments: {args}")

    # why? see declaration of arg
    path: str
    if path := args.add_to_path:
        if not path.endswith(":"):
            path += ":"
        os.environ["PATH"] = f"{path}:{os.environ['PATH']}"

    if args.command == "analyze":
        precontext = local_context(args)
        context = precontext.to_context()
        if MIXED_MODE.get():
            pipeline = MixedAnalyzer(context)
        elif FUNCTION_MODE.get():
            pipeline = FunctionAnalyzer(context)
        else:
            pipeline = CommitAnalyzer(context)

    elif args.command == "pov":
        precontext = local_context(args)
        context = precontext.to_context({VulnerabilityStateFields.REPORT})

        if POV_USE_CALL_TREE_PROMPTING.get():
            pipeline = VerifierAgentCodePath(context)

        # TODO: CLI Arg covpilot instead?
        elif COV_PILOT_MODE.get():
            pipeline = CovPilot(context)

        elif FUNCTION_MODE.get():
            pipeline = FunctionVerifier(context)

        else:
            pipeline = PovBuilderAgent(context)

    elif args.command == "patch":
        precontext = local_context(args)
        context = precontext.to_context({VulnerabilityStateFields.REPORT, VulnerabilityStateFields.REPRODUCER})
        pipeline = PatcherAgent(context)

    else:
        raise ValueError(f"We don't know what '{args.command}' means as an agent?")

    context = pipeline.run()

    context.save()


if __name__ == "__main__":
    main()
