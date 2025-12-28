"""
test the reproducer agents functionality (without LSP)

it's really hard to test singular components from the agent
so this is basically a test of the full agent
some of the tests provide inputs that target a specific function in the agent
(expecting the rest to run smoothly)
"""

import argparse
import shutil
from pathlib import Path
from test.setup_test_env import OSS_FUZZ_PATH

from git import Repo
from langgraph.errors import GraphRecursionError

from crs.agents.pov_builder.pov_builder_agent import PovBuilderAgent
from crs.base.context import CrsContext
from crs.base.context import ReproducerDescription
from crs.base.precontext import CrsPreContext
from crs.base.precontext import VulnerabilityStateFields
from crscommon.settings.settings import SETTINGS
from oss_fuzz_integration.prepare_project import transform_project

DATA_PATH = Path(__file__).parent.parent / "test/data"
MOTIVATIONAL_DATA_PATH = DATA_PATH / "motivational-example"
CALL_TREE_SUB_DIR = Path("calltree/")
CALL_TREE_DATA_PATH = DATA_PATH / "calltree/"
LOG_DIR = Path("test_log/")


def apply_config(settings: dict[str, str]) -> None:
    """
    configure the CRS, 'mock_chat' is a required field to be configured
    names are the "normal-lower-case-names" that are defined as the settings name in a settings.py
    you can add and overwrite all settings using the settings dict
    """
    config = {
        "ai-model-name": "MockLLM",
        "pov-standalone-mode": "true",
        "pov-skip-build": "true",
        "output-dir": LOG_DIR.as_posix(),
        # the value is totally irrelevant (for the test), but the CRS requires it to be set
        "pov-target-function": "handle_request",
        "call-tree-analysis-dir": (LOG_DIR / CALL_TREE_SUB_DIR).as_posix(),
        **settings,
    }

    if "mock-chat" not in config:
        raise ValueError("The config of for the tests MUST contain a mock chat. Settings name: 'mock-chat'")

    SETTINGS.apply(config)


def create_required_crs_folders() -> None:
    """set up the environment for the tests"""
    LOG_DIR.mkdir(exist_ok=True, parents=True)

    # copy inspector logs that we would normally download if not already on disk
    # but we can't download if the project does not publicly exist
    shutil.copytree(CALL_TREE_DATA_PATH, LOG_DIR / CALL_TREE_SUB_DIR, dirs_exist_ok=True)


def prepare_motivational_example() -> None:
    """
    clone the butchered motivational example that is used for testing if run.sh is invoked and sanitizers are detected
    the cloned this is not buildable
    """
    if (Path(__file__).parent.parent / "cp-motivational-example-HEAD").exists():
        return

    if not (OSS_FUZZ_PATH / "projects" / "motivational-example").exists():
        shutil.copytree(MOTIVATIONAL_DATA_PATH, OSS_FUZZ_PATH / "projects" / "motivational-example", dirs_exist_ok=True)

    args = argparse.Namespace()
    args.commit_hash = None
    args.double_build_mode = False
    args.success_list_path = None
    args.failed_list_path = None
    args.no_oss_fuzz_checkout = False
    args.single_cp = False
    args.unique_tag = "motivational-example-docker-container"
    args.success_list_path = Path("/dev/null")

    transform_project(
        "motivational-example",
        None,
        DATA_PATH,
        OSS_FUZZ_PATH,
        Repo(OSS_FUZZ_PATH),
        Path(__file__).parent.parent / "oss_fuzz_integration",
        args,
    )

    # move the project.yaml from the data folder to the cp folder
    shutil.copy(
        DATA_PATH / "motivational-example" / "project.yaml",
        DATA_PATH / "cp-motivational-example-HEAD" / "project.yaml",
    )


def get_crs_context() -> CrsContext:
    """get the minimal required CRS context"""
    return CrsPreContext(cp_path=DATA_PATH / "cp-motivational-example-HEAD").to_context(
        {VulnerabilityStateFields.REPORT}
    )


def test_standalone_pov_builder_success() -> None:
    """
    test if a valid AIMessage with a valid gen-reproducer tool call
    results in a graceful stop of the PoV Builder

    for this a non-functional mini-example was built
    it is not buildable and does not contain a valid run.sh but a mocked one

    this tests if:
    - the PoV Builder agent is called correctly
    - the gen-reproducer tool works as expected (detect sanitizer)
    - the sanitizer checks run as expected
    - the ReproducerDescription was created and added to the Context
    - the reproducer is located at the expected location

    does NOT test if:
    - LLM calls work (MockLLM does not call anything)
    - the prompt was generated correctly
    """

    create_required_crs_folders()
    prepare_motivational_example()

    apply_config({"mock-chat": (DATA_PATH / "mock-chat-reproducer-successful.json").as_posix()})

    ctx = get_crs_context()
    try:
        agent = PovBuilderAgent(ctx)
        ctx = agent.run()
    # StopIteration happens when first AI message doesn't hit
    # and MockLLM is called again
    # make this error more understandable
    except StopIteration as exc:
        raise RuntimeError("The pseudo-vulnerability was not triggered by a valid tool call.") from exc

    assert len(ctx.vulnerabilities) == 1

    vuln = ctx.vulnerabilities[0].reproducer

    # when the thing exists all fields must hold valid data due to pydantic
    assert isinstance(vuln, ReproducerDescription)

    # check if pov is at expected location
    assert vuln.path.parent == LOG_DIR

    # check that blob was written successfully
    assert vuln.path.read_bytes() == vuln.blob

    # TODO: maybe also chat for pov-builder json and md?


def _run_for_failure(ctx: CrsContext) -> None:
    """
    run the CRS and expect NO success from the provided messages
    -> we want a StopIteration error from exhausting all messages oaded into MockLLM
    """
    try:
        agent = PovBuilderAgent(ctx)
        ctx = agent.run()
    # StopIteration happens when first AI message doesn't hit
    # and MockLLM is called again, we only provide one input and this one is INVALID python
    # so we expect to die, if we don't there is a problem, where the faulty reproducer was detected as valid.
    except StopIteration:
        pass

    except RuntimeError as e:
        if "StopIteration" not in str(e):
            raise e

    else:
        raise GraphRecursionError("This CRS execution should not have ended gracefully, bc no crashing was provided.")

    # TODO: check if the logs are still at the expected location


def test_standalone_pov_builder_failure() -> None:
    """
    we provide a reproducer of a valid type that does not trigger or mocked run.sh
    we expect that the agent does not terminate gracefully through the provided messages
    """
    create_required_crs_folders()
    prepare_motivational_example()

    apply_config({"mock-chat": (DATA_PATH / "mock-chat-reproducer-does-not-trigger.json").as_posix()})

    _run_for_failure(get_crs_context())


def test_standalone_pov_builder_invalid_pov() -> None:
    """
    this test calls the gen-reproducer tool with a list as result instead of str or bytes
    we expect a death here

    essentially tests if the sandbox catches reproducers that have the wrong type
    """
    create_required_crs_folders()
    prepare_motivational_example()

    apply_config({"mock-chat": (DATA_PATH / "mock-chat-reproducer-is-list.json").as_posix()})

    _run_for_failure(get_crs_context())
