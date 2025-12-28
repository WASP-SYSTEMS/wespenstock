"""Settings for CRS logger."""

from pathlib import Path

from crscommon.settings import setting

OUTPUT_DIR = setting(
    "output-dir",
    Path,
    description=(
        "Filesystem directory to store the CRS' outputs. "
        "Like the reproducer, agent logs, logs, coverage data etc. "
        "Default is '.' (the current working directory)."
    ),
    default=Path.cwd(),
    cli_option="--output-dir",
)

DUMP_ALL_CONFIG = setting(
    "dump-all-config",
    bool,
    description="Spend extra effort to gather more information about the CRS' environment",
    default=False,
    cli_option="--dump-all-config",
)
