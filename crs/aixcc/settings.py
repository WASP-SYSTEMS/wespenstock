"Settings."

from crscommon.settings.settings import setting

RUNSH_POV_TIMEOUT = setting(
    "runsh-timeout",
    int,
    description="Timeout for executing PoVs via run.sh (in seconds)",
    default=300,
    cli_option="--runsh-timeout",
)
