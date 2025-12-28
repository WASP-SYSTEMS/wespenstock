"""
Module for testing logging

Run "python3 -m crscommon.logging" to see if basic logging setup works.
"""

import argparse

from ..settings import init_settings
from ..settings import load_settings
from .dump_config import dump_config
from .logging_provider import LOGGING_PROVIDER

log = LOGGING_PROVIDER.new_logger("TEST")


def main() -> None:
    "Main function for logging testing."
    p = argparse.ArgumentParser()
    init_settings(p)
    a = p.parse_args()
    load_settings(a)
    LOGGING_PROVIDER.init_logging()
    dump_config(log)
    log.info("Logging test done!")


if __name__ == "__main__":
    main()
