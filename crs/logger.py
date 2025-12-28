"""Logger for the CRS."""

from crscommon.logging.logging_provider import LOGGING_PROVIDER

CRS_LOGGER = LOGGING_PROVIDER.new_logger("crs", hook_exception=True)
