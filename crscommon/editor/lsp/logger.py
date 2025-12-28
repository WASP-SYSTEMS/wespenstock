"""Logger for lsp related stuff."""

from crscommon.logging.logging_provider import LOGGING_PROVIDER

LSP_LOGGER = LOGGING_PROVIDER.new_logger("lsp", log_to_console=False)
