"""
Repository for CRS tuning knobs.

Settings are declared using setting(), bound to CLI arguments using init_settings(), and loaded from CLI arguments and
environment variables by load_settings().
By default, a setting has no corresponding CLI option, use "cli_option=" to specify one.
By default, a setting's environment variable is derived from its name (e.g., the setting "pet-the-kittens" is mapped
to "CRS_PET_THE_KITTENS"), but settings may specify custom environment variable names.

Entry points (e.g. main functions) are expected to call init_settings() at some early time.
init_settings() may be passed an argparse ArgumentParser in order to allow settings to be set on the command line.
Afterwards, load_settings() must be called to actually set the settings.
If using CLI parsing, load_settings() must be passed the parsed arguments object from parse_args().
(Therefore, init_settings() must be called before parse_args(), and load_settings() after parse_args().)

Consumers of settings are expected to declare their settings using setting() in some reasonably central location,
e.g. a settings.py file in their package, and then to import and use them in their main code.
"""

from crscommon.settings.settings import Setting
from crscommon.settings.settings import init_settings
from crscommon.settings.settings import load_settings
from crscommon.settings.settings import setting

# WARNING: Use the setting() function (not the Setting class) to declare settings unless you know what you're doing!
#          Setting may be used to type-hint optional settings like Setting[int | None]. (Their type cannot be
#          inferred, even from the type passed as an argument, due to Python deficiencies.)
__all__ = ["init_settings", "load_settings", "setting", "Setting"]
