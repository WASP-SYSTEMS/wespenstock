"""Settings framework implementation."""

import os
import typing
from argparse import ArgumentParser
from argparse import Namespace
from types import NoneType
from types import UnionType
from typing import Any
from typing import Callable
from typing import Generic
from typing import Iterator
from typing import TypeVar

T = TypeVar("T")

SettingType = type[T] | UnionType | Callable[[str], T]


class Setting(Generic[T]):  # pylint: disable=R0902
    """
    A single setting.

    manager: The SettingsManager this setting belongs to.

    See the module-level setting() for descriptions of the other parameters.
    """

    def __init__(
        self,
        manager: "SettingsManager",
        name: str,
        type: SettingType[T],  # pylint: disable=W0622
        *,
        env_name: str | None = None,
        cli_option: str | list[str] | None = None,
        description: str | None = None,
        default: T | None = None,
        parent: "Setting[T] | None" = None,
    ):
        if isinstance(cli_option, str):
            cli_option = [cli_option]
        self.manager = manager
        self.name = name
        self.type = type
        self.env_name = env_name or f"CRS_{name.upper().replace('-', '_')}"
        self.cli_option = cli_option
        self.description = description
        self.value = default
        self.parent = parent
        self._is_optional = typing.get_origin(type) is UnionType and NoneType in typing.get_args(type)

    @property
    def _parser(self) -> Callable[[str], T]:
        """
        A callable suitable for parsing non-None values of this setting.

        If self.type is an optional, this unwraps it to the enclosed type; otherwise, just returns self.type.

        The returned callable might be more suitable for parsing strings than self.type (calling an optional given a
        string just raises an exception).
        """
        if typing.get_origin(self.type) is UnionType:
            subtypes = [t for t in typing.get_args(self.type) if t is not NoneType]
            if len(subtypes) == 0:
                raise RuntimeError(f"Setting {self.name!r} cannot be set?!")
            if len(subtypes) == 1:
                return subtypes[0]
            # Let the caller deal with it (it will probably fail, but the above should already work for the important
            # case of optionals).
        assert callable(self.type), f"Setting type {self.type} cannot be used to parse strings"
        return self.type

    @property
    def _description(self) -> str:
        "A string describing this setting's type."
        tp = self.type
        if typing.get_origin(tp) is UnionType:
            subtypes = [t for t in typing.get_args(tp) if t is not NoneType]
            if len(subtypes) == 1:
                tp = subtypes[0]
        try:
            return tp.__name__  # type: ignore  # Yes, this is going to be fine, we'll catch it.
        except AttributeError:
            return str(tp)

    @property
    def present(self) -> bool:
        "Whether the setting is duly and properly set."
        return self.value is not None or self._is_optional or (self.parent is not None and self.parent.present)

    def get(self) -> T:
        "Return the setting's value, or raise an error if it is missing."
        self.manager.ensure_loaded(False)
        if self.value is None:
            if self.parent is not None and self.parent.present:
                return self.parent.get()
            if not self._is_optional:
                raise RuntimeError(f"Setting {self.name!r} absent or not loaded")
        return self.value  # type: ignore

    def must_get(self) -> T:
        "Return the setting's value, or raise an error if it is missing or not set."
        self.manager.ensure_loaded(False)
        if self.value is None:
            if self.parent is not None:
                return self.parent.must_get()
            raise RuntimeError(f"Setting {self.name!r} absent or not loaded")
        return self.value

    def set(self, value: T) -> None:
        "Set the setting's value."
        self.value = value

    def parse(self, text: str) -> None:
        "Given a string (e.g. from the environment), parse it and set this setting's value to the result."
        self.value = self._parser(text)

    def add_to_cli(self, parser: ArgumentParser) -> None:
        "If this setting has a CLI option, add it to the given parser."
        if self.cli_option is None:
            return
        parser.add_argument(
            *self.cli_option,
            metavar=f"<{self._description.upper()}>",
            dest=f"setting-{self.name}",
            help=self.description,
            type=self._parser,
        )

    def get_from_cli(self, args: Namespace) -> None:
        "If this setting has a CLI option, set its value from the given parsed arguments."
        if self.cli_option is None:
            return
        value = getattr(args, f"setting-{self.name}")
        if value is None:
            return
        self.value = value


class SettingsManager:
    """
    A collection of settings and means of to bulk-configure them.
    """

    def __init__(self) -> None:
        self.settings: dict[str, Setting] = {}
        self.may_add = True
        self.loaded = False

    def __iter__(self) -> Iterator[Setting]:
        "Iterating a SettingsManager yields the contained settings."
        return iter(self.settings.values())

    @property
    def values(self) -> dict[str, object]:
        "Return a dict of all set settings and their values."
        return {k: s.value for k, s in self.settings.items()}

    def _missing_error(self, s: Setting) -> Exception:
        "Internal: Prepare an exception with a fancy error message for a missing setting."
        msg = f"Missing required setting {s.name!r}"
        msg += f"\nHint: Set the environment variable {s.env_name}"
        if s.description is not None:
            msg += f"\nSetting description: {s.description}"
        return RuntimeError(msg)

    def _invalid_error(self, s: Setting, exc: BaseException) -> Exception:
        "Internal: Prepare an exception with a fancy error message for an invalid setting value string."
        msg = f"Invalid value for setting {s.name!r}, expected type {s.type}"
        msg += f"Exception: {type(exc).__name__}: {exc}"
        if s.description is not None:
            msg += f"\nSetting description: {s.description}"
        return RuntimeError(msg)

    def add(self, name: str, type: SettingType[T], *args: Any, **kwargs: Any) -> Setting[T]:  # pylint: disable=W0622
        """
        Register the given setting with this SettingsManager.

        All arguments are forwarded to the Setting constructor; see the module-level setting() for their meaning.
        The new setting's manager is this SettingsManager.

        Returns the new setting.
        """
        if name in self.settings:
            raise ValueError(f"Trying to declare setting {name!r} which already exists")
        if not self.may_add or self.loaded:
            raise RuntimeError(f"Trying to declare setting {name!r} after settings have been finalized")
        result = Setting(self, name, type, *args, **kwargs)
        self.settings[name] = result
        return result

    def add_to_cli(self, parser: ArgumentParser | None = None) -> None:
        """
        Add all settings to the given parser.

        parser: An argparse ArgumentParser to configure options for those settings that have CLI options.

        For symmetry with load(), this allows passing no parser; in that case, this does nothing.
        """
        if parser is not None:
            for s in self:
                s.add_to_cli(parser)
        self.may_add = False

    def apply(self, values: dict[str, str]) -> None:
        """
        Set the given settings to the given values.

        values is a mapping from setting names (*not* environment variables) to unparsed strings.
        """
        for s in self:
            try:
                v = values[s.name]
            except KeyError:
                continue
            s.parse(v)
        self.loaded = True

    def load(self, args: Namespace | None = None, need_all: bool = True) -> None:
        """
        Load values for all settings in this manager and parse them.

        args: If given, parsed command-line arguments to get setting values from.
        need_all: If true, raise an error if the value for a setting is missing. If false, ignore missing settings.

        Setting values are drawn from args, and also environment variables using the settings' env_names.
        """
        for s in self:
            try:
                env_text = os.environ[s.env_name]
            except KeyError:
                pass
            else:
                try:
                    s.parse(env_text)
                except ValueError as exc:
                    raise self._invalid_error(s, exc) from exc
            if args is not None:
                s.get_from_cli(args)
            if need_all and not s.present:
                raise self._missing_error(s)
        self.loaded = True

    def ensure_loaded(self, need_all: bool = True) -> None:
        """
        Ensure all settings have been loaded.

        need_all: If true (the default), also ensure that that all settings are fully configured.

        This ensures that apply() or load() have been called at least once before this call.
        """
        if not self.loaded:
            # Developers: To fix this, call crscommon.settings.init_settings() somewhere after all settings have been
            # declared but before any are used.
            # If you are using a custom SettingsManager, call its apply() or load() methods instead (as appropriate).
            raise RuntimeError("CRS settings were never loaded!")
        if need_all:
            for s in self:
                if not s.present:
                    raise self._missing_error(s)


SETTINGS = SettingsManager()


def setting(
    name: str,
    type: SettingType[T],  # pylint: disable=W0622
    *,
    env_name: str | None = None,
    cli_option: str | list[str] | None = None,
    description: str | None = None,
    default: T | None = None,
    parent: Setting[T] | None = None,
) -> Setting[T]:
    """
    Declare a setting.

    name: The setting's name, must be unique among settings in the SettingsManager.
    type: The setting's type or factory function, something callable given a string.
    env_name: Environment variable name for the setting. Default: derived from name.
    cli_option: Name (or names) of CLI options as understood by argparse add_argument().
    description: A string to help the user to determine what to set the setting to.
    default: The default value for the setting.
    parent: A setting to use as a fallback if this setting is not set.

    The setting is stored in the global settings manager, which is also used by init_settings() and load_settings().
    """
    return SETTINGS.add(
        name,
        type,
        env_name=env_name,
        cli_option=cli_option,
        description=description,
        default=default,
        parent=parent,
    )


def init_settings(parser: ArgumentParser | None = None) -> None:
    """
    Initialize the settings manager.

    parser: An argparse ArgumentParser to configure options for those settings that have CLI options.

    After this call, no new settings may be declared.

    This uses the same global settings manager as setting().
    """
    SETTINGS.add_to_cli(parser)


def load_settings(args: Namespace | None = None) -> None:
    """
    Load values for all configured settings.

    args: If you have passed an ArgumentParser to init_settings(), pass its parsed arguments here.

    After this call, all declared settings are bound to values (unless they are optional).

    This uses the same global settings manager as setting().
    """
    SETTINGS.load(args)
