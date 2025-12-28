"""
Editor tests for python.
"""

from pathlib import Path

import pytest

from crscommon.editor.editor import SourceEditor
from crscommon.editor.symbol import LineInSymbolDefinition
from crscommon.editor.symbol import SymbolOccurrence

from .setup_test_env import TEST_PYTHON_SRC_PATH
from .setup_test_env import reset_python_env

TEST_PYTHON_TEXT_DOC = TEST_PYTHON_SRC_PATH / "src/dotenv/main.py"


@pytest.fixture(autouse=True)
def reset_env() -> None:
    """Reset env before every test."""
    reset_python_env()


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("symbol_name", "symbol_name", True),
        ("symbol_name", "symbolname", False),
        ("abs.mod.path.Hello", "Hello", True),
        ("Hello", "abs.mod.path.Hello", True),
        ("aaa", "a", False),
    ],
)
def test_symbol_names_equal(py_editor: SourceEditor, a: str, b: str, expected: bool) -> None:
    """test_symbol_names_equal"""

    assert py_editor.li.symbol_names_equal(a, b) is expected


@pytest.mark.parametrize(
    "occurrence, expected_symbol",
    [
        (
            SymbolOccurrence(
                name="load_dotenv",
                file=TEST_PYTHON_SRC_PATH / "src/dotenv/ipython.py",
                line=33,
            ),
            """def load_dotenv(
    dotenv_path: Optional[StrPath] = None,
    stream: Optional[IO[str]] = None,
    verbose: bool = False,
    override: bool = False,
    interpolate: bool = True,
    encoding: Optional[str] = "utf-8",
) -> bool:
    \"""Parse a .env file and then load all the variables found as environment variables.

    Parameters:
        dotenv_path: Absolute or relative path to .env file.
        stream: Text stream (such as `io.StringIO`) with .env content, used if
            `dotenv_path` is `None`.
        verbose: Whether to output a warning the .env file is missing.
        override: Whether to override the system environment variables with the variables
            from the `.env` file.
        encoding: Encoding to be used to read the file.
    Returns:
        Bool: True if at least one environment variable is set else False

    If both `dotenv_path` and `stream` are `None`, `find_dotenv()` is used to find the
    .env file with it's default parameters. If you need to change the default parameters
    of `find_dotenv()`, you can explicitly call `find_dotenv()` and pass the result
    to this function as `dotenv_path`.
    \"""
    if dotenv_path is None and stream is None:
        dotenv_path = find_dotenv()

    dotenv = DotEnv(
        dotenv_path=dotenv_path,
        stream=stream,
        verbose=verbose,
        interpolate=interpolate,
        override=override,
        encoding=encoding,
    )
    return dotenv.set_as_environment_variables()
""",
        ),
        (
            SymbolOccurrence(
                name="parse_stream",
                file=TEST_PYTHON_SRC_PATH / "src/dotenv/main.py",
                line=84,
            ),
            """def parse_stream(stream: IO[str]) -> Iterator[Binding]:
    reader = Reader(stream)
    while reader.has_next():
        yield parse_binding(reader)
""",
        ),
        (
            LineInSymbolDefinition(
                name="parse_stream",
                file=TEST_PYTHON_SRC_PATH / "src/dotenv/parser.py",
                line=173,
            ),
            """def parse_stream(stream: IO[str]) -> Iterator[Binding]:
    reader = Reader(stream)
    while reader.has_next():
        yield parse_binding(reader)
""",
        ),
    ],
)
def test_get_symbol_definition(
    py_editor: SourceEditor, occurrence: SymbolOccurrence | LineInSymbolDefinition, expected_symbol: str
) -> None:
    """Test symbol retrieval"""
    assert py_editor.get_symbol_definition(occurrence).definition == expected_symbol


@pytest.mark.parametrize(
    "occurrence, function_names",
    [
        (
            SymbolOccurrence(
                name="parse_stream",
                file=TEST_PYTHON_SRC_PATH / "src/dotenv/main.py",
                line=84,
            ),
            [
                "parse",
                "main",
                "set_key",
                "unset_key",
                "parse_stream",
            ],
        )
    ],
)
def test_get_functions_referencing_symbol(
    py_editor: SourceEditor, occurrence: SymbolOccurrence, function_names: list[str]
) -> None:
    """Test enclosing function extraction"""

    functions = py_editor.get_functions_referencing_symbol(occurrence)
    assert {f.name for f in functions} == set(function_names)


@pytest.mark.parametrize(
    "file, replacement",
    [
        (
            TEST_PYTHON_TEXT_DOC,
            ("if mapping.error:", "if mapping.error"),
        ),
    ],
)
def test_diagnostics(py_editor: SourceEditor, file: Path, replacement: tuple[str, str]) -> None:
    """Test diagnostics."""

    py_editor.li.open_file(file)

    changed_text_doc = TEST_PYTHON_TEXT_DOC.read_text("utf-8").replace(*replacement)
    TEST_PYTHON_TEXT_DOC.write_text(changed_text_doc, encoding="utf-8")

    diagnostics = py_editor.get_diagnostics()

    reset_python_env()

    error_found = False

    for d in diagnostics[file.absolute()]:
        if d.severity == 1:
            error_found = True

    assert error_found is True
