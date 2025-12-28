# pylint: disable=line-too-long
"""
Pydantic Models for typing return values of API calls to Fuzz Introspector
"""

import json
from logging import Logger
from pathlib import Path
from typing import Any
from typing import Callable
from typing import TypeVar

import requests
from pydantic import BaseModel
from pydantic import field_validator
from pydantic import model_validator

T = TypeVar("T", bound=BaseModel)


class SourceAndExecutable(BaseModel):
    """
    Subtype for /api/harness-source-and-executable
    """

    executable: str
    source: str


class HarnessSourceAndExecutableResponse(BaseModel):
    """
    BaseModel for /api/harness-source-and-executable
    """

    pairs: list[SourceAndExecutable]
    result: str


class FunctionEntry(BaseModel):
    """
    Subtype for /api/all-functions
    """

    function_filename: str
    function_name: str
    function_signature: str
    is_reached: bool
    project: str
    raw_function_name: str
    reached_by_fuzzers: list[str]
    return_type: str | None
    source_line_begin: int | None = None
    source_line_end: int | None = None
    source_line: int | None = None

    @model_validator(mode="after")
    def validate_line_fields(self) -> Any:
        """
        Check because FunctionEntry must either contain start AND end lines OR just one source line
        """
        if self.source_line is None:
            if self.source_line_begin is None or self.source_line_end is None:
                raise ValueError(
                    "Either 'source_line' OR both 'source_line_begin' and 'source_line_end' must be present."
                )
        return self

    @field_validator("source_line_begin", "source_line_end", "source_line", mode="before")
    @classmethod
    def null_to_none(cls, v: int | None) -> int | None:
        """
        API sometimes returns null instead of int or None -> we want to treat it as None
        """
        if v in ("null", ""):
            return None
        return v


class AllFunctionsResponse(BaseModel):
    """
    BaseModel for /api/all-functions
    """

    functions: list[FunctionEntry]


def read_json_from_file(filepath: Path) -> Any:
    """
    Reads and returns JSON data from the specified file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        The data parsed from the JSON file (typically a dict or list),
        None for empty files or an Exception on failure.
    """
    try:
        with filepath.open(encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                if filepath.stat().st_size == 0:  # Empty file check
                    return None
                raise ValueError(f"Invalid JSON in {filepath}: {e}") from e
    except OSError as e:
        raise OSError(f"Failed to read JSON from {filepath}: {e}") from e


def write_json_to_file(data: Any, filepath: Path) -> None:
    """
    Writes the provided data to a JSON file at the specified path.

    Args:
        data: The data (usually a dict or list) to be written to the file.
        filepath: The path where the JSON file should be written.
    """
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        raise OSError(f"Failed to write JSON to {filepath}: {e}") from e


def fetch_and_cache_api_data(
    url: str, params: dict, filepath: Path, data_extractor: Callable[[dict], T], log: Logger, retries: int = 6
) -> T | None:
    """
    Fetches API data with retries and caches it in a file. If the file already exists, data is read from it.

    Args:
        url: API endpoint.
        params: Query parameters for the API.
        filepath: Path to the file for caching.
        data_extractor: Function to extract relevant data from the JSON response.
        retries: Number of retries for the API call.

    Returns:
        Extracted data from the API response or cache.
    """
    # Read API response data from file if we already received the information before
    if filepath.exists():
        log.info(f"Reading cached data from {filepath} instead of resending API request")
        response_data = read_json_from_file(filepath)
        if response_data is not None:
            return data_extractor(response_data)
        log.warning(f"Cached file {filepath} is invalid. Sending API request.")

    # no saved data available - send API request
    api_data = fetch_api_data(url, params, data_extractor, log, retries)
    if api_data is not None:
        write_json_to_file(api_data.model_dump(), filepath)
        return api_data
    return None


def fetch_api_data(
    url: str, params: dict, data_extractor: Callable[[dict], T], log: Logger, retries: int = 6
) -> T | None:
    """
    Fetches API data with retries.

    Args:
        url: API endpoint.
        params: Query parameters for the API.
        data_extractor: Function to extract relevant data from the JSON response.
        retries: Number of retries for the API call.

    Returns:
        Extracted data from the API response.
    """
    headers = {"accept": "application/json"}
    log.info(f"Sending request to {url} with params {params}")
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=180)
            response.raise_for_status()
            api_data = response.json()
            return data_extractor(api_data)
        except requests.RequestException as e:
            log.warning(f"Attempt {attempt + 1} (url: {url}, params: {params}) failed: {e}")
            if attempt == retries - 1:
                log.error(f"Max retries reached for request to {url} with params {params}")
        except Exception as e:  # pylint: disable=broad-exception-caught
            log.error(f"Unexpected error processing response (url: {url}, params: {params}): {e}")
    return None
