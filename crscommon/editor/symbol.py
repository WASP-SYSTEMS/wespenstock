"""Types for working with symbols."""

from pathlib import Path

from pydantic import BaseModel
from pydantic import ConfigDict

from .lsp.lsp_types import Location


class SymbolOccurrence(BaseModel):
    """Describes occurrence of symbol in source code"""

    name: str  # symbol name
    file: Path  # file in which the symbol occurs
    line: int  # line in which the symbol occurs (lines are zero based)


class LineInSymbolDefinition(BaseModel):
    """Describes a line inside a symbol definition"""

    name: str  # symbol name
    file: Path  # file with the definition
    line: int  # line inside the symbol definition (lines are zero based)


class SymbolDescription(BaseModel):
    """Detailed description of symbol."""

    model_config = ConfigDict(frozen=True)

    name: str  # name of the symbol
    definition: str  # definition of the symbol
    location: Location  # location of the symbol
