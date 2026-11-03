"""The company a research turn is about, and where the ticker came from."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ResolvedCompany:
    ticker: str
    source: Literal["explicit", "extracted", "session"]
