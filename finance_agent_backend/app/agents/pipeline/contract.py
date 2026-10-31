"""Pipeline contract: what every agent declares, what it receives, and what the run records."""

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Dict, FrozenSet, List, Literal, Mapping, Optional, Tuple

from pydantic import BaseModel

StatusLine = Tuple[Optional[str], Optional[Dict[str, Any]]]


class RunNote(BaseModel):
    """A disclosure the report carries: an agent was skipped or failed, or a user parameter was applied."""

    agent: str
    kind: Literal["skipped", "failed", "parameter"]
    message: str


@dataclass(frozen=True)
class Conversation:
    """The chat turn the Supervisor resolves. `ticker` is "" and `fiscal_year` 0 when the caller named none."""

    user_query: str
    messages: Optional[List[Dict[str, str]]]
    session_state: Optional[Dict[str, Any]]
    ticker: str
    fiscal_year: int
    run_date: date
    callbacks: Optional[List[Any]] = None


@dataclass(frozen=True)
class RunContext:
    """Facts the Supervisor resolved for one research run; every agent receives the same instance."""

    ticker: str
    company_name: str
    fiscal_year: int
    available_fiscal_years: Tuple[int, ...]
    document_id: Optional[str]
    user_query: str
    route: str
    year_substituted: bool
    valuation_overrides: Mapping[str, Any]
    terminal_growth_rate: float
    valuation_date: date
    callbacks: Optional[List[Any]] = None


class UndeclaredUpstreamError(KeyError):
    """An agent read an upstream output it did not declare in needs, uses or needs_any."""


class Upstream:
    """Read-only view of the upstream outputs one agent declared, plus the run's notes so far."""

    def __init__(self, outputs: Mapping[str, Any], declared: FrozenSet[str], notes: Tuple[RunNote, ...] = ()):
        self._outputs = {key: outputs.get(key) for key in declared}
        self.notes = notes

    def get(self, key: str) -> Any:
        if key not in self._outputs:
            raise UndeclaredUpstreamError(key)
        return self._outputs[key]


@dataclass(frozen=True)
class AgentSpec:
    """An agent's door into the pipeline: what it reads, what it writes, how it runs and reports."""

    name: str
    label: str
    output_key: str
    run: Callable[[RunContext, Upstream], Any]
    milestone: Callable[[Optional[Any]], StatusLine]
    needs: FrozenSet[str] = frozenset()
    uses: FrozenSet[str] = frozenset()
    needs_any: FrozenSet[str] = frozenset()
    skip_message: str = ""
    writes_final: bool = False

    @property
    def declared(self) -> FrozenSet[str]:
        return self.needs | self.uses | self.needs_any
