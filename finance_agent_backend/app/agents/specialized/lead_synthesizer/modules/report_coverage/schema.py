"""What the report can cover: the route it is written for, its layout and the run notes it discloses."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ReportCoverage:
    route: str  # the route the prose is written for (the full report when a single-topic route lost its agent)
    layout: str  # report/<layout>.j2
    notes: Tuple[str, ...]  # run notes, each once, in first-seen order
