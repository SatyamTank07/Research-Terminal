"""What did management add, expand or drop in Item 1A versus last year?"""

from app.agents.specialized.risk_analyst.modules.disclosure_changes.calculate import (
    SentenceIndex,
    build_prior_index,
    classify_disclosure_change,
    diff_risk_disclosures,
    is_preamble,
    shingles,
)
from app.agents.specialized.risk_analyst.modules.disclosure_changes.schema import DisclosurePassage, RiskDisclosureDiff

__all__ = [
    "DisclosurePassage",
    "RiskDisclosureDiff",
    "SentenceIndex",
    "build_prior_index",
    "classify_disclosure_change",
    "diff_risk_disclosures",
    "is_preamble",
    "shingles",
]
