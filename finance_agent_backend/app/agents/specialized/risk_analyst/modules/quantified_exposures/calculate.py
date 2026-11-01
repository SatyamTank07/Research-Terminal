"""Which risks does the company put a number on? Item 1A sentences that state a percentage or currency figure about
revenue, customers, charges, supply or trade, which of them are material, and which the verified risks leave uncited."""

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from app.agents.shared.filing.narrative_search import NarrativeChunkResult
from app.agents.shared.filing.sentences import normalize_text, split_sentences
from app.agents.specialized.risk_analyst.modules.disclosure_changes import is_preamble, shingles
from app.agents.specialized.risk_analyst.modules.disclosure_changes.thresholds import PASSAGE_MAX_CHARS
from app.agents.specialized.risk_analyst.modules.quantified_exposures.schema import QuantifiedDisclosure
from app.agents.specialized.risk_analyst.modules.quantified_exposures.thresholds import (
    COVERAGE_CONTAINMENT,
    MAX_QUANTIFIED_DISCLOSURES,
    MAX_REPAIR_DISCLOSURES,
    MAX_UNCOVERED_LISTED,
)

QUANTIFIED_FIGURE = re.compile(r"\d[\d,]*(?:\.\d+)?\s?%|[$€£]\s?\d")
_QUANTIFIED_TOPIC = re.compile(
    r"revenue|sales|customer|charge|inventor|purchase\s+obligation|impairment|write-?(?:down|off)"
    r"|tariff|backlog|supply|capacity|export|licens",
    re.IGNORECASE,
)
_MATERIAL_DISCLOSURE = re.compile(
    r"customer|revenue|net\s+sales|charge|write-?(?:down|off)|impairment|purchase\s+obligation|inventor",
    re.IGNORECASE,
)
_STATUTORY_MAXIMUM = re.compile(r"penalt|fines?\b|whichever\s+is\s+greater|up\s+to\s+[$€£]?\d", re.IGNORECASE)


@dataclass(frozen=True)
class QuantifiedCoverage:
    """Checklist sentences no verified risk cites, the ones listed in data quality, and the note (if any)."""

    uncovered: Tuple[QuantifiedDisclosure, ...]
    listed: Tuple[str, ...]
    note: Optional[str]


def extract_quantified_disclosures(
    chunks: Sequence[NarrativeChunkResult],
    max_items: int = MAX_QUANTIFIED_DISCLOSURES,
) -> List[QuantifiedDisclosure]:
    """Item 1A sentences that state a figure about revenue, customers, charges, supply or trade, in filing order."""
    found: List[QuantifiedDisclosure] = []
    seen = set()
    for c in chunks:
        for sentence in split_sentences(c.content):
            key = normalize_text(sentence)
            if key in seen or is_preamble(sentence):
                continue
            if QUANTIFIED_FIGURE.search(sentence) and _QUANTIFIED_TOPIC.search(sentence):
                seen.add(key)
                found.append(QuantifiedDisclosure(chunk_id=c.chunk_id, text=sentence[:PASSAGE_MAX_CHARS]))
                if len(found) >= max_items:
                    return found
    return found


def is_material_disclosure(text: str) -> bool:
    """Checklist items worth a coverage repair: realized or stated revenue / customer / charge exposure,
    excluding hypothetical statutory maximums (e.g. "penalties of up to 4% of worldwide revenue")."""
    return bool(text) and bool(_MATERIAL_DISCLOSURE.search(text)) and not _STATUTORY_MAXIMUM.search(text)


def disclosure_covered(text: str, quotes: Sequence[str]) -> bool:
    """True if any quote overlaps the disclosure sentence (shingle overlap / smaller side >= 0.5)."""
    target = shingles(text)
    if not target:
        return False
    for quote in quotes:
        q = shingles(quote)
        if q and len(target & q) / min(len(target), len(q)) >= COVERAGE_CONTAINMENT:
            return True
    return False


def check_quantified_coverage(quantified: Sequence[QuantifiedDisclosure], quotes: Sequence[str]) -> QuantifiedCoverage:
    """Checklist disclosures (stated figures) that none of the verified risks' quotes cite."""
    uncovered = tuple(d for d in quantified if not disclosure_covered(d.text, quotes))
    note = (
        f"{len(uncovered)} of {len(quantified)} quantified Item 1A disclosures are not cited by any verified risk."
        if uncovered else None
    )
    return QuantifiedCoverage(
        uncovered=uncovered, listed=tuple(d.text for d in uncovered[:MAX_UNCOVERED_LISTED]), note=note
    )


def repair_targets(uncovered: Sequence[QuantifiedDisclosure]) -> List[QuantifiedDisclosure]:
    """The uncited material figures a coverage repair asks about, at most MAX_REPAIR_DISCLOSURES."""
    return [d for d in uncovered if is_material_disclosure(d.text)][:MAX_REPAIR_DISCLOSURES]
