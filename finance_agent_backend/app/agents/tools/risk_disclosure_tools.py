"""Deterministic Risk-Disclosure Toolkit (Item 1A).

Pure-Python helpers backing the evidence-first Risk Analyst:
- Year-over-year disclosure diff (sentence-level 3-word-shingle Jaccard).
- Likelihood x impact severity matrix, aggregate risk profile, materiality / boilerplate screens.
- Markdown renderers for the verified risk matrix and evidence table.

Nothing here calls an LLM, and the screening helpers never raise on malformed LLM input.
"""

import logging
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field

from app.agents.shared.filing.narrative_search import NarrativeChunkResult
from app.agents.shared.filing.sentences import normalize_text, split_sentences, split_words

logger = logging.getLogger("finance_agent.tools.risk_disclosure")

UNCHANGED_SIMILARITY = 0.6
MODIFIED_SIMILARITY = 0.3
MAX_NEW_PASSAGES = 12
MAX_REMOVED_PASSAGES = 8
PASSAGE_MAX_CHARS = 400

IMMATERIALITY_PATTERNS = re.compile(
    r"not\s+(?:currently\s+)?(?:expected|believed|anticipated|likely)\s+to\s+(?:have|result\s+in|be)\s+(?:a\s+)?material"
    r"|(?:will|would|does|did|could)\s+not\s+(?:have|result\s+in)\s+(?:a\s+)?material\s+adverse"
    r"|(?:do|does|did)\s+not\s+believe\s+[^.]{0,160}?material"
    r"|\bimmaterial\b",
    re.IGNORECASE,
)

BOILERPLATE_PATTERNS = re.compile(
    r"(?:general|adverse|unfavorable|weak|deteriorating)\s+(?:macro)?economic\s+conditions"
    r"|economic\s+(?:downturns?|slowdowns?|uncertainty|instability)"
    r"|economic\s+conditions\s+and\s+market\s+volatility"
    r"|stock\s+price\s+volatility"
    r"|volatil\w*\s+(?:of|in)\s+(?:our\s+)?(?:stock|share)\s+price"
    r"|(?:stock|share)\s+price\s+(?:may|could)\s+(?:be\s+)?(?:volatile|fluctuate)"
    r"|\bmarket\s+volatility\b"
    r"|forward-looking\s+statements",
    re.IGNORECASE,
)

QUANTIFIED_FIGURE = re.compile(r"\d[\d,]*(?:\.\d+)?\s?%|[$€£]\s?\d")
_QUANTIFIED_TOPIC = re.compile(
    r"revenue|sales|customer|charge|inventor|purchase\s+obligation|impairment|write-?(?:down|off)"
    r"|tariff|backlog|supply|capacity|export|licens",
    re.IGNORECASE,
)
REALIZED_EFFECT_PATTERNS = re.compile(
    r"\b(?:we|the\s+company)\s+(?:have\s+|has\s+|had\s+)?(?:incurred|recorded|recognized|experienced|lost)\b"
    r"|\b(?:have|has|had)\s+(?:adversely\s+|negatively\s+|materially\s+)?(?:affected|impacted|harmed)\b"
    r"|\bresulted\s+in\b|\bwrite-?(?:down|off)s?\b",
    re.IGNORECASE,
)
MAX_QUANTIFIED_DISCLOSURES = 15
COVERAGE_CONTAINMENT = 0.5

_PREAMBLE_PATTERNS = re.compile(
    r"not\s+exhaustive|forward-looking|in\s+addition\s+to\s+the\s+other\s+information"
    r"|should\s+be\s+carefully\s+considered|past\s+financial\s+performance\s+should\s+not",
    re.IGNORECASE,
)



# ==============================================================================
# 1. Schemas
# ==============================================================================
class DisclosurePassage(BaseModel):
    """A contiguous run of new (current year) or removed (prior year) Item 1A sentences."""

    chunk_id: str = Field(..., description="Chunk the passage came from (current filing for new, prior filing for removed)")
    text: str = Field(..., description="Passage text, trimmed")
    similarity: float = Field(..., description="Best shingle-Jaccard similarity to the other year's sentences")


class QuantifiedDisclosure(BaseModel):
    """An Item 1A sentence stating a magnitude (revenue share, charge, regional share...)."""

    chunk_id: str
    text: str


class RiskDisclosureDiff(BaseModel):
    """Sentence-level year-over-year comparison of Item 1A disclosures."""

    prior_fiscal_year: int
    current_sentences: int
    prior_sentences: int
    unchanged_sentences: int
    modified_sentences: int
    new_sentences: int
    removed_sentences: int
    change_ratio_pct: float = Field(..., description="(new + modified) / current sentences, in percent")
    new_passages: List[DisclosurePassage] = Field(default_factory=list)
    removed_passages: List[DisclosurePassage] = Field(default_factory=list)


# ==============================================================================
# 3. Sentence shingles
# ==============================================================================
def _shingles(text: str, n: int = 3) -> Set[Tuple[str, ...]]:
    words = split_words(text)
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


class SentenceIndex:
    """Inverted shingle index over a set of sentences for fast best-match Jaccard lookups."""

    def __init__(self, sentences: Iterable[str]):
        self.shingles: List[Set[Tuple[str, ...]]] = []
        self._postings: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
        for idx, sentence in enumerate(sentences):
            sh = _shingles(sentence)
            self.shingles.append(sh)
            for s in sh:
                self._postings[s].append(idx)

    def __len__(self) -> int:
        return len(self.shingles)

    def best_similarity(self, text: str) -> float:
        query = _shingles(text)
        if not query:
            return 0.0
        overlap: Dict[int, int] = defaultdict(int)
        for s in query:
            for idx in self._postings.get(s, ()):
                overlap[idx] += 1
        best = 0.0
        for idx, inter in overlap.items():
            union = len(query) + len(self.shingles[idx]) - inter
            if union:
                best = max(best, inter / union)
        return best

    def containment(self, text: str) -> float:
        """Share of `text`'s shingles found anywhere in the index (robust to partial-sentence quotes)."""
        query = _shingles(text)
        if not query:
            return 0.0
        return sum(1 for s in query if s in self._postings) / len(query)


def _chunk_sentences(chunks: Sequence[NarrativeChunkResult]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for c in chunks:
        for s in split_sentences(c.content):
            if not _PREAMBLE_PATTERNS.search(s):
                out.append((c.chunk_id, s))
    return out


def _group_passages(items: List[Tuple[str, str, float]], cap: int) -> List[DisclosurePassage]:
    """Merges consecutive same-chunk sentences, keeps the `cap` longest passages, in filing order."""
    passages: List[DisclosurePassage] = []
    for chunk_id, sentence, sim in items:
        last = passages[-1] if passages else None
        if last and last.chunk_id == chunk_id and len(last.text) + len(sentence) + 1 <= PASSAGE_MAX_CHARS:
            last.text = f"{last.text} {sentence}"
            last.similarity = round(max(last.similarity, sim), 3)
        else:
            passages.append(DisclosurePassage(chunk_id=chunk_id, text=sentence[:PASSAGE_MAX_CHARS], similarity=round(sim, 3)))
    keep = sorted(range(len(passages)), key=lambda i: len(passages[i].text), reverse=True)[:cap]
    return [passages[i] for i in sorted(keep)]


def build_prior_index(prior_chunks: Sequence[NarrativeChunkResult]) -> SentenceIndex:
    return SentenceIndex(s for _, s in _chunk_sentences(prior_chunks))


def diff_risk_disclosures(
    current_chunks: Sequence[NarrativeChunkResult],
    prior_chunks: Sequence[NarrativeChunkResult],
    prior_fiscal_year: int,
) -> RiskDisclosureDiff:
    """Classifies each current-year sentence as unchanged / modified / new versus the prior filing."""
    current = _chunk_sentences(current_chunks)
    prior = _chunk_sentences(prior_chunks)
    prior_index = SentenceIndex(s for _, s in prior)
    current_index = SentenceIndex(s for _, s in current)

    counts = {"unchanged": 0, "modified": 0, "new": 0}
    new_items: List[Tuple[str, str, float]] = []
    for chunk_id, sentence in current:
        sim = prior_index.best_similarity(sentence)
        if sim >= UNCHANGED_SIMILARITY:
            counts["unchanged"] += 1
        elif sim >= MODIFIED_SIMILARITY:
            counts["modified"] += 1
        else:
            counts["new"] += 1
            new_items.append((chunk_id, sentence, sim))

    removed_items = [
        (chunk_id, sentence, sim)
        for chunk_id, sentence in prior
        if (sim := current_index.best_similarity(sentence)) < MODIFIED_SIMILARITY
    ]

    n_current = len(current)
    return RiskDisclosureDiff(
        prior_fiscal_year=prior_fiscal_year,
        current_sentences=n_current,
        prior_sentences=len(prior),
        unchanged_sentences=counts["unchanged"],
        modified_sentences=counts["modified"],
        new_sentences=counts["new"],
        removed_sentences=len(removed_items),
        change_ratio_pct=round(100.0 * (counts["new"] + counts["modified"]) / n_current, 1) if n_current else 0.0,
        new_passages=_group_passages(new_items, MAX_NEW_PASSAGES),
        removed_passages=_group_passages(removed_items, MAX_REMOVED_PASSAGES),
    )


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
            if key in seen or _PREAMBLE_PATTERNS.search(sentence):
                continue
            if QUANTIFIED_FIGURE.search(sentence) and _QUANTIFIED_TOPIC.search(sentence):
                seen.add(key)
                found.append(QuantifiedDisclosure(chunk_id=c.chunk_id, text=sentence[:PASSAGE_MAX_CHARS]))
                if len(found) >= max_items:
                    return found
    return found


_MATERIAL_DISCLOSURE = re.compile(
    r"customer|revenue|net\s+sales|charge|write-?(?:down|off)|impairment|purchase\s+obligation|inventor",
    re.IGNORECASE,
)
_STATUTORY_MAXIMUM = re.compile(r"penalt|fines?\b|whichever\s+is\s+greater|up\s+to\s+[$€£]?\d", re.IGNORECASE)


def is_material_disclosure(text: str) -> bool:
    """Checklist items worth a coverage repair: realized or stated revenue / customer / charge exposure,
    excluding hypothetical statutory maximums (e.g. "penalties of up to 4% of worldwide revenue")."""
    return bool(text) and bool(_MATERIAL_DISCLOSURE.search(text)) and not _STATUTORY_MAXIMUM.search(text)


def disclosure_covered(text: str, quotes: Sequence[str]) -> bool:
    """True if any quote overlaps the disclosure sentence (shingle overlap / smaller side >= 0.5)."""
    target = _shingles(text)
    if not target:
        return False
    for quote in quotes:
        q = _shingles(quote)
        if q and len(target & q) / min(len(target), len(q)) >= COVERAGE_CONTAINMENT:
            return True
    return False


def substantiates_high_impact(quotes: Sequence[str], quantified_exposure: Optional[str]) -> bool:
    """High impact needs a filing-stated magnitude or a realized (not hypothetical) effect in the quotes."""
    if quantified_exposure:
        return True
    return any(QUANTIFIED_FIGURE.search(q) or REALIZED_EFFECT_PATTERNS.search(q) for q in quotes if q)


def classify_disclosure_change(quotes: Sequence[str], prior_index: Optional[SentenceIndex]) -> str:
    """New if every quote is new vs the prior filing, Expanded if any is new/modified, else Unchanged.

    Quotes are often sentence fragments, so novelty uses shingle containment in the prior filing
    rather than whole-sentence Jaccard.
    """
    if prior_index is None or not len(prior_index) or not quotes:
        return "Not Assessed"
    sims = [prior_index.containment(q) for q in quotes]
    if all(s < MODIFIED_SIMILARITY for s in sims):
        return "New"
    if any(s < UNCHANGED_SIMILARITY for s in sims):
        return "Expanded"
    return "Unchanged"


# ==============================================================================
# 4. Materiality screens
# ==============================================================================
def states_immaterial(text: str) -> bool:
    return bool(text) and bool(IMMATERIALITY_PATTERNS.search(text))


def is_boilerplate(text: str) -> bool:
    return bool(text) and bool(BOILERPLATE_PATTERNS.search(text))


# ==============================================================================
# 5. Profile
# ==============================================================================
def derive_overall_profile(severities: Sequence[str]) -> str:
    """Aggregate rating; "Not Assessed" when no verified risk exists (never a reassuring "Low")."""
    if not severities:
        return "Not Assessed"
    severe = sum(1 for s in severities if s == "Severe")
    moderate = sum(1 for s in severities if s == "Moderate")
    if severe >= 2:
        return "High"
    if severe == 1 or moderate >= 3:
        return "Moderate"
    return "Low"


# ==============================================================================
# 6. Markdown renderers
# ==============================================================================
def _cell(text: Any) -> str:
    return str(text if text not in (None, "") else "—").replace("|", "/").replace("\n", " ")


def render_risk_matrix_markdown(risks: Sequence[Any]) -> str:
    """Renders the verified likelihood x impact risk matrix (accepts RiskItem models or dicts)."""
    if not risks:
        return "_No risks survived evidence verification against the filing._"
    get = (lambda r, k: r.get(k)) if isinstance(risks[0], dict) else getattr
    lines = [
        "| # | Risk | Category | Likelihood | Impact | Severity | Model line(s) hit | vs prior year |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :---: |",
    ]
    for r in risks:
        transmission = get(r, "financial_transmission") or []
        lines.append(
            f"| {_cell(get(r, 'risk_id'))} | {_cell(get(r, 'risk_title'))} | {_cell(get(r, 'risk_category'))} | "
            f"{_cell(get(r, 'likelihood'))} | {_cell(get(r, 'impact'))} | **{_cell(get(r, 'severity'))}** | "
            f"{_cell(', '.join(transmission))} | {_cell(get(r, 'disclosure_change'))} |"
        )
    return "\n".join(lines)


def render_risk_evidence_table(citations: Sequence[Dict[str, Any]]) -> str:
    if not citations:
        return "_No verified filing evidence._"
    lines = ["| Chunk ID | Section | Breadcrumb |", "| :--- | :--- | :--- |"]
    for c in citations:
        lines.append(f"| `{_cell(c.get('chunk_id'))}` | {_cell(c.get('item'))} | {_cell(c.get('breadcrumb'))} |")
    return "\n".join(lines)


__all__ = [
    "DisclosurePassage",
    "QuantifiedDisclosure",
    "RiskDisclosureDiff",
    "SentenceIndex",
    "extract_quantified_disclosures",
    "disclosure_covered",
    "is_material_disclosure",
    "substantiates_high_impact",
    "QUANTIFIED_FIGURE",
    "build_prior_index",
    "diff_risk_disclosures",
    "classify_disclosure_change",
    "states_immaterial",
    "is_boilerplate",
    "derive_overall_profile",
    "render_risk_matrix_markdown",
    "render_risk_evidence_table",
]
