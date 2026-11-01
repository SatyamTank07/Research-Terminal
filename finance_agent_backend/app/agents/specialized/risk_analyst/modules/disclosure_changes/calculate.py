"""Year-over-year Item 1A comparison: which sentences are unchanged, modified, new or removed, and whether a
risk's quotes are new, expanded or unchanged versus the prior filing (3-word-shingle Jaccard / containment).

Until their own modules take them, this file also holds the overall profile and the markdown renderers. Nothing
here calls an LLM.
"""

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.agents.shared.filing.narrative_search import NarrativeChunkResult
from app.agents.shared.filing.sentences import split_sentences, split_words
from app.agents.specialized.risk_analyst.modules.disclosure_changes.schema import (
    DisclosurePassage,
    RiskDisclosureDiff,
)
from app.agents.specialized.risk_analyst.modules.disclosure_changes.thresholds import (
    MAX_NEW_PASSAGES,
    MAX_REMOVED_PASSAGES,
    MODIFIED_SIMILARITY,
    PASSAGE_MAX_CHARS,
    UNCHANGED_SIMILARITY,
)

_PREAMBLE_PATTERNS = re.compile(
    r"not\s+exhaustive|forward-looking|in\s+addition\s+to\s+the\s+other\s+information"
    r"|should\s+be\s+carefully\s+considered|past\s+financial\s+performance\s+should\s+not",
    re.IGNORECASE,
)


def is_preamble(sentence: str) -> bool:
    """Item 1A introduction boilerplate ("not exhaustive", "forward-looking", ...) that is not a risk."""
    return bool(_PREAMBLE_PATTERNS.search(sentence))


# ==============================================================================
# 1. Sentence shingles
# ==============================================================================
def shingles(text: str, n: int = 3) -> Set[Tuple[str, ...]]:
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
            sh = shingles(sentence)
            self.shingles.append(sh)
            for s in sh:
                self._postings[s].append(idx)

    def __len__(self) -> int:
        return len(self.shingles)

    def best_similarity(self, text: str) -> float:
        query = shingles(text)
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
        query = shingles(text)
        if not query:
            return 0.0
        return sum(1 for s in query if s in self._postings) / len(query)


def _chunk_sentences(chunks: Sequence[NarrativeChunkResult]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for c in chunks:
        for s in split_sentences(c.content):
            if not is_preamble(s):
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
# 2. Profile
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
# 3. Markdown renderers
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
    "SentenceIndex",
    "is_preamble",
    "shingles",
    "build_prior_index",
    "diff_risk_disclosures",
    "classify_disclosure_change",
    "derive_overall_profile",
    "render_risk_matrix_markdown",
    "render_risk_evidence_table",
]
