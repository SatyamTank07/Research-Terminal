"""Deterministic Risk-Disclosure Toolkit (Item 1A).

Pure-Python helpers backing the evidence-first Risk Analyst:
- Full-section retrieval of a filing's Item 1A narrative in filing order (exact item match).
- Prompt context assembly under a character budget.
- Year-over-year disclosure diff (sentence-level 3-word-shingle Jaccard).
- Verbatim quote verification and numeric-claim support checks against retrieved chunks.
- Likelihood x impact severity matrix, aggregate risk profile, materiality / boilerplate screens.
- Markdown renderers for the verified risk matrix and evidence table.

Nothing here calls an LLM, and the screening helpers never raise on malformed LLM input.
"""

import logging
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agents.tools.rag_narrative_tools import NarrativeChunkResult

logger = logging.getLogger("finance_agent.tools.risk_disclosure")

RISK_SECTION_ITEM = "Item 1A"
DEFAULT_CHAR_BUDGET = 160_000
BUDGETED_CHUNK_LEAD_CHARS = 600
MAX_SUMMARY_CHUNKS = 6

UNCHANGED_SIMILARITY = 0.6
MODIFIED_SIMILARITY = 0.3
MIN_SENTENCE_CHARS = 60
MAX_NEW_PASSAGES = 12
MAX_REMOVED_PASSAGES = 8
PASSAGE_MAX_CHARS = 400

QUOTE_FUZZY_COVERAGE = 0.85
MIN_QUOTE_WORDS = 4

LEVELS = ("High", "Medium", "Low")
SEVERITY_ORDER = {"Severe": 0, "Moderate": 1, "Low": 2}
LEVEL_ORDER = {"High": 0, "Medium": 1, "Low": 2}

_SEVERITY_MATRIX = {
    ("High", "High"): "Severe",
    ("Medium", "High"): "Severe",
    ("Low", "High"): "Moderate",
    ("High", "Medium"): "Moderate",
    ("Medium", "Medium"): "Moderate",
    ("Low", "Medium"): "Low",
    ("High", "Low"): "Low",
    ("Medium", "Low"): "Low",
    ("Low", "Low"): "Low",
}

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

_SENTENCE_SPLIT = re.compile(r"(?<=[.;:!?])\s+(?=[A-Z•\"“(])|\s*•\s*")
_WORD = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_CHAR_MAP = str.maketrans({
    "‘": "'", "’": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "‟": '"', "″": '"',
    "–": "-", "—": "-", "−": "-", " ": " ",
})


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
# 2. Retrieval & context assembly
# ==============================================================================
def _to_result(chunk: Any, doc: Any, item: Optional[str] = None) -> NarrativeChunkResult:
    return NarrativeChunkResult(
        chunk_id=str(chunk.id),
        document_id=str(doc.id),
        ticker=doc.ticker,
        fiscal_year=doc.fiscal_year,
        item=item or chunk.item or "Narrative",
        breadcrumb=chunk.breadcrumb,
        sub_section=chunk.sub_section,
        content=chunk.content,
        chunk_index=chunk.chunk_index,
    )


def _filing_chunks_query(db: Session, ticker: str, fiscal_year: int):
    from app.models import ANNUAL_REPORT_FILING_TYPE, Document, DocumentChunk

    return (
        db.query(DocumentChunk, Document)
        .join(Document, Document.id == DocumentChunk.document_id)
        .filter(
            Document.ticker == ticker.strip().upper(),
            Document.fiscal_year == fiscal_year,
            Document.filing_type == ANNUAL_REPORT_FILING_TYPE,
        )
    )


def fetch_section_chunks(
    ticker: str,
    fiscal_year: int,
    item: str = RISK_SECTION_ITEM,
    db: Optional[Session] = None,
) -> List[NarrativeChunkResult]:
    """Returns every narrative chunk stored under `item` for the 10-K, in filing order (exact item match)."""
    from app.database import SessionLocal
    from app.models import DocumentChunk

    owns_db = db is None
    if owns_db:
        db = SessionLocal()
    try:
        rows = (
            _filing_chunks_query(db, ticker, fiscal_year)
            .filter(DocumentChunk.chunk_type == "narrative", DocumentChunk.item == item)
            .order_by(DocumentChunk.chunk_index.asc())
            .all()
        )
        return [_to_result(chunk, doc) for chunk, doc in rows]
    finally:
        if owns_db:
            db.close()


# Standard Form 10-K item order and title keywords, used to find section boundaries from the
# headings themselves when the stored item labels are wrong (e.g. headings rendered as tables).
TEN_K_ITEM_ORDER = [
    "1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A", "8",
    "9", "9A", "9B", "9C", "10", "11", "12", "13", "14", "15", "16",
]
TEN_K_ITEM_TITLES = {
    "1": r"business", "1A": r"risk\s+factors", "1B": r"unresolved\s+staff", "1C": r"cybersecurity",
    "2": r"properties", "3": r"legal\s+proceedings", "4": r"mine\s+safety", "5": r"market\s+for",
    "6": r"reserved|selected", "7": r"management", "7A": r"quantitative\s+and\s+qualitative",
    "8": r"financial\s+statements", "9": r"changes\s+in\s+and\s+disagreements", "9A": r"controls\s+and\s+procedures",
    "9B": r"other\s+information", "9C": r"foreign\s+jurisdictions", "10": r"directors", "11": r"executive\s+compensation",
    "12": r"security\s+ownership", "13": r"certain\s+relationships", "14": r"principal\s+account",
    "15": r"exhibits?", "16": r"form\s+10-k\s+summary",
}
HEADING_MAX_CHARS = 300
_HEADING_LEAD = re.compile(r"^[\s|#*>_:\-]*item\s*(\d{1,2}[a-c]?)\b\.?[\s|:\-—*]*", re.IGNORECASE)
_ITEM_MENTION = re.compile(r"\bitem\s*(\d{1,2}[a-c]?)\b", re.IGNORECASE)


def heading_item(content: str) -> Optional[str]:
    """Returns the 10-K item a chunk opens with ("1A", "7"...) if the chunk is a section heading.

    Short chunks (e.g. a heading rendered as a one-row table) qualify on the "Item N" lead alone;
    longer chunks must also carry the item's title right after it, so cross-references such as
    "see Item 1A" do not start a section. Tables of contents (>= 3 distinct items) never qualify.
    """
    match = _HEADING_LEAD.match(content or "")
    if not match:
        return None
    number = match.group(1).upper()
    if number not in TEN_K_ITEM_TITLES:
        return None
    if len({m.upper() for m in _ITEM_MENTION.findall(content)}) >= 3:
        return None
    if len(content) <= HEADING_MAX_CHARS:
        return number
    following = content[match.end():match.end() + 120]
    return number if re.match(rf"[\W_]*(?:{TEN_K_ITEM_TITLES[number]})", following, re.IGNORECASE) else None


def locate_section_chunks(
    ticker: str,
    fiscal_year: int,
    item: str = RISK_SECTION_ITEM,
    db: Optional[Session] = None,
) -> Tuple[List[NarrativeChunkResult], str]:
    """Finds a 10-K section's narrative chunks. Returns (chunks, located_by).

    located_by:
    - "item_label": chunks stored under the exact item label.
    - "heading_boundary": labels were wrong, so the section was cut from the filing between its own
      heading and the next later item heading; chunks are relabeled with `item`.
    - "not_found": no label and no heading for the item.
    """
    from app.database import SessionLocal
    from app.models import DocumentChunk

    target = item.split()[-1].upper()
    owns_db = db is None
    if owns_db:
        db = SessionLocal()
    try:
        labeled = fetch_section_chunks(ticker, fiscal_year, item, db=db)
        if labeled:
            return labeled, "item_label"
        if target not in TEN_K_ITEM_ORDER:
            return [], "not_found"

        rows = _filing_chunks_query(db, ticker, fiscal_year).order_by(DocumentChunk.chunk_index.asc()).all()
        target_rank = TEN_K_ITEM_ORDER.index(target)
        start = next((i for i, (chunk, _) in enumerate(rows) if heading_item(chunk.content) == target), None)
        if start is None:
            return [], "not_found"

        section: List[NarrativeChunkResult] = []
        start_chunk, start_doc = rows[start]
        if start_chunk.chunk_type == "narrative" and len(start_chunk.content) > HEADING_MAX_CHARS:
            section.append(_to_result(start_chunk, start_doc, item))
        for chunk, doc in rows[start + 1:]:
            found = heading_item(chunk.content)
            if found is not None and TEN_K_ITEM_ORDER.index(found) > target_rank:
                break
            if chunk.chunk_type == "narrative":
                section.append(_to_result(chunk, doc, item))
        return section, "heading_boundary"
    finally:
        if owns_db:
            db.close()


_NOT_PROVIDED = re.compile(
    r"not\s+applicable|smaller\s+reporting\s+compan|not\s+required\s+to\s+(?:provide|include)|\bomitted\b",
    re.IGNORECASE,
)
NOT_PROVIDED_MAX_CHARS = 2_500


def section_not_provided(chunks: Sequence[NarrativeChunkResult]) -> bool:
    """True when a located section is a short statement that no disclosure is provided
    (e.g. "Not applicable" or the smaller-reporting-company exemption)."""
    text = " ".join(c.content for c in chunks).strip()
    if not text:
        return True
    return len(text) <= NOT_PROVIDED_MAX_CHARS and bool(_NOT_PROVIDED.search(text))


def _is_summary_chunk(content: str) -> bool:
    return content.count("•") >= 3 or "summary" in content[:400].lower()


def build_section_context(
    chunks: Sequence[NarrativeChunkResult],
    char_budget: int = DEFAULT_CHAR_BUDGET,
) -> Tuple[str, str, int]:
    """Renders chunks as `[chunk:<id>] text` blocks. Returns (context, coverage_mode, chars_included).

    Within budget every chunk is included in full ("full_section"). Over budget the leading
    "Summary of Risk Factors" chunks stay whole and every other chunk is cut to its lead text,
    which carries the risk heading ("budgeted_section").
    """
    blocks = [(c.chunk_id, c.content.strip()) for c in chunks]
    total = sum(len(text) for _, text in blocks)
    if total <= char_budget:
        mode = "full_section"
        parts = [f"[chunk:{cid}] {text}" for cid, text in blocks]
    else:
        mode = "budgeted_section"
        parts = []
        used = 0
        leading_summary = True
        for idx, (cid, text) in enumerate(blocks):
            if leading_summary and idx < MAX_SUMMARY_CHUNKS and _is_summary_chunk(text):
                body = text
            else:
                leading_summary = False
                body = text[:BUDGETED_CHUNK_LEAD_CHARS] + (" [...]" if len(text) > BUDGETED_CHUNK_LEAD_CHARS else "")
            if used + len(body) > char_budget:
                break
            parts.append(f"[chunk:{cid}] {body}")
            used += len(body)
    context = "\n\n".join(parts)
    return context, mode, sum(len(p) for p in parts)


# ==============================================================================
# 3. Text normalization, sentences & shingles
# ==============================================================================
def normalize_text(text: str) -> str:
    """Lowercases, maps curly quotes/dashes to ASCII and collapses whitespace."""
    return re.sub(r"\s+", " ", (text or "").translate(_CHAR_MAP)).strip().lower()


def _words(text: str) -> List[str]:
    return _WORD.findall(normalize_text(text))


def _shingles(text: str, n: int = 3) -> Set[Tuple[str, ...]]:
    words = _words(text)
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def split_sentences(text: str) -> List[str]:
    """Splits disclosure text into sentences / bullets, keeping substantive ones only."""
    flat = re.sub(r"\s+", " ", text or "").strip()
    return [s.strip() for s in _SENTENCE_SPLIT.split(flat) if len(s.strip()) >= MIN_SENTENCE_CHARS]


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
# 4. Evidence verification
# ==============================================================================
def verify_quote(quote: str, chunk_text: str, allow_fuzzy: bool = True) -> bool:
    """True if `quote` appears in `chunk_text` verbatim (after normalization) or near-verbatim.

    Ellipses split the quote into segments that must each match. With `allow_fuzzy`, a segment
    that is not an exact substring passes if >= 85% of its words appear in a same-length window
    of the chunk.
    """
    if not isinstance(quote, str) or not isinstance(chunk_text, str):
        return False
    haystack = normalize_text(chunk_text)
    segments = [s.strip(" .,;:\"'") for s in re.split(r"\.\.\.|…|\[\.\.\.\]", normalize_text(quote))]
    segments = [s for s in segments if s]
    if not segments or sum(len(s.split()) for s in segments) < MIN_QUOTE_WORDS:
        return False
    hay_words = _WORD.findall(haystack)
    for segment in segments:
        if segment in haystack:
            continue
        if not allow_fuzzy:
            return False
        seg_words = _WORD.findall(segment)
        if not seg_words or not _fuzzy_window_match(seg_words, hay_words):
            return False
    return True


def _fuzzy_window_match(needle: List[str], hay: List[str]) -> bool:
    """Near-verbatim match: >= 85% of words in a same-length window, and every number exact."""
    size = len(needle)
    if size > len(hay):
        return False
    target = defaultdict(int)
    for w in needle:
        target[w] += 1
    numbers = [w for w in target if any(ch.isdigit() for ch in w)]
    for start in range(0, len(hay) - size + 1):
        window = defaultdict(int)
        for w in hay[start:start + size + 2]:
            window[w] += 1
        if any(window.get(n, 0) < target[n] for n in numbers):
            continue
        hit = sum(min(cnt, window.get(w, 0)) for w, cnt in target.items())
        if hit / size >= QUOTE_FUZZY_COVERAGE:
            return True
    return False


def _canonical_number(token: str) -> str:
    return token.replace(",", "").rstrip("0").rstrip(".") if "." in token else token.replace(",", "")


def numbers_supported(text: Optional[str], evidence_texts: Sequence[str]) -> bool:
    """True if every number in `text` also appears in the evidence (commas / trailing zeros ignored)."""
    if not text:
        return True
    claimed = {_canonical_number(n) for n in _NUMBER.findall(text)}
    if not claimed:
        return True
    available = {_canonical_number(n) for t in evidence_texts for n in _NUMBER.findall(t or "")}
    return claimed <= available


def states_immaterial(text: str) -> bool:
    return bool(text) and bool(IMMATERIALITY_PATTERNS.search(text))


def is_boilerplate(text: str) -> bool:
    return bool(text) and bool(BOILERPLATE_PATTERNS.search(text))


# ==============================================================================
# 5. Severity & profile
# ==============================================================================
def normalize_level(value: Any) -> Optional[str]:
    """Maps free-form LLM ratings onto High / Medium / Low (None if unrecognizable)."""
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in ("high", "severe", "critical", "very high", "elevated"):
        return "High"
    if v in ("medium", "moderate", "mid", "med"):
        return "Medium"
    if v in ("low", "minor", "limited", "very low"):
        return "Low"
    return None


def cap_level(level: str, ceiling: str) -> str:
    """Returns the lower of `level` and `ceiling` on the High > Medium > Low scale."""
    return level if LEVEL_ORDER[level] >= LEVEL_ORDER[ceiling] else ceiling


def derive_severity(likelihood: str, impact: str) -> str:
    return _SEVERITY_MATRIX.get((likelihood, impact), "Moderate")


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
    "RISK_SECTION_ITEM",
    "DisclosurePassage",
    "QuantifiedDisclosure",
    "RiskDisclosureDiff",
    "SentenceIndex",
    "extract_quantified_disclosures",
    "disclosure_covered",
    "is_material_disclosure",
    "substantiates_high_impact",
    "QUANTIFIED_FIGURE",
    "fetch_section_chunks",
    "locate_section_chunks",
    "heading_item",
    "section_not_provided",
    "TEN_K_ITEM_ORDER",
    "build_section_context",
    "build_prior_index",
    "diff_risk_disclosures",
    "classify_disclosure_change",
    "split_sentences",
    "normalize_text",
    "verify_quote",
    "numbers_supported",
    "states_immaterial",
    "is_boilerplate",
    "normalize_level",
    "cap_level",
    "derive_severity",
    "derive_overall_profile",
    "render_risk_matrix_markdown",
    "render_risk_evidence_table",
    "SEVERITY_ORDER",
    "LEVEL_ORDER",
]
