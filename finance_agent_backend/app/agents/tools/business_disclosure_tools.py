"""Deterministic Business-Disclosure Toolkit (Item 1 + segment note + revenue disaggregation).

Pure-Python helpers backing the evidence-first Business Strategist:
- Whole-filing chunk loading in filing order and heading-based section inference, so citations
  carry the true 10-K item even when the stored section labels are wrong.
- Segment-note extraction (ASC 280): revenue and the segment profit measure by reportable segment,
  multi-year, from the three common table layouts:
    * "stacked_by_segment":  a header row per segment, metric rows below it, years as columns;
    * "stacked_by_metric":   a header row per metric, segment rows below it, years as columns;
    * "segments_as_columns": segments as columns, metric rows, the year in the table title or
      in in-table "Year Ended ..." rows (one table per year is merged across chunks).
- Revenue-disaggregation extraction (ASC 606): net sales by product / service line, nested
  sub-lines (a row equal to the sum of the rows below it) and subtotals resolved, reconciled to the
  table total and to consolidated revenue.
- Unit handling (thousands / millions / billions captions, cross-checked against the income
  statement) and period columns in either order.
- Sentence screens for customer-concentration and pricing disclosures anywhere in the filing.
- Markdown renderers for the verified segment and revenue-mix tables.

Every extractor returns empty results plus notes instead of guessing when a table does not
parse cleanly or does not reconcile. Nothing here calls an LLM, and nothing raises on odd input.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agents.shared.filing.sections import TEN_K_ITEM_ORDER, heading_item
from app.agents.shared.filing.sentences import normalize_text, split_sentences
from app.services.financial_tables import (
    StatementGrid,
    StatementRow,
    markdown_to_rows,
    normalize_label,
    parse_statement_table,
    parse_value_token,
)

logger = logging.getLogger("finance_agent.tools.business_disclosure")

# Reconciliation tolerances: relative share of the reference value, plus an absolute floor in
# table units to absorb per-row rounding (each row is rounded to the table's unit).
RECONCILE_REL_TOLERANCE = 0.01
RECONCILE_ABS_UNITS_PER_ROW = 1.0
SCALE_MATCH_TOLERANCE = 0.02

MIN_SEGMENTS = 2
MIN_REVENUE_LINES = 2
GEOGRAPHIC_SHARE = 0.5
MAX_CONCENTRATION_SENTENCES = 6
MAX_PRICING_SENTENCES = 6
MAX_SENTENCE_CHARS = 420
CONTEXT_TAIL_CHARS = 500
CAPTION_TAIL_CHARS = 200  # statement unit captions sit directly above the table

UNITS_MILLIONS = "USD millions"


# ==============================================================================
# 1. Schemas
# ==============================================================================
class FilingChunk(BaseModel):
    """One stored chunk of the 10-K (narrative or table) in filing order."""

    chunk_id: str
    chunk_index: int
    chunk_type: str
    item: Optional[str] = None
    breadcrumb: Optional[str] = None
    sub_section: Optional[str] = None
    content: str = ""


class SegmentFigure(BaseModel):
    """Revenue and segment profit for one reportable segment and fiscal year (USD millions)."""

    segment_name: str
    fiscal_year: int
    revenue: float
    segment_profit: Optional[float] = None
    profit_measure: Optional[str] = None
    revenue_share_pct: Optional[float] = None
    profit_margin_pct: Optional[float] = None
    profit_share_pct: Optional[float] = None
    revenue_growth_pct: Optional[float] = None
    reportable: bool = True
    source_chunk_id: str


class RevenueLineFigure(BaseModel):
    """Net sales of one disaggregated product / service line for one fiscal year (USD millions)."""

    name: str
    fiscal_year: int
    revenue: float
    revenue_share_pct: Optional[float] = None
    revenue_growth_pct: Optional[float] = None
    parent: Optional[str] = None
    source_chunk_id: str


class DisclosureSentence(BaseModel):
    """A filing sentence surfaced by a deterministic screen, with its chunk and (inferred) item."""

    chunk_id: str
    item: Optional[str] = None
    text: str
    kind: str = Field(..., description="'revenue_concentration' | 'receivables_concentration' | 'pricing'")


class SegmentExtraction(BaseModel):
    """Result of the deterministic segment / revenue-line extraction for one filing."""

    segments: List[SegmentFigure] = Field(default_factory=list)
    revenue_lines: List[RevenueLineFigure] = Field(default_factory=list)
    segment_layout: Optional[str] = None
    segment_basis: str = "not_found"
    segment_reconciliation: str = "not_applicable"
    segment_source_chunk_ids: List[str] = Field(default_factory=list)
    revenue_lines_source_chunk_id: Optional[str] = None
    revenue_lines_reconciliation: str = "not_applicable"
    consolidated_revenue: Optional[float] = None
    consolidated_revenue_year: Optional[int] = None
    latest_fiscal_year: Optional[int] = None
    units: str = UNITS_MILLIONS
    segment_tables_found: int = 0
    revenue_tables_found: int = 0
    notes: List[str] = Field(default_factory=list)


# ==============================================================================
# 2. Filing loading & section inference
# ==============================================================================
def load_filing_chunks(ticker: str, fiscal_year: int, db: Optional[Session] = None) -> List[FilingChunk]:
    """Returns every chunk of the 10-K for `ticker` / `fiscal_year` in filing order."""
    from app.database import SessionLocal
    from app.models import ANNUAL_REPORT_FILING_TYPE, Document, DocumentChunk

    owns_db = db is None
    if owns_db:
        db = SessionLocal()
    try:
        rows = (
            db.query(DocumentChunk)
            .join(Document, Document.id == DocumentChunk.document_id)
            .filter(
                Document.ticker == ticker.strip().upper(),
                Document.fiscal_year == fiscal_year,
                Document.filing_type == ANNUAL_REPORT_FILING_TYPE,
            )
            .order_by(DocumentChunk.chunk_index.asc())
            .all()
        )
        return [
            FilingChunk(
                chunk_id=str(c.id),
                chunk_index=c.chunk_index,
                chunk_type=c.chunk_type or "narrative",
                item=c.item,
                breadcrumb=c.breadcrumb,
                sub_section=c.sub_section,
                content=c.content or "",
            )
            for c in rows
        ]
    finally:
        if owns_db:
            db.close()


def infer_section_items(chunks: Sequence[FilingChunk]) -> Dict[str, str]:
    """Maps chunk_id -> 10-K item inferred from the filing's own section headings.

    Headings only move forward through the standard item order, so cross-references and stray
    backward mentions never re-open an earlier section. Chunks before the first heading are
    mapped to "Cover Page".
    """
    mapping: Dict[str, str] = {}
    current: Optional[str] = None
    for chunk in chunks:
        found = heading_item(chunk.content)
        if found is not None and (current is None or TEN_K_ITEM_ORDER.index(found) > TEN_K_ITEM_ORDER.index(current)):
            current = found
        mapping[chunk.chunk_id] = f"Item {current}" if current else "Cover Page"
    return mapping


# ==============================================================================
# 3. Label vocabulary
# ==============================================================================
_LOSS_SUFFIX = re.compile(
    r"\s*/?\s*\((?:loss|losses|deficit)\)|\s+(?:and|or)\s+loss(?:es)?\b|\s*/\s*loss\b", re.IGNORECASE
)
_REVENUE_LABEL = re.compile(
    r"^(?:total\s+)?(?:segment\s+)?(?:net\s+)?(?:operating\s+)?(?:revenues?|sales|net\s+sales|sales\s+and\s+revenues?)"
    r"(?:\s*,\s*net|\s+net)?(?:\s+(?:from|to)\s+external\s+customers)?$"
)
_PROFIT_PATTERNS: List[Tuple[int, re.Pattern]] = [
    (0, re.compile(r"^(?:total\s+)?(?:segment\s+)?(?:operating|operations)\s+(?:income|profit|earnings|loss)$")),
    (0, re.compile(r"^(?:segment\s+)?(?:income|profit|earnings|loss)\s+from\s+operations$")),
    (0, re.compile(r"^(?:total\s+)?(?:reportable\s+)?segments?(?:'s)?\s+(?:operating\s+)?(?:income|profit|earnings|results?|margin)$")),
    (0, re.compile(r"^(?:segment\s+)?(?:adjusted\s+)?(?:ebitda|ebit|oibda)$")),
    (1, re.compile(r"^(?:segment\s+)?contribution(?:\s+(?:margin|profit|income))?$")),
    (2, re.compile(r"^(?:segment\s+)?gross\s+(?:profit|margin)$")),
]
_TOTAL_LABEL = re.compile(r"^(?:total|consolidated|combined)\b|\btotals?$|^net\s+(?:sales|revenues?)$|^(?:revenues?|sales)$")
_NON_REPORTABLE = re.compile(
    r"corporate|unallocated|eliminat|intersegment|inter-segment|reconcil|^other$|all\s+other|other\s+(?:operating\s+)?segments?|adjustments?$"
)
_INCOME_STATEMENT_ROW = re.compile(r"^net\s+(?:income|earnings|loss)|^(?:total\s+)?cost\s+of|earnings\s+per\s+share|per\s+share")
_SEGMENT_SIGNAL = re.compile(r"segment|chief\s+operating\s+decision\s+maker|\bcodm\b", re.IGNORECASE)
_REVENUE_SIGNAL = re.compile(r"revenue|net\s+sales|\bsales\b", re.IGNORECASE)
_DISAGGREGATION_SIGNAL = re.compile(
    r"disaggregat|similar\s+products|product\s+(?:and|or)\s+service|by\s+(?:product|service|category|end\s+market|source|type)"
    r"|end\s+market|revenue\s+(?:streams?|categor|by)",
    re.IGNORECASE,
)
_SINGLE_SEGMENT = re.compile(
    r"\b(?:one|single|a\s+single)\s+(?:reportable\s+|operating\s+)+segment|\bone\s+(?:reportable|operating)\s+segment",
    re.IGNORECASE,
)
_SEGMENT_SUFFIX = re.compile(r"\s+(?:reportable\s+|operating\s+)?segment$", re.IGNORECASE)
_FOOTNOTE = re.compile(r"(?:\s*\((?:\d{1,2}|[a-z])\))+\s*$")
_YEAR = re.compile(r"(?<![\d,.])((?:19|20)\d{2})(?![\d,.]\d)")

_GEOGRAPHY = re.compile(
    r"\b(?:americas?|north\s+america|south\s+america|latin\s+america|central\s+america|europe|european|emea|apac|asia"
    r"|asia[\s-]pacific|greater\s+china|china|hong\s+kong|taiwan|japan|korea|india|singapore|australia|new\s+zealand"
    r"|indonesia|malaysia|thailand|vietnam|philippines|rest\s+of\s+(?:the\s+)?world|other\s+countries|other\s+foreign"
    r"|foreign\s+countries|international|domestic|united\s+states|usa|canada|mexico|brazil|argentina|chile|colombia"
    r"|germany|france|united\s+kingdom|ireland|netherlands|switzerland|italy|spain|sweden|norway|denmark|finland"
    r"|belgium|austria|poland|russia|israel|saudi\s+arabia|united\s+arab\s+emirates|africa|middle\s+east|turkey|nordics?)\b"
    r"|(?<![a-z])u\.\s?[sk]\.?(?![a-z])",
    re.IGNORECASE,
)
_GEO_NEUTRAL = re.compile(r"^(?:all\s+)?other$|corporate|eliminat|unallocated", re.IGNORECASE)


def clean_name(label: str) -> str:
    """Display name for a table label: footnote markers and trailing ':' removed, whitespace collapsed."""
    name = " ".join((label or "").replace("’", "'").split()).strip().strip("*").strip()
    name = _FOOTNOTE.sub("", name).rstrip(":").strip()
    return name


def name_key(name: str) -> str:
    """Comparison key for matching names across tables and LLM output."""
    s = normalize_text(clean_name(name)).replace("&", " and ")
    s = _SEGMENT_SUFFIX.sub("", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _metric_key(label: str) -> str:
    return _LOSS_SUFFIX.sub("", normalize_label(clean_name(label))).strip()


def is_revenue_label(label: str) -> bool:
    return bool(_REVENUE_LABEL.match(_metric_key(label)))


def profit_rank(label: str) -> Optional[int]:
    """Rank of a segment-profit label (0 = operating income / segment profit ... 2 = gross profit)."""
    key = _metric_key(label)
    if not key or "non-operating" in key or "nonoperating" in key:
        return None
    for rank, pattern in _PROFIT_PATTERNS:
        if pattern.match(key):
            return rank
    return None


def is_total_label(label: str) -> bool:
    return bool(_TOTAL_LABEL.search(_metric_key(label)))


def is_geographic(name: str) -> bool:
    return bool(_GEOGRAPHY.search(clean_name(name)))


def _geographic_share(names: Sequence[str]) -> float:
    considered = [n for n in names if not _GEO_NEUTRAL.search(clean_name(n))]
    if not considered:
        return 0.0
    return sum(1 for n in considered if is_geographic(n)) / len(considered)


def segment_basis(names: Sequence[str]) -> str:
    """'geographic' when every reportable segment is a region, 'mixed' when some are, else 'business'."""
    considered = [n for n in names if not _GEO_NEUTRAL.search(clean_name(n))]
    geo = sum(1 for n in considered if is_geographic(n))
    if considered and geo == len(considered):
        return "geographic"
    return "mixed" if geo else "business"


# ==============================================================================
# 4. Units
# ==============================================================================
_UNIT_CAPTIONS: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"in\s+thousands(?!\s+of\s+(?:units|shares|vehicles|customers|employees|square))|\$\s*in\s+000s?\b"), 1000.0),
    (re.compile(r"in\s+millions"), 1.0),
    (re.compile(r"in\s+billions"), 0.001),
]


def scale_from_caption(text: str) -> Tuple[float, bool]:
    """Divisor converting table values to millions, and whether a unit caption was found.

    The earliest caption wins, so "(In millions, except shares, which are reflected in thousands)"
    reads as millions.
    """
    lower = (text or "").lower()
    hits = [(m.start(), divisor) for pattern, divisor in _UNIT_CAPTIONS for m in [pattern.search(lower)] if m]
    if not hits:
        return 1.0, False
    return min(hits)[1], True


def _scale_factor(value: float, reference: float) -> Optional[float]:
    """Returns the multiplier (1, 1/1000 or 1000) that makes `value` match `reference`, if any."""
    if not value or not reference:
        return None
    for factor in (1.0, 0.001, 1000.0):
        if abs(value * factor - reference) <= SCALE_MATCH_TOLERANCE * abs(reference):
            return factor
    return None


def _close(a: float, b: float, rows: int = 1) -> bool:
    tolerance = max(RECONCILE_REL_TOLERANCE * abs(b), RECONCILE_ABS_UNITS_PER_ROW * max(rows, 1))
    return abs(a - b) <= tolerance


# ==============================================================================
# 5. Table parsing helpers
# ==============================================================================
@dataclass
class _Table:
    chunk: FilingChunk
    item: Optional[str]
    context: str
    grid: StatementGrid
    raw_rows: List[List[str]]
    title: Optional[str]
    divisor: float
    caption_found: bool

    @property
    def signal_text(self) -> str:
        return f"{self.title or ''} {self.chunk.sub_section or ''} {self.context}"


@dataclass
class _SegmentCandidate:
    layout: str
    tables: List[_Table]
    # entity display name -> year -> {"revenue", "profit", "profit_label", "profit_rank"}
    data: Dict[str, Dict[int, Dict[str, Any]]] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)

    def put(self, entity: str, year: int, key: str, value: Any) -> None:
        if entity not in self.data:
            self.data[entity] = {}
            self.order.append(entity)
        self.data[entity].setdefault(year, {})[key] = value


def _numeric_tokens(cells: Iterable[str]) -> List[float]:
    out: List[float] = []
    for cell in cells:
        tok = parse_value_token(cell)
        if isinstance(tok, float):
            out.append(tok)
    return out


def _complete(row: StatementRow, grid: StatementGrid) -> bool:
    """Strict: the row carries exactly one value per detected period (no change / % columns)."""
    return bool(grid.periods) and len(row.tokens) == len(grid.periods) and len(row.values) == len(grid.periods)


def _build_tables(chunks: Sequence[FilingChunk], items: Dict[str, str]) -> List[_Table]:
    tables: List[_Table] = []
    last_narrative = ""
    for chunk in chunks:
        if chunk.chunk_type != "table":
            last_narrative = chunk.content or ""
            continue
        content = chunk.content or ""
        low = content.lower()
        if "exhibit" in low[:300] or "index to consolidated financial statements" in low:
            continue
        try:
            grid = parse_statement_table(content, chunk_id=chunk.chunk_id)
            title, raw_rows = markdown_to_rows(content)
        except Exception as e:  # malformed markdown must never abort the extraction
            logger.debug(f"Table {chunk.chunk_id} unparseable: {e}")
            continue
        context = last_narrative[-CONTEXT_TAIL_CHARS:]
        divisor, caption = scale_from_caption(f"{title or ''} {content[:600]} {context[-CAPTION_TAIL_CHARS:]}")
        tables.append(
            _Table(
                chunk=chunk,
                item=items.get(chunk.chunk_id) or chunk.item,
                context=context,
                grid=grid,
                raw_rows=raw_rows,
                title=title,
                divisor=divisor,
                caption_found=caption,
            )
        )
    return tables


# ==============================================================================
# 6. Consolidated revenue (income statement)
# ==============================================================================
def find_consolidated_revenue(tables: Sequence[_Table]) -> Tuple[Dict[int, float], Optional[_Table]]:
    """Consolidated revenue by year (USD millions) from the income statement, and its table.

    The income statement is the table that carries a revenue total row AND a net income row
    with clean period columns; the candidate with the most such rows wins.
    """
    best: Optional[Tuple[int, Dict[int, float], _Table]] = None
    for t in tables:
        grid = t.grid
        if len(grid.periods) < 2:
            continue
        rows = [r for r in grid.value_rows if _complete(r, grid)]
        has_net_income = any(re.match(r"^net\s+(?:income|earnings|loss)", r.norm_label) for r in rows)
        if not has_net_income:
            continue
        revenue_rows = [r for r in rows if is_revenue_label(r.label)]
        if not revenue_rows:
            continue
        # The revenue total is the largest revenue-labelled row (covers "Total net sales" after components).
        revenue_row = max(revenue_rows, key=lambda r: abs(r.values.get(max(grid.periods), 0.0)))
        score = len(rows) + (20 if any("per share" in r.norm_label or "diluted" in r.norm_label for r in rows) else 0)
        values = {y: v / t.divisor for y, v in revenue_row.values.items()}
        if best is None or score > best[0]:
            best = (score, values, t)
    if best is None:
        return {}, None
    return best[1], best[2]


# ==============================================================================
# 7. Segment note extraction
# ==============================================================================
def _entity(label: str) -> str:
    return _SEGMENT_SUFFIX.sub("", clean_name(label)).strip()


def _record_metrics(cand: _SegmentCandidate, entity: str, label: str, values: Dict[int, float]) -> None:
    if is_revenue_label(label):
        for year, v in values.items():
            if "revenue" not in cand.data.get(entity, {}).get(year, {}):
                cand.put(entity, year, "revenue", v)
        return
    rank = profit_rank(label)
    if rank is None:
        return
    for year, v in values.items():
        current = cand.data.get(entity, {}).get(year, {})
        if "profit_rank" not in current or rank < current["profit_rank"]:
            cand.put(entity, year, "profit", v)
            cand.put(entity, year, "profit_label", _LOSS_SUFFIX.sub("", clean_name(label)).strip())
            cand.put(entity, year, "profit_rank", rank)


def _parse_stacked(table: _Table) -> List[_SegmentCandidate]:
    """Years-as-columns layouts: one header row per segment (or per metric) with value rows below."""
    grid = table.grid
    if not grid.periods:
        return []
    by_segment = _SegmentCandidate(layout="stacked_by_segment", tables=[table])
    by_metric = _SegmentCandidate(layout="stacked_by_metric", tables=[table])
    header: Optional[str] = None
    for row in grid.rows:
        if row.is_header:
            header = row.label
            continue
        if header is None or not _complete(row, grid):
            continue
        # Layout A: header = segment, row = metric.
        if is_revenue_label(row.label) or profit_rank(row.label) is not None:
            _record_metrics(by_segment, _entity(header), row.label, row.values)
        # Layout B: header = metric, row = segment.
        elif is_revenue_label(header) or profit_rank(header) is not None:
            _record_metrics(by_metric, _entity(row.label), header, row.values)
    return [c for c in (by_segment, by_metric) if c.data]


def _parse_columns(table: _Table) -> Optional[_SegmentCandidate]:
    """Segments-as-columns layout; the year comes from the title or in-table 'Year Ended ...' rows."""
    if not table.raw_rows:
        return None
    header = table.raw_rows[0]
    entities = [clean_name(c) for c in header[1:] if c and c.strip()]
    if len(entities) < MIN_SEGMENTS or any(_YEAR.search(e) for e in entities):
        return None
    if any(parse_value_token(e) is not None and isinstance(parse_value_token(e), float) for e in entities):
        return None
    title_years = _YEAR.findall(f"{table.title or ''}")
    year: Optional[int] = int(title_years[-1]) if title_years else None
    cand = _SegmentCandidate(layout="segments_as_columns", tables=[table])
    for row in table.raw_rows[1:]:
        if not row:
            continue
        label = row[0] or ""
        tokens = _numeric_tokens(row[1:])
        if not tokens:
            years = _YEAR.findall(" ".join(c for c in row if c))
            if years:
                year = int(years[-1])
            continue
        if year is None or len(tokens) != len(entities):
            continue
        if is_revenue_label(label) or profit_rank(label) is not None:
            for entity, value in zip(entities, tokens):
                _record_metrics(cand, _entity(entity), label, {year: value})
    return cand if cand.data else None


def _merge_column_candidates(cands: List[_SegmentCandidate]) -> List[_SegmentCandidate]:
    """Merges per-year segments-as-columns tables that share the same set of segments."""
    merged: Dict[Tuple[str, ...], _SegmentCandidate] = {}
    out: List[_SegmentCandidate] = []
    for cand in cands:
        key = tuple(sorted(name_key(e) for e in cand.order))
        base = merged.get(key)
        if base is None:
            merged[key] = cand
            out.append(cand)
            continue
        base.tables.extend(cand.tables)
        for entity, years in cand.data.items():
            target = next((e for e in base.order if name_key(e) == name_key(entity)), entity)
            for year, metrics in years.items():
                if year not in base.data.get(target, {}):
                    for k, v in metrics.items():
                        base.put(target, year, k, v)
    return out


@dataclass
class _SegmentResult:
    candidate: _SegmentCandidate
    reportable: List[str]
    others: List[str]
    total: Optional[str]
    latest: int
    years: List[int]
    divisor: float
    reconciliation: str
    score: float


def _evaluate_segment_candidate(
    cand: _SegmentCandidate,
    consolidated: Dict[int, float],
    notes: List[str],
) -> Optional[_SegmentResult]:
    totals = [e for e in cand.order if is_total_label(e)]
    members = [e for e in cand.order if e not in totals]
    reportable = [e for e in members if not _NON_REPORTABLE.search(normalize_text(e))]
    others = [e for e in members if e not in reportable]
    if len(reportable) < MIN_SEGMENTS:
        return None
    years = sorted(
        {y for y in set().union(*(cand.data[e].keys() for e in reportable))
         if all("revenue" in cand.data[e].get(y, {}) for e in reportable)},
        reverse=True,
    )
    if not years:
        return None
    latest = years[0]
    if not any("profit" in cand.data[e].get(latest, {}) for e in reportable):
        return None  # revenue-only: a revenue breakdown, not a segment note

    total = totals[0] if totals else None
    member_sum = sum(cand.data[e].get(latest, {}).get("revenue", 0.0) for e in members)
    divisor = cand.tables[0].divisor
    reconciliation = "unreconciled"
    total_rev = cand.data[total].get(latest, {}).get("revenue") if total else None
    if total_rev is not None:
        if not _close(member_sum, total_rev, len(members)):
            notes.append(
                f"Segment table (chunk {cand.tables[0].chunk.chunk_id}) rejected: segment revenue "
                f"{member_sum:,.0f} does not reconcile to its total {total_rev:,.0f}."
            )
            return None
        reconciliation = "in_table_total"
    reference = consolidated.get(latest)
    if reference:
        factor = _scale_factor((total_rev if total_rev is not None else member_sum) / divisor, reference)
        if factor is None:
            if total_rev is None:
                notes.append(
                    f"Segment table (chunk {cand.tables[0].chunk.chunk_id}) rejected: segment revenue does not "
                    "reconcile to consolidated revenue and the table has no total."
                )
                return None
            notes.append("Segment revenue total differs from consolidated revenue (intersegment or unallocated items).")
        else:
            divisor = divisor / factor
            if reconciliation == "unreconciled":
                reconciliation = "income_statement"

    signal = " ".join(t.signal_text for t in cand.tables) + " " + " ".join(cand.order)
    score = (
        len(years) * 2
        + (6 if _SEGMENT_SIGNAL.search(signal) else 0)
        + (4 if reconciliation != "unreconciled" else 0)
        + (2 if all("profit" in cand.data[e].get(latest, {}) for e in reportable) else 0)
        + (2 if any((t.item or "") in ("Item 8", "Item 15") for t in cand.tables) else 0)
        + latest / 10_000.0
    )
    return _SegmentResult(cand, reportable, others, total, latest, years, divisor, reconciliation, score)


def _pct(part: Optional[float], whole: Optional[float]) -> Optional[float]:
    if part is None or not whole:
        return None
    return round(100.0 * part / whole, 1)


def _growth(current: Optional[float], prior: Optional[float]) -> Optional[float]:
    if current is None or prior is None or prior <= 0:
        return None
    return round(100.0 * (current - prior) / prior, 1)


def _segment_figures(result: _SegmentResult) -> List[SegmentFigure]:
    cand = result.candidate
    figures: List[SegmentFigure] = []
    for year in result.years:
        def rev(e: str) -> Optional[float]:
            v = cand.data[e].get(year, {}).get("revenue")
            return None if v is None else v / result.divisor

        members = result.reportable + result.others
        total_rev = rev(result.total) if result.total else None
        if total_rev is None:
            total_rev = sum(rev(e) or 0.0 for e in members) or None
        profits = [cand.data[e].get(year, {}).get("profit") for e in result.reportable]
        profit_pool = sum(p / result.divisor for p in profits if p is not None and p > 0)
        for entity in members:
            metrics = cand.data[entity].get(year, {})
            revenue = rev(entity)
            if revenue is None:
                continue
            profit = metrics.get("profit")
            profit = None if profit is None else profit / result.divisor
            prior = cand.data[entity].get(year - 1, {}).get("revenue")
            reportable = entity in result.reportable
            figures.append(
                SegmentFigure(
                    segment_name=entity,
                    fiscal_year=year,
                    revenue=round(revenue, 2),
                    segment_profit=None if profit is None else round(profit, 2),
                    profit_measure=metrics.get("profit_label"),
                    revenue_share_pct=_pct(revenue, total_rev),
                    profit_margin_pct=_pct(profit, revenue) if revenue else None,
                    profit_share_pct=_pct(profit, profit_pool) if reportable and profit is not None and profit_pool else None,
                    revenue_growth_pct=_growth(revenue, None if prior is None else prior / result.divisor),
                    reportable=reportable,
                    source_chunk_id=cand.tables[0].chunk.chunk_id,
                )
            )
    return figures


# ==============================================================================
# 8. Revenue disaggregation extraction
# ==============================================================================
@dataclass
class _RevenueResult:
    table: _Table
    lines: List[Tuple[str, Dict[int, float], Optional[str]]]  # (name, values, parent)
    total: Dict[int, float]
    divisor: float
    reconciliation: str
    score: float
    latest: int


def _sum_values(rows: Sequence[Dict[int, float]], periods: Sequence[int]) -> Dict[int, float]:
    return {p: sum(r.get(p, 0.0) for r in rows) for p in periods}


def _values_match(a: Dict[int, float], b: Dict[int, float], periods: Sequence[int], rows: int) -> bool:
    return all(p in a and p in b and _close(a[p], b[p], rows) for p in periods)


def _resolve_nesting(rows: List[StatementRow], periods: Sequence[int]) -> List[Tuple[StatementRow, Optional[str]]]:
    """Marks rows that are components of the row above them (parent == sum of the next k >= 2 rows)."""
    out: List[Tuple[StatementRow, Optional[str]]] = []
    i = 0
    while i < len(rows):
        parent = rows[i]
        out.append((parent, None))
        consumed = 0
        for k in range(len(rows) - i - 1, 1, -1):
            children = rows[i + 1:i + 1 + k]
            if _values_match(_sum_values([c.values for c in children], periods), parent.values, periods, k):
                out.extend((c, clean_name(parent.label)) for c in children)
                consumed = k
                break
        i += 1 + consumed
    return out


def _parse_revenue_block(table: _Table) -> Optional[Tuple[List[Tuple[StatementRow, Optional[str]]], Dict[int, float], bool]]:
    """Finds the first revenue block of a years-as-columns table and reconciles its lines to a total row.

    Returns (lines with parent, total values, is_income_statement) or None.
    """
    grid = table.grid
    periods = grid.periods
    if not periods:
        return None
    block: List[StatementRow] = []
    header_label = ""
    for row in grid.rows:
        if row.is_header:
            if block:
                break
            header_label = row.label
            continue
        block.append(row)
    if len(block) < MIN_REVENUE_LINES + 1:
        return None

    lines: List[StatementRow] = []
    best: Optional[Tuple[List[StatementRow], StatementRow]] = None
    for row in block:
        if not _complete(row, grid):
            if best is None:
                return None  # change / % columns or a ragged row before the total: not reliable
            break
        if is_total_label(row.label):
            nested = _resolve_nesting(lines, periods)
            top = [r for r, parent in nested if parent is None]
            if len(top) >= MIN_REVENUE_LINES and _values_match(
                _sum_values([r.values for r in top], periods), row.values, periods, len(top)
            ):
                best = (list(lines), row)
            continue  # a subtotal that is not the running total stays out of the lines
        if profit_rank(row.label) is not None or _INCOME_STATEMENT_ROW.search(row.norm_label):
            break
        lines.append(row)
    if best is None:
        return None
    kept, total_row = best
    is_income_statement = any(
        _INCOME_STATEMENT_ROW.search(r.norm_label) or profit_rank(r.label) is not None for r in grid.value_rows
    )
    signal = f"{header_label} {total_row.label} {table.signal_text}"
    if not _REVENUE_SIGNAL.search(signal):
        return None
    return _resolve_nesting(kept, periods), dict(total_row.values), is_income_statement


def _evaluate_revenue_table(
    table: _Table,
    consolidated: Dict[int, float],
    segment_keys: set,
    notes: List[str],
) -> Optional[_RevenueResult]:
    parsed = _parse_revenue_block(table)
    if parsed is None:
        return None
    nested, total, is_income_statement = parsed
    latest = max(table.grid.periods)
    top_names = [clean_name(r.label) for r, parent in nested if parent is None]
    if _geographic_share(top_names) >= GEOGRAPHIC_SHARE:
        return None  # entity-wide geographic disclosure, not a product / service mix
    divisor = table.divisor
    reconciliation = "in_table_total"
    reference = consolidated.get(latest)
    if reference:
        factor = _scale_factor(total[latest] / divisor, reference)
        if factor is None:
            return None  # partial breakdown (e.g. one segment or one revenue type), not total revenue
        divisor = divisor / factor
        reconciliation = "income_statement"
    lines = [(clean_name(r.label), {p: v for p, v in r.values.items()}, parent) for r, parent in nested]
    same_as_segments = bool(segment_keys) and {name_key(n) for n in top_names} <= segment_keys
    score = (
        len(top_names)
        + 0.5 * (len(lines) - len(top_names))
        + (4 if _DISAGGREGATION_SIGNAL.search(table.signal_text) else 0)
        + (3 if reconciliation == "income_statement" else 0)
        - (6 if is_income_statement else 0)
        - (3 if same_as_segments else 0)
        + latest / 10_000.0
    )
    return _RevenueResult(table, lines, total, divisor, reconciliation, score, latest)


def _revenue_figures(result: _RevenueResult) -> List[RevenueLineFigure]:
    figures: List[RevenueLineFigure] = []
    for year in sorted(result.table.grid.periods, reverse=True):
        total = result.total.get(year)
        total = None if total is None else total / result.divisor
        for name, values, parent in result.lines:
            if year not in values:
                continue
            revenue = values[year] / result.divisor
            prior = values.get(year - 1)
            figures.append(
                RevenueLineFigure(
                    name=name,
                    fiscal_year=year,
                    revenue=round(revenue, 2),
                    revenue_share_pct=_pct(revenue, total),
                    revenue_growth_pct=_growth(revenue, None if prior is None else prior / result.divisor),
                    parent=parent,
                    source_chunk_id=result.table.chunk.chunk_id,
                )
            )
    return figures


# ==============================================================================
# 9. Orchestration
# ==============================================================================
def extract_segment_disclosures(
    chunks: Sequence[FilingChunk],
    fiscal_year: int,
    items: Optional[Dict[str, str]] = None,
) -> SegmentExtraction:
    """Extracts reportable-segment financials and revenue lines from a filing's tables.

    Never raises: any failure yields empty lists plus a note.
    """
    out = SegmentExtraction()
    try:
        _extract_into(out, chunks, fiscal_year, items or {})
    except Exception as e:  # defensive: a parser bug must not take the agent down
        logger.exception(f"Segment extraction failed: {e}")
        out.segments, out.revenue_lines = [], []
        out.notes.append(f"Segment / revenue-line extraction failed ({type(e).__name__}); figures omitted.")
    return out


def _extract_into(out: SegmentExtraction, chunks: Sequence[FilingChunk], fiscal_year: int, items: Dict[str, str]) -> None:
    tables = _build_tables(chunks, items)
    if not tables:
        out.notes.append("The filing has no parseable tables; segment and revenue-line figures not extracted.")
        return

    consolidated, income_table = find_consolidated_revenue(tables)
    if consolidated:
        year = max(consolidated)
        out.consolidated_revenue = round(consolidated[year], 2)
        out.consolidated_revenue_year = year
    else:
        out.notes.append("Consolidated revenue (income statement) not found; reconciliation limited to in-table totals.")

    captions = [t for t in ([income_table] if income_table else []) + list(tables) if t.caption_found]
    if not captions:
        out.units = f"{UNITS_MILLIONS} (assumed: no unit caption found in the filing tables)"
        out.notes.append("No unit caption found; table values are assumed to be stated in USD millions.")

    single_segment = any(_SINGLE_SEGMENT.search(c.content or "") for c in chunks if c.chunk_type == "narrative")

    # --- Segment note ---------------------------------------------------------
    stacked: List[_SegmentCandidate] = []
    columns: List[_SegmentCandidate] = []
    for t in tables:
        stacked.extend(_parse_stacked(t))
        col = _parse_columns(t)
        if col:
            columns.append(col)
    candidates = stacked + _merge_column_candidates(columns)
    rejections: List[str] = []
    results = [r for r in (_evaluate_segment_candidate(c, consolidated, rejections) for c in candidates) if r]
    out.segment_tables_found = len(results)
    chosen = max(results, key=lambda r: r.score) if results else None
    if chosen:
        out.segments = _segment_figures(chosen)
        out.segment_layout = chosen.candidate.layout
        out.segment_basis = segment_basis(chosen.reportable)
        out.segment_reconciliation = chosen.reconciliation
        out.segment_source_chunk_ids = list(dict.fromkeys(t.chunk.chunk_id for t in chosen.candidate.tables))
        out.latest_fiscal_year = chosen.latest
        if chosen.reconciliation == "unreconciled":
            out.notes.append("Segment revenue could not be reconciled to a total or to consolidated revenue.")
        if out.segment_basis == "geographic":
            out.notes.append(
                "Reportable segments are geographic regions; product / service mix comes from the revenue disaggregation."
            )
        measures = {f.profit_measure for f in out.segments if f.profit_measure}
        if any(profit_rank(m) == 2 for m in measures):
            out.notes.append("Segment profit is reported as gross profit (operating expenses are not allocated to segments).")
    else:
        out.notes.extend(rejections[:2])
        if single_segment:
            out.segment_basis = "single_segment"
            out.notes.append("The filing describes a single reportable segment; no segment breakdown applies.")
        else:
            out.notes.append(
                "No segment table with revenue and a segment profit measure for two or more segments was found."
            )

    # --- Revenue disaggregation ----------------------------------------------
    segment_keys = {name_key(f.segment_name) for f in out.segments}
    rev_notes: List[str] = []
    rev_results = [
        r for r in (_evaluate_revenue_table(t, consolidated, segment_keys, rev_notes) for t in tables) if r
    ]
    out.revenue_tables_found = len(rev_results)
    best = max(rev_results, key=lambda r: r.score) if rev_results else None
    if best:
        out.revenue_lines = _revenue_figures(best)
        out.revenue_lines_source_chunk_id = best.table.chunk.chunk_id
        out.revenue_lines_reconciliation = best.reconciliation
        out.latest_fiscal_year = out.latest_fiscal_year or best.latest
        if best.reconciliation != "income_statement":
            out.notes.append("Revenue lines reconcile to their table total but could not be checked against consolidated revenue.")
    else:
        out.notes.append(
            "No product / service revenue disaggregation table reconciling to total revenue was found; "
            "revenue lines omitted."
        )

    latest = out.latest_fiscal_year
    if latest is not None and latest != fiscal_year:
        out.notes.append(
            f"Latest period in the extracted tables is {latest} while the filing is labelled FY{fiscal_year}; "
            "years follow the table column headers."
        )
    logger.info(
        f"Segment extraction: {len({f.segment_name for f in out.segments})} segments "
        f"({out.segment_layout}, {out.segment_basis}, {out.segment_reconciliation}), "
        f"{len({f.name for f in out.revenue_lines})} revenue lines ({out.revenue_lines_reconciliation})"
    )


# ==============================================================================
# 10. Sentence screens (customer concentration, pricing)
# ==============================================================================
_CUSTOMER_WORD = re.compile(
    r"\bcustomers?\b|\bclients?\b|\bentit(?:y|ies)\b|\bdistributors?\b|\bpurchasers?\b|\bresellers?\b|\bwholesalers?\b|\bretailers?\b",
    re.IGNORECASE,
)
# A customer's share must be asserted with a verb ("accounted for 12%", "represented 22%",
# "no entity represented 10% or more"), not merely co-occur with a percentage in the sentence.
_SHARE_CLAIM = re.compile(
    r"(?:accounted\s+for|represent(?:ed|s|ing)?|comprised|constituted|made\s+up|amounted\s+to|exceed(?:ed|s)?"
    r"|more\s+than|in\s+excess\s+of)[^.;]{0,40}?(?:\d+(?:\.\d+)?\s?%|ten\s+percent)",
    re.IGNORECASE,
)
_REVENUE_WORD = re.compile(r"revenue|net\s+sales|\bsales\b", re.IGNORECASE)
_RECEIVABLE_WORD = re.compile(r"receivable", re.IGNORECASE)
_PRICING_WORD = re.compile(
    r"\bpric(?:e|es|ing)\b|average\s+selling\s+prices?|\basps?\b|pass(?:ed|ing)?\s+(?:through|on)\b", re.IGNORECASE
)
_PRICING_MOVE = re.compile(
    r"increas|decreas|reduc|higher|lower|premium|pressure|discount|raise|declin|pass|power|competitive|promotion|markdown",
    re.IGNORECASE,
)
_PRICING_EXCLUDE = re.compile(
    r"stock\s+price|share\s+price|price\s+of\s+our\s+(?:common\s+)?stock|exercise\s+price|purchase\s+price|strike\s+price"
    r"|transaction\s+price|selling\s+price\s+(?:basis|method)|standalone\s+selling|stand-alone\s+selling|fair\s+value"
    r"|forward-looking|price\s+risk|interest\s+rate|commodity\s+price|conversion\s+price|repurchase|offering\s+price",
    re.IGNORECASE,
)
# Input-cost sentences ("competitive pricing" from suppliers) are not about the company's own pricing,
# unless they say whether costs can be passed on to customers.
_INPUT_COST = re.compile(r"supplier|vendor|component|raw\s+material|wafer|commodit|procure", re.IGNORECASE)
_PASS_THROUGH = re.compile(r"pass(?:ed|ing)?\s+(?:through|on|along)|to\s+(?:our\s+)?customers", re.IGNORECASE)


def _is_pricing_sentence(sentence: str) -> bool:
    if not (_PRICING_WORD.search(sentence) and _PRICING_MOVE.search(sentence)) or _PRICING_EXCLUDE.search(sentence):
        return False
    return not _INPUT_COST.search(sentence) or bool(_PASS_THROUGH.search(sentence))


def _sentence_key(text: str) -> str:
    return re.sub(r"\s+%", "%", normalize_text(text))


def find_disclosure_sentences(
    chunks: Sequence[FilingChunk],
    items: Dict[str, str],
    exclude_chunk_ids: Optional[set] = None,
) -> List[DisclosureSentence]:
    """Customer-concentration and pricing sentences from narrative chunks outside `exclude_chunk_ids`.

    Concentration sentences are ranked revenue-based first, then receivables; within each kind
    sentences outside Item 1A (MD&A, notes) are preferred over risk-factor language.
    """
    exclude = exclude_chunk_ids or set()
    seen: set = set()
    found: List[Tuple[int, int, int, DisclosureSentence]] = []
    for chunk in chunks:
        if chunk.chunk_type != "narrative" or chunk.chunk_id in exclude:
            continue
        item = items.get(chunk.chunk_id) or chunk.item
        risk_rank = 1 if item == "Item 1A" else 0
        for sentence in split_sentences(chunk.content):
            key = _sentence_key(sentence)
            if key in seen:
                continue
            text = sentence[:MAX_SENTENCE_CHARS]
            if _CUSTOMER_WORD.search(sentence) and _SHARE_CLAIM.search(sentence):
                if _REVENUE_WORD.search(sentence):
                    kind, rank = "revenue_concentration", 0
                elif _RECEIVABLE_WORD.search(sentence):
                    kind, rank = "receivables_concentration", 1
                else:
                    kind = ""
                if kind:
                    seen.add(key)
                    found.append((rank, risk_rank, chunk.chunk_index, DisclosureSentence(chunk_id=chunk.chunk_id, item=item, text=text, kind=kind)))
                    continue
            if _is_pricing_sentence(sentence):
                seen.add(key)
                found.append((2, risk_rank, chunk.chunk_index, DisclosureSentence(chunk_id=chunk.chunk_id, item=item, text=text, kind="pricing")))
    found.sort(key=lambda x: (x[0], x[1], x[2]))
    concentration = [d for r, _, _, d in found if r < 2][:MAX_CONCENTRATION_SENTENCES]
    pricing = [d for r, _, _, d in found if r == 2][:MAX_PRICING_SENTENCES]
    return concentration + pricing


# ==============================================================================
# 11. Markdown renderers
# ==============================================================================
def _money(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"({abs(value):,.0f})" if value < 0 else f"{value:,.0f}"


def _pct_cell(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.1f}%"


def _cell(text: Any) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def render_segment_table_markdown(segments: Sequence[SegmentFigure]) -> str:
    """Latest-year segment table (revenue, share, profit, margin, profit share, YoY) for verbatim embedding."""
    if not segments:
        return "_No reportable-segment financials were extracted from the filing._"
    latest = max(f.fiscal_year for f in segments)
    rows = [f for f in segments if f.fiscal_year == latest]
    measures = {f.profit_measure for f in rows if f.profit_measure}
    measure = measures.pop() if len(measures) == 1 else "Segment profit"
    lines = [
        f"| Segment | FY{latest} revenue ($M) | % of revenue | YoY growth | {_cell(measure)} ($M) | Margin | % of segment profit |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for f in rows:
        name = f.segment_name if f.reportable else f"{f.segment_name} (not a reportable segment)"
        lines.append(
            f"| {_cell(name)} | {_money(f.revenue)} | {_pct_cell(f.revenue_share_pct)} | {_pct_cell(f.revenue_growth_pct)} | "
            f"{_money(f.segment_profit)} | {_pct_cell(f.profit_margin_pct)} | {_pct_cell(f.profit_share_pct)} |"
        )
    lines.append(f"\n_Source: 10-K segment note (chunk `{rows[0].source_chunk_id}`); figures in USD millions as reported._")
    return "\n".join(lines)


def render_revenue_mix_markdown(lines_: Sequence[RevenueLineFigure]) -> str:
    """Latest-year product / service revenue mix with sub-lines indented, for verbatim embedding."""
    if not lines_:
        return "_No product / service revenue disaggregation was extracted from the filing._"
    latest = max(f.fiscal_year for f in lines_)
    rows = [f for f in lines_ if f.fiscal_year == latest]
    out = [
        f"| Revenue line | FY{latest} revenue ($M) | % of revenue | YoY growth |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for f in rows:
        name = f"&nbsp;&nbsp;↳ {f.name}" if f.parent else f.name
        out.append(f"| {_cell(name)} | {_money(f.revenue)} | {_pct_cell(f.revenue_share_pct)} | {_pct_cell(f.revenue_growth_pct)} |")
    out.append(f"\n_Source: 10-K revenue disaggregation (chunk `{rows[0].source_chunk_id}`); figures in USD millions as reported._")
    return "\n".join(out)


__all__ = [
    "FilingChunk",
    "SegmentFigure",
    "RevenueLineFigure",
    "DisclosureSentence",
    "SegmentExtraction",
    "load_filing_chunks",
    "infer_section_items",
    "extract_segment_disclosures",
    "find_consolidated_revenue",
    "find_disclosure_sentences",
    "render_segment_table_markdown",
    "render_revenue_mix_markdown",
    "clean_name",
    "name_key",
    "is_geographic",
    "segment_basis",
    "scale_from_caption",
    "is_revenue_label",
    "profit_rank",
]
