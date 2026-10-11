# Filing Citations & 10-K Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every filing-backed claim or figure in a research report carries a clickable `[n]` marker that opens the full 10-K in a side panel, scrolled to and highlighting the cited chunk.

**Architecture:** A per-message `CitationRegistry` numbers `SourceSpec`s (verbatim quote / derived-from-statements / market input) and emits markdown markers `[[n]](#cite-n)`; `finalize` resolves chunk metadata in one query, drops dead sources and renumbers. A `ReportCiter` maps specialist artifacts to specs for the report templates, and per-chunk figure attribution in `figure_verification.py` adds markers to LLM prose deterministically. The frontend turns `#cite-n` links into buttons that open a `FilingViewerPanel` fed by a new `GET /api/documents/{id}/content` endpoint (plus a `locate` fallback for re-ingested filings).

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy, Jinja2, Pydantic v2, pytest/unittest (backend, in Docker); React 19, TypeScript 6, react-markdown 10, Tailwind v4, lucide-react, Bun (frontend).

**Spec:** `docs/superpowers/specs/2026-10-11-filing-citations-design.md`

## Global Constraints

- Marker format is exactly `[[n]](#cite-n)` with no whitespace before or after; removing a marker must leave the surrounding text byte-identical.
- Source kinds are exactly `verbatim | derived | market`; `market` sources have no chunks and are never navigable.
- No new LLM calls and no prompt changes (deterministic-first; very limited live API budget). Tests are hermetic: no network on ports 80/443; mock OpenAI/yfinance.
- No compatibility shims: `Final10KResearchReport.all_citations` and `_consolidate_citations` are removed and replaced by `sources: List[SourceRef]`.
- At most 3 markers per prose sentence.
- No hardcoded tickers or years in production code; everything must generalize across filings.
- Backend tests run in Docker: `docker exec finance_agent pytest <path> -v` (from any directory). Frontend checks run on the host from `finance_agent_frontend/`: `bun run build` and `bun run lint`.
- Frontend: `verbatimModuleSyntax` is on (use `import type` for types), `noUnusedLocals`/`noUnusedParameters` are on, no new npm dependencies.
- Loggers are namespaced `finance_agent.<area>`. Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Work on branch `feat/filing-citations`.

## Review Focus

1. **A dropped source must not leave a gap or a stray marker.** When a cited chunk was deleted (for example after re-ingestion), the marker must vanish with no stray brackets or spaces, and the surviving markers must be renumbered 1..N. Covered in Task 1: `test_dead_source_marker_removed_exactly` and the renumbering test.
2. **Markers must never reach the LLM.** The audit tables are shared between the report and the synthesizer's prompt context, so the prompt context must render with zero `#cite-` markers. The structured fields (`executive_summary`, `three_pillar_thesis`) must also stay marker-free. Covered in Task 4: `test_structured_fields_and_prompt_context_carry_no_markers`.
3. **A citation bug must not take the report down.** If citation building fails, the report must still publish, with no markers, empty `sources`, and a note in the footer. Covered in Task 4: `test_citation_failure_publishes_report_without_markers`.
4. **Old messages can point at deleted chunks.** After a filing is re-ingested, an old message's `chunk_id`/`document_id` no longer exist. The viewer must still open at the closest location (same breadcrumb, then the same item, then the nearest index) and flag it as approximate. Covered in Task 5 by the locate fallback tests, and in Task 7 by the stale demo source in the manual check.
5. **A sentence whose figure matches many sources must not be flooded with markers.** Each figure gets its single best source, sentence order is kept, and the cap is 3 markers. Covered in Task 2: `test_figures_in_sentence_order_deduped` and `test_limit`.

---

### Task 1: Citation registry

**Files:**
- Create: `finance_agent_backend/app/agents/tools/citation_registry.py`
- Test: `finance_agent_backend/tests/test_tool_citation_registry.py`

**Interfaces:**
- Consumes: `app.database.SessionLocal`, `app.models.Document`, `app.models.DocumentChunk`.
- Produces (used by Tasks 3, 4, 7):
  - `marker(n: int) -> str` returns `"[[n]](#cite-n)"`
  - `strip_markers(text: str) -> str`
  - `@dataclass(frozen=True) SourceSpec(kind, chunk_ids: Tuple[str, ...] = (), quote=None, derived_from=None, provider=None, label=None)` with property `key`
  - `SourceChunk(BaseModel)`: `chunk_id, document_id, chunk_index, ticker, fiscal_year, item, breadcrumb`
  - `SourceRef(BaseModel)`: `n, kind, label, document_id, ticker, fiscal_year, chunks: List[SourceChunk], quote, derived_from, provider`
  - `resolve_chunk_metadata(chunk_ids: List[str]) -> Dict[str, SourceChunk]`
  - `CitationRegistry(resolver: Optional[Callable[[List[str]], Dict[str, SourceChunk]]] = None)`:
    - `.cite(spec: Optional[SourceSpec]) -> str`
    - `.cite_all(specs: Iterable[Optional[SourceSpec]], limit: int = 3) -> str`
    - `.finalize(markdown: str) -> Tuple[str, List[SourceRef]]`

- [ ] **Step 1: Write the failing tests**

Create `finance_agent_backend/tests/test_tool_citation_registry.py`:

```python
"""Numbered filing citations (app/agents/tools/citation_registry.py)."""

import unittest
from unittest.mock import MagicMock

from app.agents.tools.citation_registry import (
    CitationRegistry,
    SourceChunk,
    SourceSpec,
    marker,
    strip_markers,
)


def _chunk(cid: str, idx: int = 0) -> SourceChunk:
    return SourceChunk(
        chunk_id=cid, document_id="doc-1", chunk_index=idx, ticker="ACME", fiscal_year=2025,
        item="Item 7", breadcrumb=f"ACME > Item 7 > {cid}",
    )


def _resolver(known):
    return MagicMock(side_effect=lambda ids: {cid: _chunk(cid, i) for i, cid in enumerate(ids) if cid in known})


QUOTE = SourceSpec(kind="verbatim", chunk_ids=("c1",), quote="Net sales  increased 6%")
DERIVED = SourceSpec(kind="derived", chunk_ids=("c2", "c3"), derived_from="Computed from the audited income statement (FY2025)")
MARKET = SourceSpec(kind="market", label="Share price", provider="yfinance_live")


class TestCitationRegistry(unittest.TestCase):
    def test_marker_format(self):
        self.assertEqual(marker(3), "[[3]](#cite-3)")

    def test_numbering_and_dedupe(self):
        reg = CitationRegistry(resolver=_resolver(set()))
        self.assertEqual(reg.cite(QUOTE), marker(1))
        self.assertEqual(reg.cite(DERIVED), marker(2))
        # Same quote modulo whitespace/case -> same number.
        self.assertEqual(reg.cite(SourceSpec(kind="verbatim", chunk_ids=("c1",), quote="net sales increased 6%")), marker(1))
        # Derived sources dedupe on the chunk set only; the first description wins.
        self.assertEqual(reg.cite(SourceSpec(kind="derived", chunk_ids=("c3", "c2"), derived_from="other")), marker(2))
        self.assertEqual(reg.cite(MARKET), marker(3))
        self.assertEqual(reg.cite(SourceSpec(kind="market", label="Beta", provider="yfinance_live")), marker(4))

    def test_nothing_to_point_at_returns_empty(self):
        reg = CitationRegistry(resolver=_resolver(set()))
        self.assertEqual(reg.cite(None), "")
        self.assertEqual(reg.cite(SourceSpec(kind="derived", chunk_ids=())), "")
        self.assertEqual(reg.cite(SourceSpec(kind="verbatim", chunk_ids=(), quote="x")), "")

    def test_cite_all_distinct_and_limited(self):
        reg = CitationRegistry(resolver=_resolver(set()))
        specs = [QUOTE, None, QUOTE, DERIVED, MARKET, SourceSpec(kind="verbatim", chunk_ids=("c9",), quote="x")]
        self.assertEqual(reg.cite_all(specs), marker(1) + marker(2) + marker(3))

    def test_finalize_renumbers_by_first_appearance_and_resolves_once(self):
        resolver = _resolver({"c1", "c2", "c3"})
        reg = CitationRegistry(resolver=resolver)
        q, d, m = reg.cite(QUOTE), reg.cite(DERIVED), reg.cite(MARKET)
        text, refs = reg.finalize(f"Price{m}. Sales{q}. Table{d}. Again{q}.")
        self.assertEqual(text, f"Price{marker(1)}. Sales{marker(2)}. Table{marker(3)}. Again{marker(2)}.")
        self.assertEqual([r.n for r in refs], [1, 2, 3])
        self.assertEqual([r.kind for r in refs], ["market", "verbatim", "derived"])
        resolver.assert_called_once_with(["c1", "c2", "c3"])
        self.assertEqual(refs[0].provider, "yfinance_live")
        self.assertEqual(refs[0].chunks, [])
        self.assertEqual(refs[1].quote, "Net sales  increased 6%")
        self.assertEqual(refs[1].document_id, "doc-1")
        self.assertEqual(refs[1].ticker, "ACME")
        self.assertEqual([c.chunk_id for c in refs[2].chunks], ["c2", "c3"])
        self.assertEqual(refs[2].derived_from, "Computed from the audited income statement (FY2025)")

    def test_dead_source_marker_removed_exactly(self):
        reg = CitationRegistry(resolver=_resolver({"c2", "c3"}))
        q, d = reg.cite(QUOTE), reg.cite(DERIVED)
        text, refs = reg.finalize(f"Sales grew{q}. Margins held{d}.")
        self.assertEqual(text, f"Sales grew. Margins held{marker(1)}.")
        self.assertEqual([r.n for r in refs], [1])

    def test_partial_chunk_loss_keeps_source(self):
        reg = CitationRegistry(resolver=_resolver({"c3"}))
        d = reg.cite(DERIVED)
        _, refs = reg.finalize(f"x{d}")
        self.assertEqual([c.chunk_id for c in refs[0].chunks], ["c3"])

    def test_unused_specs_are_not_published(self):
        reg = CitationRegistry(resolver=_resolver({"c1", "c2", "c3"}))
        reg.cite(QUOTE)
        d = reg.cite(DERIVED)
        _, refs = reg.finalize(f"only{d}")
        self.assertEqual([r.kind for r in refs], ["derived"])

    def test_labels(self):
        reg = CitationRegistry(resolver=_resolver({"c1"}))
        q = reg.cite(QUOTE)
        m = reg.cite(MARKET)
        labelled = reg.cite(SourceSpec(kind="verbatim", chunk_ids=("c1",), quote="other", label="Custom"))
        _, refs = reg.finalize(q + m + labelled)
        self.assertEqual([r.label for r in refs], ["ACME > Item 7 > c1", "Share price", "Custom"])

    def test_no_chunks_skips_resolver(self):
        resolver = _resolver(set())
        reg = CitationRegistry(resolver=resolver)
        text, refs = reg.finalize(f"p{reg.cite(MARKET)}")
        resolver.assert_not_called()
        self.assertEqual(text, f"p{marker(1)}")

    def test_strip_markers(self):
        self.assertEqual(strip_markers(f"a{marker(1)}{marker(12)}."), "a.")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker exec finance_agent pytest tests/test_tool_citation_registry.py -v`
Expected: collection ERROR `ModuleNotFoundError: No module named 'app.agents.tools.citation_registry'`.

- [ ] **Step 3: Implement the registry**

Create `finance_agent_backend/app/agents/tools/citation_registry.py`:

```python
"""Numbered filing citations for one assistant message.

Producers register a `SourceSpec` and get back an inline markdown marker `[[n]](#cite-n)` (never with
surrounding whitespace). `finalize` resolves chunk metadata in one query, drops sources whose chunks no longer
exist (removing their markers byte-exactly), renumbers the survivors 1..N in order of first appearance and
returns the markdown together with the `SourceRef` list persisted as `ChatMessage.sources`.
"""

import logging
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from app.database import SessionLocal
from app.models import Document, DocumentChunk

logger = logging.getLogger("finance_agent.agents.citations")

SourceKind = Literal["verbatim", "derived", "market"]
MAX_MARKERS = 3
_MARKER = re.compile(r"\[\[(\d+)\]\]\(#cite-\1\)")


def marker(n: int) -> str:
    """Inline markdown marker for source `n`."""
    return f"[[{n}]](#cite-{n})"


def strip_markers(text: str) -> str:
    """Removes every citation marker."""
    return _MARKER.sub("", text)


@dataclass(frozen=True)
class SourceSpec:
    """What a producer knows about a source; chunk metadata is resolved later in one query."""

    kind: SourceKind
    chunk_ids: Tuple[str, ...] = ()
    quote: Optional[str] = None
    derived_from: Optional[str] = None
    provider: Optional[str] = None
    label: Optional[str] = None

    @property
    def key(self) -> Tuple[str, Tuple[str, ...], str]:
        """Dedupe key: quotes by text, derived sources by chunk set, market inputs by label and provider."""
        if self.kind == "verbatim":
            detail = " ".join((self.quote or "").split()).lower()
        elif self.kind == "market":
            detail = f"{self.label}|{self.provider}"
        else:
            detail = ""
        return self.kind, tuple(sorted(set(self.chunk_ids))), detail


class SourceChunk(BaseModel):
    """Location of one cited chunk inside an ingested filing."""

    chunk_id: str
    document_id: str
    chunk_index: int
    ticker: str
    fiscal_year: int
    item: Optional[str] = None
    breadcrumb: Optional[str] = None


class SourceRef(BaseModel):
    """One numbered source as persisted in ChatMessage.sources and rendered by the frontend."""

    n: int
    kind: SourceKind
    label: str
    document_id: Optional[str] = None
    ticker: Optional[str] = None
    fiscal_year: Optional[int] = None
    chunks: List[SourceChunk] = Field(default_factory=list)
    quote: Optional[str] = None
    derived_from: Optional[str] = None
    provider: Optional[str] = None


ChunkResolver = Callable[[List[str]], Dict[str, SourceChunk]]


def resolve_chunk_metadata(chunk_ids: List[str]) -> Dict[str, SourceChunk]:
    """chunk id -> location metadata for the ids that still exist (one query)."""
    if not chunk_ids:
        return {}
    db = SessionLocal()
    try:
        rows = (
            db.query(
                DocumentChunk.id,
                DocumentChunk.document_id,
                DocumentChunk.chunk_index,
                DocumentChunk.item,
                DocumentChunk.breadcrumb,
                Document.ticker,
                Document.fiscal_year,
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .filter(DocumentChunk.id.in_(chunk_ids))
            .all()
        )
    except Exception as e:
        logger.warning(f"Chunk metadata lookup failed ({e!r}); filing sources will be dropped.")
        return {}
    finally:
        db.close()
    return {
        r.id: SourceChunk(
            chunk_id=r.id,
            document_id=r.document_id,
            chunk_index=r.chunk_index,
            ticker=r.ticker,
            fiscal_year=r.fiscal_year,
            item=r.item,
            breadcrumb=r.breadcrumb,
        )
        for r in rows
    }


class CitationRegistry:
    """Numbers the sources cited by one message."""

    def __init__(self, resolver: Optional[ChunkResolver] = None):
        self._resolver = resolver
        self._specs: List[SourceSpec] = []
        self._numbers: Dict[Tuple[str, Tuple[str, ...], str], int] = {}

    def cite(self, spec: Optional[SourceSpec]) -> str:
        """Marker for `spec`; empty when there is nothing to point at."""
        if spec is None or (spec.kind != "market" and not spec.chunk_ids):
            return ""
        n = self._numbers.get(spec.key)
        if n is None:
            self._specs.append(spec)
            n = self._numbers[spec.key] = len(self._specs)
        return marker(n)

    def cite_all(self, specs: Iterable[Optional[SourceSpec]], limit: int = MAX_MARKERS) -> str:
        """Concatenated markers for distinct specs, at most `limit`."""
        out: List[str] = []
        for spec in specs:
            m = self.cite(spec)
            if m and m not in out:
                out.append(m)
            if len(out) >= limit:
                break
        return "".join(out)

    def finalize(self, markdown: str) -> Tuple[str, List[SourceRef]]:
        """Resolves chunks, drops dead sources with their markers and renumbers survivors by first appearance."""
        resolver = self._resolver or resolve_chunk_metadata
        ids = sorted({cid for spec in self._specs for cid in spec.chunk_ids})
        meta = resolver(ids) if ids else {}
        renumbered: Dict[int, int] = {}
        refs: List[SourceRef] = []

        def replace(match: re.Match) -> str:
            old = int(match.group(1))
            if old not in renumbered:
                spec = self._specs[old - 1] if 0 < old <= len(self._specs) else None
                ref = self._build_ref(spec, meta, len(refs) + 1) if spec else None
                renumbered[old] = ref.n if ref else 0
                if ref:
                    refs.append(ref)
            new = renumbered[old]
            return marker(new) if new else ""

        text = _MARKER.sub(replace, markdown)
        dropped = sum(1 for n in renumbered.values() if n == 0)
        if dropped:
            logger.warning(f"Citations: dropped {dropped} source(s) whose filing chunks no longer exist.")
        return text, refs

    @staticmethod
    def _build_ref(spec: SourceSpec, meta: Dict[str, SourceChunk], n: int) -> Optional[SourceRef]:
        chunks = [meta[cid] for cid in dict.fromkeys(spec.chunk_ids) if cid in meta]
        if spec.kind != "market" and not chunks:
            return None
        first = chunks[0] if chunks else None
        if first is not None:
            default_label = first.breadcrumb or first.item or "SEC filing"
        else:
            default_label = spec.label or spec.provider or "Market data"
        return SourceRef(
            n=n,
            kind=spec.kind,
            label=spec.label or default_label,
            document_id=first.document_id if first else None,
            ticker=first.ticker if first else None,
            fiscal_year=first.fiscal_year if first else None,
            chunks=chunks,
            quote=spec.quote if spec.kind == "verbatim" else None,
            derived_from=spec.derived_from if spec.kind == "derived" else None,
            provider=spec.provider if spec.kind == "market" else None,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker exec finance_agent pytest tests/test_tool_citation_registry.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add finance_agent_backend/app/agents/tools/citation_registry.py finance_agent_backend/tests/test_tool_citation_registry.py
git commit -m "feat(citations): numbered citation registry with one-query chunk resolution

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Per-source figure attribution

**Files:**
- Modify: `finance_agent_backend/app/agents/tools/figure_verification.py`
- Test: `finance_agent_backend/tests/test_tool_figure_verification.py` (append a new class)

**Interfaces:**
- Consumes: the existing `_FIGURE`, `_parse`, `_is_figure`, `_SCALE_TO_MILLIONS`, `RELATIVE_TOLERANCE`, `_SENTENCE_BOUNDARY`, `_LINE_PREFIX` in the same module.
- Produces (used by Task 4):
  - `FigureIndex` (dataclass: `values: List[float]`, `entries: List[Tuple[float, int, K]]`)
  - `build_figure_index(entries: Iterable[Tuple[str, K]]) -> FigureIndex[K]`: entry order sets priority; earlier entries win.
  - `attribute_sentence(sentence: str, index: FigureIndex[K], limit: int = 3) -> List[K]`
  - `annotate_sentences(text: str, index: FigureIndex[K], render: Callable[[List[K]], str]) -> str`
  - `collect_known_figures`, `sentence_is_verified` and `strip_unverified_sentences` behave exactly as before.

- [ ] **Step 1: Write the failing tests**

Append to `finance_agent_backend/tests/test_tool_figure_verification.py`. Extend its import block to:

```python
from app.agents.tools.figure_verification import (
    annotate_sentences,
    attribute_sentence,
    build_figure_index,
    collect_known_figures,
    sentence_is_verified,
    strip_unverified_sentences,
)
```

Then add at the end of the file:

```python
class TestFigureAttribution(unittest.TestCase):
    def setUp(self):
        self.index = build_figure_index([
            ("416161.000000", "is-2025"),
            ("31.970000", "is-2025"),
            ("98767.000000", "cf-2025"),
            ("416161.000000", "segment"),
            ("Net sales increased 6% to $416.2 billion", "quote"),
        ])

    def test_best_source_per_figure(self):
        self.assertEqual(attribute_sentence("Revenue was $416.2 billion.", self.index), ["is-2025"])

    def test_figures_in_sentence_order_deduped(self):
        s = "Margin was 31.97% on revenue of $416,161M while FCF was $98.8 billion."
        self.assertEqual(attribute_sentence(s, self.index), ["is-2025", "cf-2025"])

    def test_years_small_integers_and_unknown_figures_are_ignored(self):
        self.assertEqual(attribute_sentence("In FY2025 the 3 segments earned $123,456M.", self.index), [])

    def test_limit(self):
        index = build_figure_index([("1000", "a"), ("2000", "b"), ("3000", "c"), ("4000", "d")])
        self.assertEqual(attribute_sentence("$1,000M, $2,000M, $3,000M and $4,000M.", index), ["a", "b", "c"])

    def test_annotate_inserts_before_final_punctuation_and_keeps_bullets(self):
        text = "- Revenue was $416.2 billion. No figures here.\n- FCF was **$98.8 billion.**"
        out = annotate_sentences(text, self.index, lambda keys: "".join(f"[{k}]" for k in keys))
        self.assertEqual(
            out, "- Revenue was $416.2 billion[is-2025]. No figures here.\n- FCF was **$98.8 billion[cf-2025].**"
        )

    def test_annotate_without_terminal_punctuation_appends(self):
        self.assertEqual(
            annotate_sentences("Revenue was $416.2 billion", self.index, lambda keys: "[x]"),
            "Revenue was $416.2 billion[x]",
        )

    def test_annotate_leaves_unattributed_text_unchanged(self):
        self.assertEqual(annotate_sentences("", self.index, lambda keys: "[x]"), "")
        self.assertEqual(annotate_sentences("No numbers. None at all.", self.index, lambda keys: "[x]"), "No numbers. None at all.")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker exec finance_agent pytest tests/test_tool_figure_verification.py -v`
Expected: ImportError `cannot import name 'annotate_sentences'`.

- [ ] **Step 3: Implement attribution**

In `finance_agent_backend/app/agents/tools/figure_verification.py`:

1. Replace the module docstring's last line block and imports so the header reads:

```python
"""Deterministic guard that keeps model-written prose tied to verified figures.

Every number in a sentence must match (within its own rounding precision) a number present in the
verified context the model was given; sentences citing anything else are removed. Scale words are
reconciled with the $-millions convention used across the pipeline ("$716.9 billion" == 716,924).
Small integers (counts, "5-year", item numbers) and calendar years are not treated as figures.

The same matcher attributes verified figures to their sources: `build_figure_index` records which source
each known figure came from, and `annotate_sentences` appends a citation marker to each sentence.
"""

import bisect
import re
from dataclasses import dataclass
from typing import Callable, Generic, Hashable, Iterable, List, Tuple, TypeVar
```

2. Replace the existing `_matches` function with:

```python
def _window(value: float, decimals: int, unit: str) -> Tuple[float, float]:
    """Inclusive $-millions interval a figure written with `decimals` precision can stand for."""
    scale = _SCALE_TO_MILLIONS.get(unit, 1.0)
    candidate = value * scale
    tolerance = 0.5 * 10 ** (-decimals) * scale + RELATIVE_TOLERANCE * candidate
    return candidate - tolerance, candidate + tolerance


def _matches(known: List[float], value: float, decimals: int, unit: str) -> bool:
    lo, hi = _window(value, decimals, unit)
    i = bisect.bisect_left(known, lo)
    return i < len(known) and known[i] <= hi
```

3. Append at the end of the module:

```python
K = TypeVar("K", bound=Hashable)
_TRAILING_PUNCT = re.compile(r"[.!?]+[\"')\]*_]*\s*$")


@dataclass
class FigureIndex(Generic[K]):
    """Known figures (in $ millions when scaled) with the source each came from; lower priority wins."""

    values: List[float]
    entries: List[Tuple[float, int, K]]


def build_figure_index(entries: Iterable[Tuple[str, K]]) -> FigureIndex[K]:
    """Indexes every figure in each (text, source) entry; the entry's position is its priority."""
    rows: List[Tuple[float, int, K]] = []
    for priority, (text, key) in enumerate(entries):
        for match in _FIGURE.finditer(text or ""):
            value, _, unit, _ = _parse(match)
            rows.append((value * _SCALE_TO_MILLIONS.get(unit, 1.0), priority, key))
    rows.sort(key=lambda row: (row[0], row[1]))
    return FigureIndex(values=[row[0] for row in rows], entries=rows)


def attribute_sentence(sentence: str, index: FigureIndex[K], limit: int = 3) -> List[K]:
    """Best (highest-priority) source for each figure in `sentence`, in sentence order, at most `limit`."""
    found: List[K] = []
    for match in _FIGURE.finditer(sentence):
        value, decimals, unit, dollar = _parse(match)
        if not _is_figure(value, decimals, unit, dollar):
            continue
        lo, hi = _window(value, decimals, unit)
        window = index.entries[bisect.bisect_left(index.values, lo):bisect.bisect_right(index.values, hi)]
        if not window:
            continue
        best = min(window, key=lambda row: row[1])[2]
        if best not in found:
            found.append(best)
        if len(found) >= limit:
            break
    return found


def _attach(sentence: str, mark: str) -> str:
    """Places `mark` before the sentence's closing punctuation (and any closing emphasis/quotes)."""
    m = _TRAILING_PUNCT.search(sentence)
    if m:
        return sentence[:m.start()] + mark + sentence[m.start():]
    return sentence.rstrip() + mark


def annotate_sentences(text: str, index: FigureIndex[K], render: Callable[[List[K]], str]) -> str:
    """Appends `render(sources)` to every sentence whose figures are attributable; other text is unchanged."""
    if not text:
        return text
    out_lines: List[str] = []
    for line in text.split("\n"):
        prefix = _LINE_PREFIX.match(line).group(1)
        body = line[len(prefix):]
        sentences = []
        for sentence in _SENTENCE_BOUNDARY.split(body):
            keys = attribute_sentence(sentence, index)
            mark = render(keys) if keys else ""
            sentences.append(_attach(sentence, mark) if mark else sentence)
        out_lines.append(prefix + " ".join(sentences))
    return "\n".join(out_lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker exec finance_agent pytest tests/test_tool_figure_verification.py tests/test_agent_lead_synthesizer.py -v`
Expected: all pass. The existing verification tests are unchanged, which proves the `_window` refactor kept behaviour.

- [ ] **Step 5: Commit**

```bash
git add finance_agent_backend/app/agents/tools/figure_verification.py finance_agent_backend/tests/test_tool_figure_verification.py
git commit -m "feat(citations): attribute verified prose figures to their sources

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Report citer (artifacts → sources)

**Files:**
- Create: `finance_agent_backend/app/agents/specialized/lead_synthesizer/report_citations.py`
- Test: `finance_agent_backend/tests/test_report_citations.py`

**Interfaces:**
- Consumes: `CitationRegistry`, `SourceSpec` (Task 1); `FinancialAuditOutput.data_quality.field_provenance` (`"FY<year>.<field>" -> {"statement", "chunk_id", "filing_year", ...}`); `BusinessMoatOutput.evidence` (`StrategyEvidence.claim/subject/chunk_id/quote`), `.segment_financials[].source_chunk_id`, `.revenue_lines[].source_chunk_id`; `RiskItem.evidence[]` / `.mitigation_evidence` (`RiskEvidence.chunk_id/quote/verified`); `ForecastOutput.citations` (dicts with `chunk_id`); `DCFValuationOutput.data_quality.share_price_source`, `.wacc_audit.{risk_free_rate_source, equity_risk_premium_source, beta_source, risk_free_rate_pct, equity_risk_premium_pct, beta}`, `.current_share_price`, `.net_debt`, `.diluted_shares`.
- Produces (used by Task 4 templates and synthesizer):
  - `ROW_STATEMENTS: Dict[str, Tuple[str, ...]]`
  - `NullCiter` (every attribute is a callable returning `""`)
  - `ReportCiter(registry, fiscal_year, financial_audit=None, business_moat=None, risk_audit=None, forecast=None, dcf_valuation=None)`. Its marker methods all return `str` and never raise:
    - `row(attr, items)`, `field(attr, year=None)`, `statements(year=None)`, `statement(name, year=None)`
    - `evidence(claim, subject=None)`, `risk(risk_item)`, `mitigation(risk_item)`
    - `segments()`, `revenue_mix()`, `forecast_drivers()`, `market(name)` where name is `"price" | "rf" | "erp" | "beta"`
  - `ReportCiter.figure_entries() -> List[Tuple[str, SourceSpec]]`
  - `ReportCiter.registry` (the `CitationRegistry`)

- [ ] **Step 1: Write the failing tests**

Create `finance_agent_backend/tests/test_report_citations.py`:

```python
"""Artifact -> filing source mapping for the code-assembled report (lead_synthesizer/report_citations.py)."""

import re
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from app.agents.specialized.business_strategist.state_business_strategist import StrategyEvidence
from app.agents.specialized.lead_synthesizer.report_citations import (
    ROW_STATEMENTS,
    NullCiter,
    ReportCiter,
)
from app.agents.state import RiskEvidence
from app.agents.tools.business_disclosure_tools import SegmentFigure
from app.agents.tools.citation_registry import CitationRegistry, marker
from tests.test_agent_lead_synthesizer import _create_mock_audit, _create_mock_moat, _create_mock_risk
from tests.valuation_fixtures import make_valuation_output

IS, BS, CF = "income_statement", "balance_sheet", "cash_flow"
TEMPLATES = Path(__file__).resolve().parents[1] / "app" / "agents" / "specialized" / "lead_synthesizer"


def provenance():
    out = {}
    for year in (2023, 2024, 2025):
        for field, statement in (("revenue", IS), ("operating_income", IS), ("operating_cash_flow", CF)):
            out[f"FY{year}.{field}"] = {"statement": statement, "chunk_id": f"{statement}-fy2025", "filing_year": 2025}
    out["FY2025.cash_and_equivalents"] = {"statement": BS, "chunk_id": "balance_sheet-fy2025", "filing_year": 2025}
    out["FY2024.cash_and_equivalents"] = {"statement": BS, "chunk_id": "balance_sheet-fy2024", "filing_year": 2024}
    return out


def audit_with_provenance():
    audit = _create_mock_audit()
    audit.data_quality.field_provenance = provenance()
    return audit


def moat_with_evidence():
    return _create_mock_moat().model_copy(update={
        "competitive_threats": ["Android OEMs"],
        "evidence": [
            StrategyEvidence(claim="primary_moat", subject="High Switching Costs", chunk_id="item1-a",
                             quote="Customers who own multiple Apple devices rely on iCloud continuity"),
            StrategyEvidence(claim="competitive_threat", subject="Android OEMs", chunk_id="item1-b",
                             quote="The Company faces substantial competition from Android OEMs"),
            StrategyEvidence(claim="competitive_threat", subject="Other", chunk_id="item1-c", quote="Other rivals"),
        ],
        "segment_financials": [
            SegmentFigure(segment_name="Americas", fiscal_year=2025, revenue=178353.0, revenue_share_pct=42.86,
                          source_chunk_id="seg-note"),
        ],
    })


def citer(**kwargs):
    defaults = dict(financial_audit=audit_with_provenance(), business_moat=moat_with_evidence(),
                    risk_audit=_create_mock_risk(), dcf_valuation=make_valuation_output())
    return ReportCiter(CitationRegistry(resolver=MagicMock(return_value={})), fiscal_year=2025, **{**defaults, **kwargs})


class TestReportCiter(unittest.TestCase):
    def test_every_audit_table_attribute_is_mapped(self):
        text = (TEMPLATES / "audit_tables.j2").read_text(encoding="utf-8")
        attrs = set(re.findall(r'row\("[^"]+", fa\.\w+, "(\w+)"', text)) | set(re.findall(r'fe\("(\w+)"', text))
        self.assertGreater(len(attrs), 40)
        self.assertEqual(attrs - set(ROW_STATEMENTS), set())

    def test_rows_on_the_same_statements_share_one_source(self):
        c = citer()
        items = audit_with_provenance().multi_year_history
        revenue = c.row("revenue", items)
        self.assertEqual(revenue, marker(1))
        self.assertEqual(c.row("operating_income", items), revenue)
        self.assertEqual(c.row("operating_cash_flow", items), marker(2))
        spec = c.registry._specs[0]
        self.assertEqual(spec.kind, "derived")
        self.assertEqual(spec.chunk_ids, ("income_statement-fy2025",))
        self.assertEqual(spec.derived_from, "Computed from the audited income statement (FY2023–FY2025)")

    def test_unmapped_or_unprovenanced_rows_get_no_marker(self):
        c = citer()
        self.assertEqual(c.row("not_a_metric", audit_with_provenance().multi_year_history), "")
        self.assertEqual(citer(financial_audit=_create_mock_audit()).row("revenue", _create_mock_audit().multi_year_history), "")

    def test_field_uses_anchor_year_by_default(self):
        c = citer()
        c.field("net_debt")
        self.assertEqual(c.registry._specs[0].chunk_ids, ("balance_sheet-fy2025",))
        c.field("net_debt", 2024)
        self.assertEqual(c.registry._specs[1].chunk_ids, ("balance_sheet-fy2024",))

    def test_statements_spans_all_three_statements(self):
        c = citer()
        c.statements()
        self.assertEqual(
            c.registry._specs[0].chunk_ids,
            ("income_statement-fy2025", "balance_sheet-fy2025", "cash_flow-fy2025"),
        )

    def test_evidence_filters_by_claim_and_subject(self):
        c = citer()
        self.assertEqual(c.evidence("competitive_threat", "Android OEMs"), marker(1))
        self.assertEqual(c.registry._specs[0].quote, "The Company faces substantial competition from Android OEMs")
        self.assertEqual(c.evidence("pricing_power"), "")
        self.assertEqual(c.evidence("competitive_threat"), marker(1) + marker(2))

    def test_risk_cites_only_verified_quotes(self):
        c = citer()
        risk = _create_mock_risk().identified_risks[0].model_copy(update={"evidence": [
            RiskEvidence(chunk_id="r1", quote="verified quote", verified=True),
            RiskEvidence(chunk_id="r2", quote="unverified quote", verified=False),
        ]})
        self.assertEqual(c.risk(risk), marker(1))
        self.assertEqual(c.mitigation(risk), "")

    def test_market_inputs_carry_provider(self):
        c = citer()
        self.assertEqual(c.market("rf"), marker(1))
        spec = c.registry._specs[0]
        self.assertEqual((spec.kind, spec.label), ("market", "Risk-free rate"))
        self.assertTrue(spec.provider)
        self.assertEqual(c.market("unknown"), "")
        self.assertEqual(citer(dcf_valuation=None).market("price"), "")

    def test_segments_and_failures_never_raise(self):
        c = citer()
        self.assertEqual(c.segments(), marker(1))
        self.assertEqual(c.revenue_mix(), "")
        self.assertEqual(c.forecast_drivers(), "")
        self.assertEqual(c.risk(None), "")

    def test_figure_entries_priority(self):
        entries = citer().figure_entries()
        kinds = [spec.kind for _, spec in entries]
        self.assertEqual(kinds[0], "derived")
        self.assertIn("market", kinds)
        self.assertEqual(kinds[-1], "verbatim")
        first_year_texts = [text for text, spec in entries[:20]]
        self.assertIn(f"{416161.0:.6f}", first_year_texts)

    def test_null_citer_is_silent(self):
        self.assertEqual(NullCiter().row("revenue", []), "")
        self.assertEqual(NullCiter().market("price"), "")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker exec finance_agent pytest tests/test_report_citations.py -v`
Expected: collection ERROR `ModuleNotFoundError: ...report_citations`.

- [ ] **Step 3: Implement the citer**

Create `finance_agent_backend/app/agents/specialized/lead_synthesizer/report_citations.py`:

```python
"""Filing sources for the code-assembled research report.

Maps specialist artifacts to `SourceSpec`s. Verified strategist and risk quotes are verbatim sources. Audit,
forecast and valuation figures are derived from the statement-table chunks the auditor recorded in
`field_provenance`. Market inputs carry their provider tag. The report templates call the marker methods
(each returns an inline marker or "", and never raises), and `figure_entries` seeds the index that
attributes LLM prose figures.
"""

import logging
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from app.agents.tools.citation_registry import CitationRegistry, SourceSpec

logger = logging.getLogger("finance_agent.agents.lead_synthesizer.citations")

IS, BS, CF = "income_statement", "balance_sheet", "cash_flow"
STATEMENT_NAMES = {IS: "income statement", BS: "balance sheet", CF: "cash flow statement"}

# Audit metric -> statements whose rows it is computed from (every attribute rendered by audit_tables.j2).
ROW_STATEMENTS: Dict[str, Tuple[str, ...]] = {
    # Table A
    "revenue": (IS,), "revenue_growth_pct": (IS,), "gross_margin_pct": (IS,), "operating_income": (IS,),
    "operating_margin_pct": (IS,), "net_income": (IS,), "net_margin_pct": (IS,),
    "operating_cash_flow": (CF,), "capital_expenditures": (CF,), "free_cash_flow": (CF,),
    "fcf_conversion_pct": (CF, IS), "stock_based_compensation": (CF,), "sbc_adjusted_fcf": (CF,),
    # Table B
    "normalized_operating_income": (IS,), "normalized_operating_margin_pct": (IS,), "normalized_net_income": (IS,),
    "adjustments_pct_of_operating_income": (IS,), "non_operating_income": (IS,),
    "non_operating_to_operating_income_pct": (IS,), "effective_tax_rate_pct": (IS,),
    "deferred_tax_to_net_income_pct": (CF, IS), "accruals": (IS, CF), "sloan_accruals_ratio_pct": (IS, CF, BS),
    # Table C
    "gross_margin_change_bps": (IS,), "operating_margin_change_bps": (IS,), "operating_expenses": (IS,),
    "operating_expense_growth_pct": (IS,), "opex_growth_minus_revenue_growth_pp": (IS,),
    "dso_days": (BS, IS), "dio_days": (BS, IS), "dpo_days": (BS, IS), "cash_conversion_cycle_days": (BS, IS),
    # Table D
    "capex_to_revenue_pct": (CF, IS), "capex_to_depreciation": (CF,), "ocf_change": (CF,), "capex_change": (CF,),
    "fcf_change": (CF,), "fcf_change_from_capex_pct": (CF,), "diluted_weighted_shares": (IS,),
    "period_end_shares_outstanding": (BS,), "period_end_share_growth_pct": (BS,),
    # Table E (balance-sheet snapshot) and valuation bridge inputs
    "cash_and_equivalents": (BS,), "marketable_securities": (BS,), "total_liquid_cash": (BS,), "total_debt": (BS,),
    "net_debt": (BS,), "stockholders_equity": (BS,), "valuation_shares_outstanding": (BS,), "current_ratio": (BS,),
    "ebitda": (IS, CF), "net_debt_to_ebitda": (BS, IS, CF), "total_debt_to_equity": (BS,),
    "roic_pct": (IS, BS), "roic_on_gross_capital_pct": (IS, BS), "roe_pct": (IS, BS),
}

VALUATION_FIGURES = (
    "implied_fair_value_per_share", "bear_fair_value", "bull_fair_value", "probability_weighted_fair_value",
    "enterprise_value", "equity_value", "terminal_value_pct_of_ev",
)
SEGMENT_FIGURES = ("revenue", "segment_profit", "revenue_share_pct", "profit_margin_pct", "profit_share_pct", "revenue_growth_pct")
REVENUE_LINE_FIGURES = ("revenue", "revenue_share_pct", "revenue_growth_pct")
MARKET_INPUTS: Dict[str, Tuple[str, Callable[[Any], Optional[str]]]] = {
    "price": ("Share price", lambda v: v.data_quality.share_price_source),
    "rf": ("Risk-free rate", lambda v: v.wacc_audit.risk_free_rate_source),
    "erp": ("Equity risk premium", lambda v: v.wacc_audit.equity_risk_premium_source),
    "beta": ("Beta", lambda v: v.wacc_audit.beta_source),
}
MARKET_FIGURES: Tuple[Tuple[str, Callable[[Any], Any]], ...] = (
    ("price", lambda v: v.current_share_price),
    ("rf", lambda v: v.wacc_audit.risk_free_rate_pct),
    ("erp", lambda v: v.wacc_audit.equity_risk_premium_pct),
    ("beta", lambda v: v.wacc_audit.beta),
)
_AUDIT_SERIES = ("multi_year_history", "earnings_quality", "cost_structure", "working_capital", "capital_intensity", "share_count_history")
_AUDIT_SNAPSHOTS = ("balance_sheet", "profitability_and_return_ratios", "solvency_and_liquidity_ratios")


def _safe(method: Callable[..., str]) -> Callable[..., str]:
    """A citation failure must never break the report: log it and emit no marker."""

    @wraps(method)
    def wrapper(self: "ReportCiter", *args: Any, **kwargs: Any) -> str:
        try:
            return method(self, *args, **kwargs)
        except Exception as e:
            logger.warning(f"Citation marker {method.__name__} failed ({e!r}); emitting none.")
            return ""

    return wrapper


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _statement_chunk_map(financial_audit: Any) -> Dict[Tuple[str, int], List[str]]:
    """(statement, fiscal year) -> statement-table chunk ids, from the auditor's field provenance."""
    out: Dict[Tuple[str, int], List[str]] = {}
    provenance = getattr(getattr(financial_audit, "data_quality", None), "field_provenance", None) or {}
    for key, entry in provenance.items():
        if not isinstance(key, str) or not key.startswith("FY") or "." not in key or not isinstance(entry, dict):
            continue
        try:
            year = int(key[2:key.index(".")])
        except ValueError:
            continue
        chunk_id, statement = entry.get("chunk_id"), entry.get("statement")
        if chunk_id and statement:
            ids = out.setdefault((statement, year), [])
            if chunk_id not in ids:
                ids.append(chunk_id)
    return out


class NullCiter:
    """Stand-in when citations cannot be built: every marker is empty."""

    def __getattr__(self, name: str) -> Callable[..., str]:
        return lambda *args, **kwargs: ""


class ReportCiter:
    """Jinja-facing helper returning inline markers for report elements."""

    def __init__(
        self,
        registry: CitationRegistry,
        fiscal_year: int,
        financial_audit: Any = None,
        business_moat: Any = None,
        risk_audit: Any = None,
        forecast: Any = None,
        dcf_valuation: Any = None,
    ):
        self.registry = registry
        self._anchor = fiscal_year
        self._fa = financial_audit
        self._moat = business_moat
        self._risk = risk_audit
        self._forecast = forecast
        self._val = dcf_valuation
        self._statement_chunks = _statement_chunk_map(financial_audit)

    # --- specs ------------------------------------------------------------
    def _statement_spec(self, statements: Sequence[str], years: Iterable[Optional[int]]) -> Optional[SourceSpec]:
        span_years = sorted({y for y in years if y is not None})
        ids: List[str] = []
        for statement in statements:
            for year in span_years:
                ids += [c for c in self._statement_chunks.get((statement, year), []) if c not in ids]
        if not ids:
            return None
        names = [STATEMENT_NAMES[s] for s in statements if s in STATEMENT_NAMES]
        listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        span = f"FY{span_years[0]}" if len(span_years) == 1 else f"FY{span_years[0]}–FY{span_years[-1]}"
        return SourceSpec(kind="derived", chunk_ids=tuple(ids), derived_from=f"Computed from the audited {listed} ({span})")

    def _field_spec(self, attr: str, year: Optional[int] = None) -> Optional[SourceSpec]:
        return self._statement_spec(ROW_STATEMENTS.get(attr, ()), [year or self._anchor])

    def _all_statements_spec(self, year: Optional[int] = None) -> Optional[SourceSpec]:
        return self._statement_spec((IS, BS, CF), [year or self._anchor])

    @staticmethod
    def _quote_spec(chunk_id: Optional[str], quote: Optional[str]) -> Optional[SourceSpec]:
        if not chunk_id or not quote:
            return None
        return SourceSpec(kind="verbatim", chunk_ids=(chunk_id,), quote=quote)

    def _evidence_specs(self, claim: str, subject: Optional[str] = None) -> List[Optional[SourceSpec]]:
        evidence = getattr(self._moat, "evidence", None) or []
        return [
            self._quote_spec(e.chunk_id, e.quote)
            for e in evidence
            if e.claim == claim and (subject is None or (e.subject or "") == subject)
        ]

    @staticmethod
    def _chunks_spec(chunk_ids: Iterable[Optional[str]], derived_from: str) -> Optional[SourceSpec]:
        ids = tuple(dict.fromkeys(str(c) for c in chunk_ids if c))
        return SourceSpec(kind="derived", chunk_ids=ids, derived_from=derived_from) if ids else None

    def _segments_spec(self) -> Optional[SourceSpec]:
        figures = getattr(self._moat, "segment_financials", None) or []
        return self._chunks_spec((f.source_chunk_id for f in figures), "Reported in the segment note (ASC 280)")

    def _revenue_mix_spec(self) -> Optional[SourceSpec]:
        figures = getattr(self._moat, "revenue_lines", None) or []
        return self._chunks_spec((f.source_chunk_id for f in figures), "Reported in the revenue disaggregation table (ASC 606)")

    def _forecast_spec(self) -> Optional[SourceSpec]:
        citations = getattr(self._forecast, "citations", None) or []
        ids = (c.get("chunk_id") for c in citations if isinstance(c, dict))
        return self._chunks_spec(ids, "MD&A passages behind the forecast drivers")

    def _market_spec(self, name: str) -> Optional[SourceSpec]:
        if self._val is None or name not in MARKET_INPUTS:
            return None
        label, provider_of = MARKET_INPUTS[name]
        provider = provider_of(self._val)
        if not provider or provider == "unavailable":
            return None
        return SourceSpec(kind="market", label=label, provider=provider)

    # --- template markers -------------------------------------------------
    @_safe
    def row(self, attr: str, items: Any) -> str:
        years = [getattr(item, "fiscal_year", None) for item in (items or [])]
        return self.registry.cite(self._statement_spec(ROW_STATEMENTS.get(attr, ()), years))

    @_safe
    def field(self, attr: str, year: Optional[int] = None) -> str:
        return self.registry.cite(self._field_spec(attr, year))

    @_safe
    def statements(self, year: Optional[int] = None) -> str:
        return self.registry.cite(self._all_statements_spec(year))

    @_safe
    def statement(self, name: str, year: Optional[int] = None) -> str:
        return self.registry.cite(self._statement_spec((name,), [year or self._anchor]))

    @_safe
    def evidence(self, claim: str, subject: Optional[str] = None) -> str:
        return self.registry.cite_all(self._evidence_specs(claim, subject))

    @_safe
    def risk(self, risk_item: Any) -> str:
        return self.registry.cite_all(self._quote_spec(e.chunk_id, e.quote) for e in risk_item.evidence if e.verified)

    @_safe
    def mitigation(self, risk_item: Any) -> str:
        ev = risk_item.mitigation_evidence
        return self.registry.cite(self._quote_spec(ev.chunk_id, ev.quote) if ev is not None and ev.verified else None)

    @_safe
    def segments(self) -> str:
        return self.registry.cite(self._segments_spec())

    @_safe
    def revenue_mix(self) -> str:
        return self.registry.cite(self._revenue_mix_spec())

    @_safe
    def forecast_drivers(self) -> str:
        return self.registry.cite(self._forecast_spec())

    @_safe
    def market(self, name: str) -> str:
        return self.registry.cite(self._market_spec(name))

    # --- prose attribution ----------------------------------------------
    def _audit_items(self) -> List[Tuple[Optional[int], Any]]:
        fa = self._fa
        if fa is None:
            return []
        items: List[Tuple[Optional[int], Any]] = []
        for name in _AUDIT_SERIES:
            for item in getattr(fa, name, None) or []:
                items.append((getattr(item, "fiscal_year", None), item))
        snapshot_year = getattr(getattr(fa, "balance_sheet", None), "fiscal_year", None) or self._anchor
        for name in _AUDIT_SNAPSHOTS:
            item = getattr(fa, name, None)
            if item is not None:
                items.append((snapshot_year, item))
        return sorted(items, key=lambda pair: -(pair[0] or 0))

    def figure_entries(self) -> List[Tuple[str, SourceSpec]]:
        """(text, source) pairs in attribution priority: audit data newest year first, valuation, segments, quotes."""
        entries: List[Tuple[str, SourceSpec]] = []

        def add(value: Any, spec: Optional[SourceSpec]) -> None:
            if spec is not None and _is_number(value):
                entries.append((f"{float(value):.6f}", spec))

        for year, item in self._audit_items():
            for attr, value in item.model_dump().items():
                if attr in ROW_STATEMENTS:
                    add(value, self._field_spec(attr, year))
        val = self._val
        if val is not None:
            for attr in VALUATION_FIGURES:
                add(getattr(val, attr, None), self._all_statements_spec())
            add(getattr(val, "net_debt", None), self._field_spec("net_debt"))
            add(getattr(val, "diluted_shares", None), self._field_spec("valuation_shares_outstanding"))
            for name, value_of in MARKET_FIGURES:
                add(value_of(val), self._market_spec(name))
        moat = self._moat
        if moat is not None:
            for figure in moat.segment_financials:
                for attr in SEGMENT_FIGURES:
                    add(getattr(figure, attr, None), self._segments_spec())
            for figure in moat.revenue_lines:
                for attr in REVENUE_LINE_FIGURES:
                    add(getattr(figure, attr, None), self._revenue_mix_spec())
            for e in moat.evidence:
                spec = self._quote_spec(e.chunk_id, e.quote)
                if spec is not None:
                    entries.append((e.quote, spec))
        if self._risk is not None:
            for risk_item in self._risk.identified_risks:
                for e in risk_item.evidence:
                    spec = self._quote_spec(e.chunk_id, e.quote) if e.verified else None
                    if spec is not None:
                        entries.append((e.quote, spec))
        return entries
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker exec finance_agent pytest tests/test_report_citations.py -v`
Expected: 11 passed. `test_every_audit_table_attribute_is_mapped` can only pass after Task 4 adds `fe(...)` calls to Table E. Until then it passes on the Table A–D rows alone (more than 40 attributes, all mapped). Re-run it after Task 4.

- [ ] **Step 5: Commit**

```bash
git add finance_agent_backend/app/agents/specialized/lead_synthesizer/report_citations.py finance_agent_backend/tests/test_report_citations.py
git commit -m "feat(citations): map specialist artifacts to filing sources for the report

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Synthesizer, templates and orchestrator emit cited reports

**Files:**
- Modify: `finance_agent_backend/app/agents/specialized/lead_synthesizer/state_lead_synthesizer.py`
- Modify: `finance_agent_backend/app/agents/specialized/lead_synthesizer/agent_lead_synthesizer.py`
- Modify: `finance_agent_backend/app/agents/specialized/lead_synthesizer/report_layout.j2`
- Modify: `finance_agent_backend/app/agents/specialized/lead_synthesizer/report_blocks.j2`
- Modify: `finance_agent_backend/app/agents/specialized/lead_synthesizer/audit_tables.j2`
- Modify: `finance_agent_backend/app/agents/orchestrator.py` (lines 307, 574, 659)
- Modify: `finance_agent_backend/tests/test_agent_lead_synthesizer.py` (setUp + line 283)
- Modify: `finance_agent_backend/tests/test_full_pipeline.py` (lines 252–273 and 484–485)
- Modify: `finance_agent_backend/tests/test_financial_audit_rendering.py` (the `report_layout.j2` render in `test_02`)
- Test: `finance_agent_backend/tests/test_agent_lead_synthesizer_citations.py`

**Interfaces:**
- Consumes: `CitationRegistry`, `SourceRef`, `strip_markers` (Task 1); `build_figure_index`, `annotate_sentences` (Task 2); `ReportCiter`, `NullCiter` (Task 3).
- Produces: `Final10KResearchReport.sources: List[SourceRef]`, replacing `all_citations`. The orchestrator streams/persists `[s.model_dump() for s in report.sources]` as `sources`, so the frontend (Tasks 6–7) receives `SourceRef` JSON.

- [ ] **Step 1: Write the failing tests**

Create `finance_agent_backend/tests/test_agent_lead_synthesizer_citations.py`:

```python
"""Cited research reports: inline markers <-> sources (lead synthesizer + report_citations)."""

import re
import unittest
from unittest.mock import patch

from app.agents.specialized.lead_synthesizer.agent_lead_synthesizer import LeadSynthesizerAgent
from app.agents.specialized.prompts import render_prompt
from app.agents.tools.citation_registry import SourceChunk
from tests.test_agent_lead_synthesizer import (
    FULL_PROSE,
    _create_mock_dcf,
    _create_mock_forecast,
    _create_mock_risk,
    _mock_llm,
)
from tests.test_report_citations import audit_with_provenance, moat_with_evidence

MARKER = re.compile(r"\[\[(\d+)\]\]\(#cite-\1\)")


def _fake_resolver(ids):
    return {
        cid: SourceChunk(chunk_id=cid, document_id="doc-aapl-2025", chunk_index=i, ticker="AAPL", fiscal_year=2025,
                         item="Item 8", breadcrumb=f"Apple Inc. (AAPL) > Item 8 > {cid}")
        for i, cid in enumerate(ids)
    }


class TestSynthesizerCitations(unittest.TestCase):
    def setUp(self):
        self.synth = LeadSynthesizerAgent()
        patcher = patch("app.agents.tools.citation_registry.resolve_chunk_metadata", side_effect=_fake_resolver)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _report(self, **overrides):
        kwargs = dict(
            business_moat=moat_with_evidence(), financial_audit=audit_with_provenance(),
            forecast=_create_mock_forecast(), dcf_valuation=_create_mock_dcf(), risk_audit=_create_mock_risk(),
        )
        kwargs.update(overrides)
        with patch.object(self.synth, "_get_llm", return_value=_mock_llm(FULL_PROSE)):
            return self.synth.synthesize(ticker="AAPL", company_name="Apple Inc.", fiscal_year=2025, **kwargs)

    def test_markers_and_sources_are_a_bijection(self):
        report = self._report()
        numbers = {int(n) for n in MARKER.findall(report.full_markdown_report)}
        self.assertTrue(numbers)
        self.assertEqual(numbers, set(range(1, len(report.sources) + 1)))
        self.assertEqual([s.n for s in report.sources], list(range(1, len(report.sources) + 1)))
        self.assertEqual({s.kind for s in report.sources}, {"verbatim", "derived", "market"})

    def test_prose_figures_are_attributed(self):
        md = self._report().full_markdown_report
        self.assertRegex(md, r"free cash flow was \$98,767M(\[\[\d+\]\]\(#cite-\d+\))+\.")

    def test_verbatim_quotes_become_sources(self):
        report = self._report()
        quotes = {s.quote for s in report.sources if s.kind == "verbatim"}
        self.assertIn("The Company is subject to complex and changing laws and regulations worldwide", quotes)
        self.assertIn("The Company faces substantial competition from Android OEMs", quotes)
        verbatim = next(s for s in report.sources if s.kind == "verbatim")
        self.assertEqual(verbatim.document_id, "doc-aapl-2025")
        self.assertEqual(len(verbatim.chunks), 1)

    def test_structured_fields_and_prompt_context_carry_no_markers(self):
        report = self._report()
        for text in (
            report.executive_summary,
            report.three_pillar_thesis.pillar_1_business_moat,
            report.three_pillar_thesis.pillar_2_financial_durability,
            report.three_pillar_thesis.pillar_3_valuation_asymmetry,
        ):
            self.assertNotIn("#cite-", text)
        context = render_prompt(
            "lead_synthesizer", ticker="AAPL", company_name="Apple Inc.", fiscal_year=2025,
            query_type="full_10k_report", financial_audit=audit_with_provenance(), business_moat=moat_with_evidence(),
            risk_audit=_create_mock_risk(), forecast=_create_mock_forecast(), dcf_valuation=_create_mock_dcf(),
        )
        self.assertNotIn("#cite-", context)

    def test_citation_failure_publishes_report_without_markers(self):
        target = "app.agents.specialized.lead_synthesizer.agent_lead_synthesizer.ReportCiter.figure_entries"
        with patch(target, side_effect=RuntimeError("boom")):
            report = self._report()
        self.assertNotIn("#cite-", report.full_markdown_report)
        self.assertEqual(report.sources, [])
        self.assertIn("Filing citations could not be built", report.full_markdown_report)

    def test_audit_rows_on_the_same_statement_share_a_marker(self):
        md = self._report(query_type="financial_audit_only").full_markdown_report
        found = re.search(r"\| Revenue(\[\[\d+\]\]\(#cite-\d+\)) \|", md)
        self.assertIsNotNone(found)
        self.assertIn(f"| Operating income{found.group(1)} |", md)
        self.assertRegex(md, r"\| Cash & equivalents\[\[\d+\]\]\(#cite-\d+\) \|")

    def test_strategist_and_risk_lines_are_marked(self):
        md = self._report().full_markdown_report
        self.assertRegex(md, r"- Android OEMs\[\[\d+\]\]\(#cite-\d+\)")
        self.assertRegex(md, r"\*\*Disclosure\*\*: Global regulatory enforcement[^\n]*\[\[\d+\]\]\(#cite-\d+\)")
        self.assertRegex(md, r"\| Current Share Price\[\[\d+\]\]\(#cite-\d+\) \|")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker exec finance_agent pytest tests/test_agent_lead_synthesizer_citations.py -v`
Expected: FAIL. `Final10KResearchReport` has no attribute `sources`, and no `#cite-` markers appear.

- [ ] **Step 3: Replace `all_citations` with `sources` in the report schema**

In `state_lead_synthesizer.py`, change the imports and the last field:

```python
"""State and Output Schemas for Lead Synthesizer Agent."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from app.agents.tools.citation_registry import SourceRef
```

Replace the `all_citations` field (lines 62–65) with:

```python
    sources: List[SourceRef] = Field(
        default_factory=list,
        description="Numbered filing sources; every [[n]](#cite-n) marker in full_markdown_report resolves to sources[n-1]",
    )
```

- [ ] **Step 4: Wire citations into the synthesizer**

In `agent_lead_synthesizer.py`:

(a) Add to the docstring after line 8: `Every filing-backed table row, quote and prose figure carries an inline [[n]](#cite-n) marker built in code (report_citations.py); the numbered sources ship as Final10KResearchReport.sources.`

(b) Replace the import on line 33 with:

```python
from app.agents.specialized.lead_synthesizer.report_citations import NullCiter, ReportCiter
from app.agents.tools.citation_registry import CitationRegistry, SourceRef, strip_markers
from app.agents.tools.figure_verification import (
    annotate_sentences,
    build_figure_index,
    collect_known_figures,
    strip_unverified_sentences,
)
```

(c) In `synthesize`, replace lines 171–184 (from `full_report_md = render_prompt(` through the `re.sub` line) with:

```python
        citer, cited_prose = self._cite_prose(prose, fiscal_year, artifacts, notes)
        full_report_md = render_prompt(
            "report_layout.j2",
            ticker=ticker,
            company_name=company_name,
            fiscal_year=fiscal_year,
            query_type=query_type,
            year_substituted=year_substituted,
            p=cited_prose,
            notes=notes,
            provenance=provenance,
            removed_sentences=removed,
            cite=citer,
            **artifacts,
        )
        full_report_md = re.sub(r"\n{3,}", "\n\n", full_report_md).strip() + "\n"
        full_report_md, sources = self._finalize_citations(citer, full_report_md)
```

(d) In the `Final10KResearchReport(...)` constructor, replace `all_citations=self._consolidate_citations(business_moat, financial_audit, forecast, risk_audit),` with `sources=sources,`.

(e) Delete the whole `_consolidate_citations` static method (lines 364–382) and add these two methods in the `# Helpers` section in its place:

```python
    @staticmethod
    def _cite_prose(
        prose: Dict[str, Optional[str]], fiscal_year: int, artifacts: Dict[str, Any], notes: List[str]
    ) -> Tuple[Any, Dict[str, Optional[str]]]:
        """Builds the report citer and appends source markers to verified prose; never raises."""
        try:
            citer = ReportCiter(CitationRegistry(), fiscal_year, **artifacts)
            index = build_figure_index(citer.figure_entries())
            cited = {
                key: annotate_sentences(text, index, citer.registry.cite_all) if text else text
                for key, text in prose.items()
            }
            return citer, cited
        except Exception as e:
            logger.exception(f"Citation build failed ({e!r}); publishing without filing markers.")
            notes.append("Filing citations could not be built for this report.")
            return NullCiter(), dict(prose)

    @staticmethod
    def _finalize_citations(citer: Any, markdown: str) -> Tuple[str, List[SourceRef]]:
        """Resolves and renumbers markers; on failure strips them so no dangling [n] is published."""
        if not isinstance(citer, ReportCiter):
            return strip_markers(markdown), []
        try:
            return citer.registry.finalize(markdown)
        except Exception as e:
            logger.exception(f"Citation finalize failed ({e!r}); publishing without filing markers.")
            return strip_markers(markdown), []
```

`notes` is the same list object that is passed to the template as `notes=notes`. So a note appended in `_cite_prose` before rendering shows up in the report footer.

- [ ] **Step 5: Add markers to the templates**

`audit_tables.j2` gets four edits:

(1) Replace line 3 (the `row` macro) with these two lines:

```
{% macro row(label, items, attr, f="{:,.1f}") %}| {{ label }}{% if cite is defined %}{{ cite.row(attr, items) }}{% endif %} |{% for it in items %} {{ n(it[attr], f) }} |{% endfor %}{{ "\n" }}{% endmacro %}
{% macro fe(attr, year) %}{% if cite is defined %}{{ cite.field(attr, year) }}{% endif %}{% endmacro %}
```

(2) Replace the body of `table_e` rows (the 15 metric lines between `| :--- | ---: |` and `{% endmacro %}`) with:

```
| Cash & equivalents{{ fe("cash_and_equivalents", bs.fiscal_year) }} | ${{ n(bs.cash_and_equivalents) }}M |
| Marketable securities{{ fe("marketable_securities", bs.fiscal_year) }} | ${{ n(bs.marketable_securities) }}M |
| Total liquid cash{{ fe("total_liquid_cash", bs.fiscal_year) }} | ${{ n(bs.total_liquid_cash) }}M |
| Total debt{{ fe("total_debt", bs.fiscal_year) }} | ${{ n(bs.total_debt) }}M |
| Net debt (negative = net cash){{ fe("net_debt", bs.fiscal_year) }} | ${{ n(bs.net_debt) }}M |
| Stockholders' equity{{ fe("stockholders_equity", bs.fiscal_year) }} | ${{ n(bs.stockholders_equity) }}M |
| Valuation share count{{ fe("valuation_shares_outstanding", bs.fiscal_year) }} | {{ n(bs.valuation_shares_outstanding) }}M ({{ bs.share_count_source }}) |
| Current ratio{{ fe("current_ratio", bs.fiscal_year) }} | {{ n(sv.current_ratio, "{:.2f}x") }} |
| EBITDA{{ fe("ebitda", bs.fiscal_year) }} | ${{ n(sv.ebitda) }}M |
| Net debt / EBITDA{{ fe("net_debt_to_ebitda", bs.fiscal_year) }} | {{ n(sv.net_debt_to_ebitda, "{:.2f}x") }} |
| Debt / equity{{ fe("total_debt_to_equity", bs.fiscal_year) }} | {{ n(sv.total_debt_to_equity, "{:.2f}x") }} |
| Effective tax rate{{ fe("effective_tax_rate_pct", bs.fiscal_year) }} | {{ n(pr.effective_tax_rate_pct, "{:.1f}%") }} ({{ pr.tax_rate_source }}) |
| ROIC ({{ pr.capital_basis }} capital, net of liquid cash){{ fe("roic_pct", bs.fiscal_year) }} | {{ n(pr.roic_pct, "{:.1f}%") }} |
| ROIC ({{ pr.capital_basis }} gross debt + equity){{ fe("roic_on_gross_capital_pct", bs.fiscal_year) }} | {{ n(pr.roic_on_gross_capital_pct, "{:.1f}%") }} |
| ROE ({{ pr.capital_basis }} equity){{ fe("roe_pct", bs.fiscal_year) }} | {{ n(pr.roe_pct, "{:.1f}%") }} |
```

(3) `financial_audit_context.j2` is **not** changed: it keeps `{% import "audit_tables.j2" as t %}` without context. That means `cite` is undefined there, and the prompt context gets no markers.

`report_layout.j2`, lines 2–3, become:

```
{% import "report_blocks.j2" as b with context %}
{% import "audit_tables.j2" as t with context %}
```

and line 106 (`{{ b.citations_table(m.citations) }}`) becomes `{{ b.evidence_note() }}`.

`report_blocks.j2` edits (each one replaces the shown line):

- Line 2: `{% import "audit_tables.j2" as t with context %}`
- Line 16: `| **Base-Case DCF Fair Value / Share**{{ cite.statements() }} | **{{ per_share(val.implied_fair_value_per_share) }}** |`
- Line 21: `| Current Share Price{{ cite.market("price") }} | {{ per_share(val.current_share_price) }}{% if val.current_share_price is not none %} ({{ val.data_quality.share_price_source }}){% endif %} |`
- Line 27: `| Discount Rate{{ cite.market("rf") }}{{ cite.market("erp") }}{{ cite.market("beta") }} | {{ pct(w.wacc_pct) }} ({{ "user override" if w.wacc_source == "user_override" else "CAPM WACC" }}) |`
- Line 35: `| Net Debt (negative = net cash){{ cite.field("net_debt") }} | {{ money(val.net_debt) }} |`
- Line 37: `| Valuation Shares{{ cite.field("valuation_shares_outstanding") }} | {{ "{:,.1f}".format(val.diluted_shares) }}M |`
- Line 52: `**Reportable segments (verified from the segment note)**{{ cite.segments() }}`
- Line 58: `**Revenue mix (verified from the disaggregation table)**{{ cite.revenue_mix() }}`
- Line 70: `| Primary moat{{ cite.evidence("primary_moat") }} | **{{ moat.economic_moat_type }}** ({{ moat.moat_assessment | replace("_", " ") }}) |`
- Line 71: `| Secondary moats{{ cite.evidence("secondary_moat") }} | {{ moat.secondary_moat_types | join(", ") if moat.secondary_moat_types else "None identified" }} |`
- Line 74: `| {{ sm.segment_name }}{{ cite.evidence("segment_moat", sm.segment_name) }} | {{ sm.moat_type }}: {{ sm.moat_rationale }} |`
- Line 77: `**Moat rationale (filing-based)**: {{ moat.moat_rationale }}{{ cite.evidence("primary_moat") }}`
- Line 90: `- {{ threat }}{{ cite.evidence("competitive_threat", threat) }}`
- Line 98: `- **Pricing power**: {{ moat.pricing_power_assessment }}{{ cite.evidence("pricing_power") }}`
- Line 99: `- **Customer concentration**: {{ moat.customer_concentration }}{{ cite.evidence("customer_concentration") }}`
- Lines 124–126 (`statements_block` loop header and title) become:

```
{% for title, key, table in (("Income Statement", "income_statement", fa.income_statement_markdown_table), ("Balance Sheet", "balance_sheet", fa.balance_sheet_markdown_table), ("Cash Flow Statement", "cash_flow", fa.cash_flow_markdown_table)) %}
{% if table %}
**{{ title }}**{{ cite.statement(key) }}
```

- Line 136: append `{{ cite.statements() }}` to the end of the `- **Baseline**: ...` line.
- Line 137: `- **Growth**: {{ fc.growth_rationale }}{{ cite.forecast_drivers() }}`
- Line 138: `- **Margins**: {{ fc.margin_rationale }}{{ cite.forecast_drivers() }}`
- Replace the `wacc_block` macro (lines 157–159) with:

```
{% macro wacc_block(val) %}
{{ val.wacc_audit.formula_breakdown_markdown }}
{% set inputs = cite.market("rf") ~ cite.market("erp") ~ cite.market("beta") %}
{% if inputs %}

_Market inputs_: risk-free rate, equity risk premium and beta{{ inputs }}
{% endif %}
{% endmacro %}
```

- In `bridge_block`, directly after line 162 (`{{ val.equity_bridge_markdown }}`), insert:

```
{% set bridge = cite.field("net_debt") %}
{% if bridge %}

_Debt, cash and share count from the audited balance sheet_{{ bridge }}
{% endif %}
```

- Line 210: `- **Disclosure**: {{ r.risk_summary }}{{ cite.risk(r) }}`
- Line 212: `- **Filing-stated mitigation**: {{ r.mitigating_factors or "None stated in the filing" }}{% if r.mitigating_factors %}{{ cite.mitigation(r) }}{% endif %}`
- Replace the whole `citations_table` macro (lines 246–256) with:

```
{% macro evidence_note() %}
Every numbered marker in this report opens the cited passage of the 10-K; the numbered source list follows the report.
{% endmacro %}
```

The line numbers above refer to the files as they were before this task. Edit from the bottom of each file up so earlier line numbers stay valid, or match on the quoted content.

- [ ] **Step 6: Update the orchestrator and the existing tests**

`orchestrator.py`:
- line 307: `"sources": [s.model_dump() for s in report.sources],`
- line 574: `"sources": [s.model_dump() for s in report_obj.sources],`
- line 659: `sources=[s.model_dump() for s in final_report.sources],`

`tests/test_agent_lead_synthesizer.py`:
- In `TestAgentLeadSynthesizer.setUp`, after `self.synthesizer = LeadSynthesizerAgent()`, add:

```python
        resolver = patch("app.agents.tools.citation_registry.resolve_chunk_metadata", return_value={})
        resolver.start()
        self.addCleanup(resolver.stop)
```

- Delete line 283 (`self.assertEqual(len(report.all_citations), 4)`). Citation coverage now lives in `test_agent_lead_synthesizer_citations.py`.

`tests/test_financial_audit_rendering.py`: `report_layout.j2` now requires `cite`, and production always passes one. In `test_02_financial_audit_instruction_and_report_layout`:
- Add `from app.agents.specialized.lead_synthesizer.report_citations import NullCiter` to the imports.
- Add `cite=NullCiter(),` to the `render_prompt("report_layout.j2", ...)` call, after `removed_sentences=0,`.

The existing exact-row assertions (`| Revenue | 96,773.0 | ...`) still hold, because `NullCiter` emits no markers.

`tests/test_full_pipeline.py`:
- Delete lines 252–258, the `all_citations = []` … `all_citations = [{"chunk_id": ...}]` block.
- Delete line 273 (`all_citations=all_citations,`).
- Delete lines 484–485 (the `# 6. Deduplicated citations` comment and its `assertGreater(len(report.all_citations), 0)`).

- [ ] **Step 7: Run the synthesizer, citer, orchestrator and pipeline tests**

Run: `docker exec finance_agent pytest tests/test_agent_lead_synthesizer_citations.py tests/test_agent_lead_synthesizer.py tests/test_report_citations.py tests/test_financial_audit_rendering.py tests/test_full_pipeline.py tests/test_unified_chat_endpoint.py -v`
Expected: all pass.
- `test_every_audit_table_attribute_is_mapped` now also covers the 15 Table E `fe(...)` attributes.
- If `rg -n "all_citations" finance_agent_backend/app finance_agent_backend/tests` finds anything, remove it. It must return no matches.

- [ ] **Step 8: Commit**

```bash
git add finance_agent_backend/app/agents/specialized/lead_synthesizer finance_agent_backend/app/agents/orchestrator.py finance_agent_backend/tests/test_agent_lead_synthesizer.py finance_agent_backend/tests/test_agent_lead_synthesizer_citations.py finance_agent_backend/tests/test_full_pipeline.py finance_agent_backend/tests/test_financial_audit_rendering.py
git commit -m "feat(citations): reports carry inline filing markers and numbered sources

Replaces Final10KResearchReport.all_citations with sources: List[SourceRef].

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Filing content and locate API

**Files:**
- Create: `finance_agent_backend/app/services/filing_content.py`
- Modify: `finance_agent_backend/app/routers/documents.py` (insert two routes directly above `@router.get("/{document_id}", ...)`; add imports)
- Test: `finance_agent_backend/tests/test_documents_content_api.py`

**Interfaces:**
- Consumes: `Document`, `DocumentChunk`, `ANNUAL_REPORT_FILING_TYPE` from `app.models`; `get_db`.
- Produces (used by Task 6):
  - `GET /api/documents/{document_id}/content` returns `{document: Document.to_dict(), toc: [{part, item, title, chunk_index}], chunks: [{id, chunk_index, part, item, breadcrumb, chunk_type, content}]}`, or 404.
  - `GET /api/documents/locate?ticker=&fiscal_year=&breadcrumb=&item=&chunk_index=` returns `{document_id, chunk_id, exact: false}`, or 404.

- [ ] **Step 1: Write the failing tests**

Create `finance_agent_backend/tests/test_documents_content_api.py`:

```python
"""Filing viewer API: whole-filing content and re-ingestion locate fallback (real PostgreSQL)."""

import unittest

from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.main import app
from app.models import Document, DocumentChunk

TICKER = "ZZCITE"
CHUNKS = [  # inserted out of order on purpose
    (2, "PART II", "Item 7", "ZZ Co (ZZCITE) > 10-K FY2099 > PART II > Item 7: MD&A", "narrative"),
    (0, "PART I", "Item 1", "ZZ Co (ZZCITE) > 10-K FY2099 > PART I > Item 1: Business", "narrative"),
    (1, "PART I", "Item 1", "ZZ Co (ZZCITE) > 10-K FY2099 > PART I > Item 1: Business", "narrative"),
    (3, "PART II", "Item 8", "ZZ Co (ZZCITE) > 10-K FY2099 > PART II > Item 8: Financial Statements", "table"),
]


def _purge(db):
    for doc in db.query(Document).filter(Document.ticker == TICKER).all():
        db.delete(doc)
    db.commit()


class TestFilingContentApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        db = SessionLocal()
        try:
            _purge(db)
            doc = Document(ticker=TICKER, company_name="ZZ Co", filing_type="10-K", fiscal_year=2099,
                           source_filename="zzcite-20991231.htm", total_chunks=len(CHUNKS))
            db.add(doc)
            db.flush()
            cls.chunk_ids = {}
            for idx, part, item, crumb, kind in CHUNKS:
                chunk = DocumentChunk(document_id=doc.id, part=part, item=item, breadcrumb=crumb, chunk_type=kind,
                                      chunk_index=idx, content=f"Chunk {idx} body", metadata_={"probe": True})
                db.add(chunk)
                db.flush()
                cls.chunk_ids[idx] = chunk.id
            db.commit()
            cls.document_id = doc.id
        finally:
            db.close()

    @classmethod
    def tearDownClass(cls):
        db = SessionLocal()
        try:
            _purge(db)
        finally:
            db.close()

    def test_content_is_ordered_and_lean(self):
        res = self.client.get(f"/api/documents/{self.document_id}/content")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["document"]["ticker"], TICKER)
        self.assertEqual([c["chunk_index"] for c in body["chunks"]], [0, 1, 2, 3])
        self.assertEqual([c["id"] for c in body["chunks"]], [self.chunk_ids[i] for i in range(4)])
        for chunk in body["chunks"]:
            self.assertEqual(set(chunk), {"id", "chunk_index", "part", "item", "breadcrumb", "chunk_type", "content"})

    def test_toc_lists_first_chunk_of_each_item(self):
        toc = self.client.get(f"/api/documents/{self.document_id}/content").json()["toc"]
        self.assertEqual(toc, [
            {"part": "PART I", "item": "Item 1", "title": "Item 1: Business", "chunk_index": 0},
            {"part": "PART II", "item": "Item 7", "title": "Item 7: MD&A", "chunk_index": 2},
            {"part": "PART II", "item": "Item 8", "title": "Item 8: Financial Statements", "chunk_index": 3},
        ])

    def test_content_unknown_document_404(self):
        self.assertEqual(self.client.get("/api/documents/does-not-exist/content").status_code, 404)

    def test_locate_prefers_same_breadcrumb_nearest_index(self):
        res = self.client.get("/api/documents/locate", params={
            "ticker": TICKER.lower(), "fiscal_year": 2099, "breadcrumb": CHUNKS[1][3], "chunk_index": 1,
        })
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"document_id": self.document_id, "chunk_id": self.chunk_ids[1], "exact": False})

    def test_locate_falls_back_to_item_then_index(self):
        by_item = self.client.get("/api/documents/locate", params={
            "ticker": TICKER, "fiscal_year": 2099, "breadcrumb": "gone", "item": "Item 8", "chunk_index": 0,
        }).json()
        self.assertEqual(by_item["chunk_id"], self.chunk_ids[3])
        by_index = self.client.get("/api/documents/locate", params={
            "ticker": TICKER, "fiscal_year": 2099, "chunk_index": 2,
        }).json()
        self.assertEqual(by_index["chunk_id"], self.chunk_ids[2])

    def test_locate_missing_filing_404(self):
        res = self.client.get("/api/documents/locate", params={"ticker": TICKER, "fiscal_year": 1900})
        self.assertEqual(res.status_code, 404)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker exec finance_agent pytest tests/test_documents_content_api.py -v`
Expected: the content tests FAIL with 404 (no route). The locate tests FAIL because `/api/documents/locate` is captured by `GET /{document_id}` and returns 404 "Document not found".

- [ ] **Step 3: Implement the service**

Create `finance_agent_backend/app/services/filing_content.py`:

```python
"""Read-side helpers for the filing viewer: a whole 10-K rebuilt from its chunks, and re-ingestion lookup."""

from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.models import ANNUAL_REPORT_FILING_TYPE, Document, DocumentChunk


def _item_title(item: str, breadcrumb: Optional[str]) -> str:
    """The breadcrumb segment naming the item (e.g. 'Item 7: MD&A'), else the bare item label."""
    for segment in (breadcrumb or "").split(" > "):
        if segment.strip().lower().startswith(item.lower()):
            return segment.strip()
    return item


def get_filing_content(db: Session, document_id: str) -> Optional[Dict[str, Any]]:
    """Every chunk of a filing in reading order plus an item table of contents; None when the filing is unknown."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if document is None:
        return None
    rows = (
        db.query(
            DocumentChunk.id, DocumentChunk.chunk_index, DocumentChunk.part, DocumentChunk.item,
            DocumentChunk.breadcrumb, DocumentChunk.chunk_type, DocumentChunk.content,
        )
        .filter(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
        .all()
    )
    toc: List[Dict[str, Any]] = []
    seen = set()
    for r in rows:
        if r.item and (r.part, r.item) not in seen:
            seen.add((r.part, r.item))
            toc.append({"part": r.part, "item": r.item, "title": _item_title(r.item, r.breadcrumb), "chunk_index": r.chunk_index})
    return {
        "document": document.to_dict(),
        "toc": toc,
        "chunks": [
            {
                "id": r.id, "chunk_index": r.chunk_index, "part": r.part, "item": r.item,
                "breadcrumb": r.breadcrumb, "chunk_type": r.chunk_type, "content": r.content,
            }
            for r in rows
        ],
    }


def locate_chunk(
    db: Session,
    ticker: str,
    fiscal_year: int,
    breadcrumb: Optional[str] = None,
    item: Optional[str] = None,
    chunk_index: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Closest chunk in the current 10-K for ticker/year: same breadcrumb, else same item, else nearest index."""
    document = (
        db.query(Document)
        .filter(
            Document.ticker == ticker.upper(),
            Document.fiscal_year == fiscal_year,
            Document.filing_type == ANNUAL_REPORT_FILING_TYPE,
        )
        .order_by(Document.created_at.desc())
        .first()
    )
    if document is None:
        return None
    rows = (
        db.query(DocumentChunk.id, DocumentChunk.chunk_index, DocumentChunk.item, DocumentChunk.breadcrumb)
        .filter(DocumentChunk.document_id == document.id)
        .all()
    )
    if not rows:
        return None
    target = chunk_index if chunk_index is not None else 0
    pool = (
        [r for r in rows if breadcrumb and r.breadcrumb == breadcrumb]
        or [r for r in rows if item and r.item == item]
        or rows
    )
    best = min(pool, key=lambda r: (abs(r.chunk_index - target), r.chunk_index))
    return {"document_id": document.id, "chunk_id": best.id, "exact": False}
```

- [ ] **Step 4: Add the routes**

In `finance_agent_backend/app/routers/documents.py`, add the import after `from app.services.ingestion.pipeline import SECIngestionPipeline`:

```python
from app.services.filing_content import get_filing_content, locate_chunk
```

Insert directly **above** `@router.get("/{document_id}", status_code=status.HTTP_200_OK)`. The literal `/locate` route has to be registered before the `/{document_id}` path parameter, or that parameter would capture it.

```python
@router.get("/locate", status_code=status.HTTP_200_OK)
def locate_filing_chunk(
    ticker: str,
    fiscal_year: int,
    breadcrumb: Optional[str] = None,
    item: Optional[str] = None,
    chunk_index: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """Re-ingestion fallback: the closest chunk in the current 10-K for a stale citation."""
    located = locate_chunk(db, ticker, fiscal_year, breadcrumb=breadcrumb, item=item, chunk_index=chunk_index)
    if located is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No 10-K filing is ingested for {ticker.upper()} FY{fiscal_year}.",
        )
    return located


@router.get("/{document_id}/content", status_code=status.HTTP_200_OK)
def get_document_content(document_id: str, db: Session = Depends(get_db)):
    """The whole filing in reading order (chunks + item table of contents) for the filing viewer."""
    content = get_filing_content(db, document_id)
    if content is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return content
```

`Optional` is already imported from `typing` in this file (`from typing import List, Optional`).

- [ ] **Step 5: Run tests to verify they pass**

Run: `docker exec finance_agent pytest tests/test_documents_content_api.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add finance_agent_backend/app/services/filing_content.py finance_agent_backend/app/routers/documents.py finance_agent_backend/tests/test_documents_content_api.py
git commit -m "feat(documents): whole-filing content and locate endpoints for the filing viewer

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Frontend citation data layer

**Files:**
- Create: `finance_agent_frontend/src/shared/types/citations.ts`
- Create: `finance_agent_frontend/src/features/filing-viewer/types.ts`
- Create: `finance_agent_frontend/src/features/filing-viewer/filingService.ts`
- Create: `finance_agent_frontend/src/features/filing-viewer/quoteHighlight.ts`
- Create: `finance_agent_frontend/src/features/filing-viewer/hooks/useFilingViewer.ts`
- Modify: `finance_agent_frontend/src/features/chat/types.ts`
- Modify: `finance_agent_frontend/src/features/chat/chatService.ts` (lines ~104 and ~141)
- Modify: `finance_agent_frontend/src/features/chat/hooks/useChat.ts` (line ~102)

**Interfaces:**
- Consumes: the backend `SourceRef` JSON (Task 4), `/api/documents/{id}/content` and `/api/documents/locate` (Task 5), and `request`/`ApiError` from `@/shared/services/apiClient`.
- Produces (used by Task 7):
  - `SourceRef`, `SourceChunk`, `SourceKind`, `isSourceRef(x): x is SourceRef`, `toSourceRefs(raw: unknown): SourceRef[]` from `@/shared/types/citations`
  - `FilingContent`, `FilingChunk`, `TocEntry`, `LocateResult` from `features/filing-viewer/types`
  - `getFilingContent(documentId)`, `locateChunk(params)` from `features/filing-viewer/filingService`
  - `highlightQuote(container: HTMLElement, quote: string): boolean` and `clearQuoteHighlight(): void`
  - `useFilingViewer()` returning `FilingViewer`: `{ source, chunkPos, filing, targetChunkId, approximate, loading, error, isOpen, openSource(ref), selectChunk(pos), close() }`

- [ ] **Step 1: Shared citation types**

Create `finance_agent_frontend/src/shared/types/citations.ts`:

```ts
/** Numbered filing source attached to an assistant message (mirrors backend SourceRef). */
export type SourceKind = 'verbatim' | 'derived' | 'market'

export interface SourceChunk {
  chunk_id: string
  document_id: string
  chunk_index: number
  ticker: string
  fiscal_year: number
  item?: string | null
  breadcrumb?: string | null
}

export interface SourceRef {
  n: number
  kind: SourceKind
  label: string
  document_id?: string | null
  ticker?: string | null
  fiscal_year?: number | null
  chunks: SourceChunk[]
  quote?: string | null
  derived_from?: string | null
  provider?: string | null
}

export function isSourceRef(value: unknown): value is SourceRef {
  if (typeof value !== 'object' || value === null) return false
  const v = value as Partial<SourceRef>
  return typeof v.n === 'number' && typeof v.kind === 'string' && Array.isArray(v.chunks)
}

/** Keeps only numbered sources; legacy citation dicts from older messages are dropped. */
export function toSourceRefs(raw: unknown): SourceRef[] {
  return Array.isArray(raw) ? raw.filter(isSourceRef) : []
}
```

- [ ] **Step 2: Switch chat types and mapping to `SourceRef`**

In `finance_agent_frontend/src/features/chat/types.ts`:
- Delete the `SourceItem` interface (lines 3–7).
- Add at the top: `import type { SourceRef } from '@/shared/types/citations'`
- Replace each of the four `sources?: SourceItem[]` with `sources?: SourceRef[]`.

In `finance_agent_frontend/src/features/chat/chatService.ts`:
- Add the import `import { toSourceRefs } from '@/shared/types/citations'`.
- Replace `sources: parsed.sources || [],` with `sources: toSourceRefs(parsed.sources),`.
- Replace `sources: m.sources || [],` with `sources: toSourceRefs(m.sources),`.

In `finance_agent_frontend/src/features/chat/hooks/useChat.ts`, `sources: replyData.sources || [],` stays valid, because `replyData.sources` is already a `SourceRef[]`. No change is needed.

- [ ] **Step 3: Filing viewer types and service**

Create `finance_agent_frontend/src/features/filing-viewer/types.ts`:

```ts
import type { IngestedDocument } from '@/features/documents/documentService'

export interface FilingChunk {
  id: string
  chunk_index: number
  part?: string | null
  item?: string | null
  breadcrumb?: string | null
  chunk_type: 'narrative' | 'table' | string
  content: string
}

export interface TocEntry {
  part?: string | null
  item: string
  title: string
  chunk_index: number
}

export interface FilingContent {
  document: IngestedDocument
  toc: TocEntry[]
  chunks: FilingChunk[]
}

export interface LocateResult {
  document_id: string
  chunk_id: string
  exact: boolean
}

export interface LocateParams {
  ticker: string
  fiscal_year: number
  breadcrumb?: string
  item?: string
  chunk_index?: number
}
```

Create `finance_agent_frontend/src/features/filing-viewer/filingService.ts`:

```ts
import { request } from '@/shared/services/apiClient'
import type { FilingContent, LocateParams, LocateResult } from './types'

export async function getFilingContent(documentId: string): Promise<FilingContent> {
  return request<FilingContent>(`/api/documents/${encodeURIComponent(documentId)}/content`)
}

export async function locateChunk(params: LocateParams): Promise<LocateResult> {
  const query = new URLSearchParams({ ticker: params.ticker, fiscal_year: String(params.fiscal_year) })
  if (params.breadcrumb) query.set('breadcrumb', params.breadcrumb)
  if (params.item) query.set('item', params.item)
  if (params.chunk_index !== undefined) query.set('chunk_index', String(params.chunk_index))
  return request<LocateResult>(`/api/documents/locate?${query.toString()}`)
}
```

- [ ] **Step 4: Quote highlighting (CSS Custom Highlight API, no DOM mutation)**

Create `finance_agent_frontend/src/features/filing-viewer/quoteHighlight.ts`:

```ts
/**
 * Marks a verbatim quote inside rendered filing text without mutating React-owned DOM, using the CSS
 * Custom Highlight API. Matching mirrors the backend's quote verification: case-insensitive, curly quotes
 * and dashes folded, whitespace runs collapsed. Returns false when the quote is not found or the browser
 * lacks the API (the chunk-level highlight still applies).
 */
const HIGHLIGHT_NAME = 'cite-quote'
const FOLD: Record<string, string> = {
  '\u2018': "'",
  '\u2019': "'",
  '\u201c': '"',
  '\u201d': '"',
  '\u2013': '-',
  '\u2014': '-',
  '\u00a0': ' ',
}

interface HighlightRegistryLike {
  set(name: string, highlight: object): unknown
  delete(name: string): boolean
}
type HighlightCtor = new (...ranges: Range[]) => object

function highlightApi(): { registry: HighlightRegistryLike; Highlight: HighlightCtor } | null {
  const registry = (CSS as unknown as { highlights?: HighlightRegistryLike }).highlights
  const Highlight = (globalThis as unknown as { Highlight?: HighlightCtor }).Highlight
  return registry && Highlight ? { registry, Highlight } : null
}

function fold(ch: string): string {
  return (FOLD[ch] ?? ch).toLowerCase()
}

export function clearQuoteHighlight(): void {
  highlightApi()?.registry.delete(HIGHLIGHT_NAME)
}

export function highlightQuote(container: HTMLElement, quote: string): boolean {
  const api = highlightApi()
  clearQuoteHighlight()
  const needle = Array.from(quote).map(fold).join('').replace(/\s+/g, ' ').trim()
  if (!api || !needle) return false

  let flat = ''
  const positions: Array<{ node: Text; offset: number }> = []
  let lastWasSpace = true
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT)
  for (let node = walker.nextNode() as Text | null; node; node = walker.nextNode() as Text | null) {
    const text = node.data
    for (let i = 0; i < text.length; i++) {
      const ch = fold(text[i])
      if (/\s/.test(ch)) {
        if (lastWasSpace) continue
        flat += ' '
        lastWasSpace = true
      } else {
        flat += ch
        lastWasSpace = false
      }
      positions.push({ node, offset: i })
    }
  }

  const start = flat.indexOf(needle)
  if (start < 0) return false
  const end = positions[start + needle.length - 1]
  const range = document.createRange()
  range.setStart(positions[start].node, positions[start].offset)
  range.setEnd(end.node, end.offset + 1)
  api.registry.set(HIGHLIGHT_NAME, new api.Highlight(range))
  return true
}
```

- [ ] **Step 5: Viewer state hook**

Create `finance_agent_frontend/src/features/filing-viewer/hooks/useFilingViewer.ts`:

```ts
import { useCallback, useRef, useState } from 'react'
import { ApiError } from '@/shared/services/apiClient'
import type { SourceRef } from '@/shared/types/citations'
import { getFilingContent, locateChunk } from '../filingService'
import type { FilingContent } from '../types'

interface ViewerState {
  source: SourceRef | null
  chunkPos: number
  filing: FilingContent | null
  targetChunkId: string | null
  approximate: boolean
  loading: boolean
  error: string | null
}

const CLOSED: ViewerState = {
  source: null,
  chunkPos: 0,
  filing: null,
  targetChunkId: null,
  approximate: false,
  loading: false,
  error: null,
}

function errorMessage(e: unknown): string {
  if (e instanceof ApiError && e.status === 404) return 'Source filing is no longer available.'
  return e instanceof Error ? e.message : 'Failed to load the filing.'
}

export function useFilingViewer() {
  const [state, setState] = useState<ViewerState>(CLOSED)
  const cache = useRef(new Map<string, FilingContent>())
  const latestRequest = useRef(0)

  const fetchFiling = useCallback(async (documentId: string): Promise<FilingContent> => {
    const cached = cache.current.get(documentId)
    if (cached) return cached
    const content = await getFilingContent(documentId)
    cache.current.set(documentId, content)
    return content
  }, [])

  const show = useCallback(
    async (source: SourceRef, chunkPos: number) => {
      const chunk = source.chunks[chunkPos]
      if (!chunk) return
      const requestId = ++latestRequest.current
      setState((prev) => ({ ...prev, source, chunkPos, loading: true, error: null, approximate: false }))
      try {
        let filing: FilingContent | null = null
        try {
          filing = await fetchFiling(chunk.document_id)
        } catch (e) {
          if (!(e instanceof ApiError && e.status === 404)) throw e
        }
        let targetChunkId = chunk.chunk_id
        let approximate = false
        if (!filing || !filing.chunks.some((c) => c.id === chunk.chunk_id)) {
          // The filing was re-ingested since this answer was written: open the closest passage instead.
          const located = await locateChunk({
            ticker: chunk.ticker,
            fiscal_year: chunk.fiscal_year,
            breadcrumb: chunk.breadcrumb ?? undefined,
            item: chunk.item ?? undefined,
            chunk_index: chunk.chunk_index,
          })
          filing = await fetchFiling(located.document_id)
          targetChunkId = located.chunk_id
          approximate = true
        }
        if (requestId !== latestRequest.current) return
        setState({ source, chunkPos, filing, targetChunkId, approximate, loading: false, error: null })
      } catch (e) {
        if (requestId !== latestRequest.current) return
        setState((prev) => ({ ...prev, filing: null, targetChunkId: null, loading: false, error: errorMessage(e) }))
      }
    },
    [fetchFiling]
  )

  const openSource = useCallback(
    (source: SourceRef) => {
      if (source.kind !== 'market') void show(source, 0)
    },
    [show]
  )

  const selectChunk = useCallback(
    (pos: number) => {
      if (state.source) void show(state.source, pos)
    },
    [show, state.source]
  )

  const close = useCallback(() => {
    latestRequest.current++
    setState(CLOSED)
  }, [])

  return { ...state, isOpen: state.source !== null, openSource, selectChunk, close }
}

export type FilingViewer = ReturnType<typeof useFilingViewer>
```

- [ ] **Step 6: Type-check and lint**

Run (from `finance_agent_frontend/`): `bun run build && bun run lint`
Expected: build succeeds and lint reports no errors. Nothing renders the new modules yet; they are only type-checked.

- [ ] **Step 7: Commit**

```bash
git add finance_agent_frontend/src/shared/types/citations.ts finance_agent_frontend/src/features/filing-viewer finance_agent_frontend/src/features/chat/types.ts finance_agent_frontend/src/features/chat/chatService.ts
git commit -m "feat(frontend): citation types, filing service, quote highlight and viewer hook

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Clickable markers, source list and filing side panel

**Files:**
- Modify: `finance_agent_frontend/src/shared/components/MarkdownRenderer.tsx`
- Create: `finance_agent_frontend/src/features/filing-viewer/SourceList.tsx`
- Create: `finance_agent_frontend/src/features/filing-viewer/FilingViewerPanel.tsx`
- Modify: `finance_agent_frontend/src/features/chat/ChatWindow.tsx` (props, destructuring, lines 197–224, lucide import)
- Modify: `finance_agent_frontend/src/App.tsx`
- Modify: `finance_agent_frontend/src/index.css` (append)

**Interfaces:**
- Consumes: `SourceRef` (Task 6), `useFilingViewer`/`FilingViewer` (Task 6), `highlightQuote`/`clearQuoteHighlight` (Task 6).
- Produces:
  - `MarkdownRenderer` props `{ content: string; sources?: SourceRef[]; onCite?: (ref: SourceRef) => void }`
  - `ChatWindow` prop `onOpenSource?: (ref: SourceRef) => void`
  - `<FilingViewerPanel viewer={FilingViewer} />`
  - `<SourceList sources onOpen />`

- [ ] **Step 1: Citation-aware `MarkdownRenderer`**

In `finance_agent_frontend/src/shared/components/MarkdownRenderer.tsx`:

Replace lines 1–9 (imports, props interface and component signature) with:

```tsx
import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { SourceRef } from '@/shared/types/citations'

interface MarkdownRendererProps {
  content: string
  /** Numbered sources of this message; `#cite-n` links render as citation markers. */
  sources?: SourceRef[]
  onCite?: (ref: SourceRef) => void
}

const CITE_HREF = /^#cite-(\d+)$/

const CitationMarker: React.FC<{ source?: SourceRef; n: string; onCite?: (ref: SourceRef) => void }> = ({
  source,
  n,
  onCite,
}) => {
  if (!source) {
    return <sup className="text-[10px] text-slate-500">[{n}]</sup>
  }
  if (source.kind === 'market' || !onCite) {
    return (
      <sup
        className="text-[10px] font-mono text-slate-400 cursor-help"
        title={source.kind === 'market' ? `Market data · ${source.provider ?? 'unknown'}` : source.label}
      >
        [{n}]
      </sup>
    )
  }
  return (
    <sup>
      <button
        type="button"
        onClick={() => onCite(source)}
        title={source.quote ? `“${source.quote}”` : source.derived_from || source.label}
        className="text-[10px] font-mono font-semibold text-sky-400 hover:text-sky-200 hover:underline px-0.5"
      >
        [{n}]
      </button>
    </sup>
  )
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, sources, onCite }) => {
```

Inside the `components={{ ... }}` object, add this entry directly before the `h1:` entry:

```tsx
          a: ({ href, children }) => {
            const cite = href ? CITE_HREF.exec(href) : null
            if (cite) {
              const source = sources?.find((s) => s.n === Number(cite[1]))
              return <CitationMarker source={source} n={cite[1]} onCite={onCite} />
            }
            return (
              <a href={href} className="text-sky-400 hover:underline">
                {children}
              </a>
            )
          },
```

- [ ] **Step 2: Numbered source list**

Create `finance_agent_frontend/src/features/filing-viewer/SourceList.tsx`:

```tsx
import React from 'react'
import { BookOpen, LineChart } from 'lucide-react'
import type { SourceRef } from '@/shared/types/citations'

interface SourceListProps {
  sources: SourceRef[]
  onOpen?: (ref: SourceRef) => void
}

export const SourceList: React.FC<SourceListProps> = ({ sources, onOpen }) => {
  if (sources.length === 0) return null
  return (
    <div className="mt-4 pt-3.5 border-t border-slate-800/80 flex flex-col gap-2">
      <span className="text-[11px] font-semibold tracking-wider uppercase text-slate-400 flex items-center gap-1">
        <BookOpen className="w-3 h-3 text-sky-400" />
        Cited Filings & Intelligence
      </span>
      <ol className="flex flex-col gap-1">
        {sources.map((s) => (
          <li key={s.n} className="text-[11px]">
            {s.kind === 'market' ? (
              <div className="flex items-center gap-1.5 px-2 py-1 text-slate-400">
                <span className="font-mono text-slate-500">[{s.n}]</span>
                <LineChart className="w-3 h-3 shrink-0" />
                <span className="truncate">
                  {s.label} · {s.provider}
                </span>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => onOpen?.(s)}
                className="w-full text-left px-2 py-1 rounded bg-[#070a12] border border-slate-800 hover:border-sky-500/40 transition-colors"
              >
                <span className="font-mono text-sky-400 mr-1.5">[{s.n}]</span>
                <span className="text-slate-200">{s.label}</span>
                {s.kind === 'verbatim' && s.quote && (
                  <span className="block truncate text-slate-500 italic">“{s.quote}”</span>
                )}
                {s.kind === 'derived' && s.derived_from && (
                  <span className="block truncate text-slate-500">{s.derived_from}</span>
                )}
              </button>
            )}
          </li>
        ))}
      </ol>
    </div>
  )
}
```

- [ ] **Step 3: Filing side panel**

Create `finance_agent_frontend/src/features/filing-viewer/FilingViewerPanel.tsx`:

```tsx
import React, { useEffect, useRef } from 'react'
import { AlertTriangle, FileText, Loader2, X } from 'lucide-react'
import { MarkdownRenderer } from '@/shared/components/MarkdownRenderer'
import type { FilingViewer } from './hooks/useFilingViewer'
import { clearQuoteHighlight, highlightQuote } from './quoteHighlight'

interface FilingViewerPanelProps {
  viewer: FilingViewer
}

export const FilingViewerPanel: React.FC<FilingViewerPanelProps> = ({ viewer }) => {
  const bodyRef = useRef<HTMLDivElement>(null)
  const { source, filing, targetChunkId, approximate, loading, error, chunkPos } = viewer

  useEffect(() => {
    if (!filing || !targetChunkId || !bodyRef.current) return
    const el = bodyRef.current.querySelector<HTMLElement>(`[data-chunk-id="${CSS.escape(targetChunkId)}"]`)
    if (!el) return
    el.scrollIntoView({ block: 'center' })
    if (source?.kind === 'verbatim' && source.quote) {
      highlightQuote(el, source.quote)
    } else {
      clearQuoteHighlight()
    }
  }, [filing, targetChunkId, source])

  useEffect(() => clearQuoteHighlight, [])

  if (!viewer.isOpen || !source) return null

  const jumpTo = (chunkIndex: number) => {
    bodyRef.current
      ?.querySelector<HTMLElement>(`[data-chunk-index="${chunkIndex}"]`)
      ?.scrollIntoView({ block: 'start' })
  }
  const doc = filing?.document

  return (
    <aside className="fixed inset-0 z-40 flex flex-col bg-[#070a12] md:static md:inset-auto md:z-auto md:w-[45%] md:max-w-[760px] md:border-l md:border-slate-800 h-screen">
      <header className="flex items-start justify-between gap-3 px-4 py-3 border-b border-slate-800">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-100">
            <FileText className="w-3.5 h-3.5 text-sky-400 shrink-0" />
            {doc ? `${doc.ticker} ${doc.filing_type} · FY${doc.fiscal_year}` : 'SEC filing'}
            <span className="font-mono text-sky-400">[{source.n}]</span>
          </div>
          <div className="text-[11px] text-slate-400 truncate">{source.label}</div>
        </div>
        <button
          type="button"
          onClick={viewer.close}
          className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-100"
          aria-label="Close filing viewer"
        >
          <X className="w-4 h-4" />
        </button>
      </header>

      {source.kind === 'verbatim' && source.quote && (
        <blockquote className="mx-4 mt-3 border-l-2 border-amber-400 bg-amber-400/5 pl-3 py-1.5 text-[11px] text-amber-100/90 italic">
          “{source.quote}”
        </blockquote>
      )}
      {source.kind === 'derived' && (
        <div className="mx-4 mt-3 text-[11px] text-slate-300">
          <div className="mb-1.5">
            Derived from {source.chunks.length} filing table{source.chunks.length === 1 ? '' : 's'}
            {source.derived_from ? ` — ${source.derived_from}` : ''}
          </div>
          {source.chunks.length > 1 && (
            <div className="flex flex-wrap gap-1">
              {source.chunks.map((c, i) => (
                <button
                  key={c.chunk_id}
                  type="button"
                  onClick={() => viewer.selectChunk(i)}
                  className={`px-2 py-0.5 rounded border text-[10px] ${
                    i === chunkPos
                      ? 'border-sky-500 text-sky-300 bg-sky-500/10'
                      : 'border-slate-700 text-slate-400 hover:border-slate-500'
                  }`}
                >
                  {c.item ?? 'Table'} · FY{c.fiscal_year}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
      {approximate && (
        <div className="mx-4 mt-3 flex items-center gap-1.5 text-[11px] text-amber-300">
          <AlertTriangle className="w-3.5 h-3.5" />
          Approximate location: the filing was re-ingested after this answer was written.
        </div>
      )}

      <div className="flex flex-1 min-h-0 mt-3">
        {filing && filing.toc.length > 0 && (
          <nav className="hidden lg:block w-44 shrink-0 overflow-y-auto border-r border-slate-800 px-2 py-2">
            {filing.toc.map((t) => (
              <button
                key={`${t.part}-${t.item}`}
                type="button"
                onClick={() => jumpTo(t.chunk_index)}
                className="block w-full text-left text-[11px] text-slate-400 hover:text-sky-300 py-1 truncate"
                title={t.title}
              >
                {t.title}
              </button>
            ))}
          </nav>
        )}
        <div ref={bodyRef} className="flex-1 overflow-y-auto px-4 py-2 space-y-3">
          {loading && (
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <Loader2 className="w-3.5 h-3.5 animate-spin" /> Loading filing…
            </div>
          )}
          {error && !loading && (
            <div className="flex items-center gap-2 text-xs text-red-400">
              <AlertTriangle className="w-3.5 h-3.5" /> {error}
            </div>
          )}
          {filing &&
            !error &&
            filing.chunks.map((c) => (
              <div
                key={c.id}
                data-chunk-id={c.id}
                data-chunk-index={c.chunk_index}
                className={
                  c.id === targetChunkId
                    ? 'rounded-md ring-1 ring-amber-400/60 bg-amber-400/5 p-2 -mx-2 scroll-mt-4'
                    : 'scroll-mt-4'
                }
              >
                <MarkdownRenderer content={c.content} />
              </div>
            ))}
        </div>
      </div>
    </aside>
  )
}
```

- [ ] **Step 4: Wire `ChatWindow`**

In `finance_agent_frontend/src/features/chat/ChatWindow.tsx`:
- Remove `ExternalLink,` from the lucide import (its only uses were the old pills).
- Add these imports below the `MarkdownRenderer` import:

```tsx
import type { SourceRef } from '@/shared/types/citations'
import { SourceList } from '@/features/filing-viewer/SourceList'
```

- Extend the props and the destructuring:

```tsx
interface ChatWindowProps {
  messages: Message[]
  isLoading: boolean
  activeMilestones?: AgentMilestone[]
  onOpenSource?: (ref: SourceRef) => void
}

export const ChatWindow: React.FC<ChatWindowProps> = ({
  messages,
  isLoading,
  activeMilestones = [],
  onOpenSource,
}) => {
```

- Replace the block from `<MarkdownRenderer content={message.content} />` through the closing `)}` of the old `message.sources && ...` pills block (lines 197–224) with:

```tsx
                          <MarkdownRenderer
                            content={message.content}
                            sources={message.sources}
                            onCite={onOpenSource}
                          />
                          <SourceList sources={message.sources ?? []} onOpen={onOpenSource} />
```

- [ ] **Step 5: Mount the panel in `App`**

In `finance_agent_frontend/src/App.tsx`:
- Add these imports:

```tsx
import { FilingViewerPanel } from '@/features/filing-viewer/FilingViewerPanel'
import { useFilingViewer } from '@/features/filing-viewer/hooks/useFilingViewer'
```

- After the `useChat` call, add: `const filingViewer = useFilingViewer()`.
- Pass `onOpenSource={filingViewer.openSource}` to `<ChatWindow ... />`.
- Directly after the closing `</div>` of the "Main Content Area" (the `flex-1 flex flex-col h-screen` div) and before the outer root `</div>`, add `<FilingViewerPanel viewer={filingViewer} />`.
- In `handleSelectConversation` and `handleNewChat`, add `filingViewer.close()` as the first statement, so a filing from another conversation doesn't stay open.

Append to `finance_agent_frontend/src/index.css`:

```css
::highlight(cite-quote) {
  background-color: rgb(250 204 21 / 0.35);
  color: #fff;
}
```

- [ ] **Step 6: Build and lint**

Run (from `finance_agent_frontend/`): `bun run build && bun run lint`
Expected: build succeeds and lint reports no errors.

- [ ] **Step 7: Manual check with a seeded demo message (no LLM calls)**

Write this script to the session scratchpad as `seed_citation_demo.py`. It is not committed. Copy it into the container with `docker cp <scratchpad>/seed_citation_demo.py finance_agent:/tmp/seed_citation_demo.py`, then run `docker exec finance_agent python /tmp/seed_citation_demo.py`:

```python
import re
from app.database import SessionLocal
from app.models import ChatMessage, Conversation, Document, DocumentChunk, User
from app.agents.tools.citation_registry import CitationRegistry, SourceSpec, marker

db = SessionLocal()
doc = (db.query(Document).filter(Document.filing_type == "10-K").order_by(Document.fiscal_year.desc()).first())
narr = (db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id, DocumentChunk.chunk_type == "narrative",
        DocumentChunk.item == "Item 1A").order_by(DocumentChunk.chunk_index).offset(3).first())
table = (db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id, DocumentChunk.chunk_type == "table",
         DocumentChunk.item == "Item 8").order_by(DocumentChunk.chunk_index).first())
quote = next(s for s in re.split(r"(?<=[.!?])\s+", narr.content) if 60 < len(s) < 300 and not s.lstrip().startswith(("#", "|")))
reg = CitationRegistry()
md = (
    f"## Citation demo ({doc.ticker} FY{doc.fiscal_year})\n\n"
    f"A risk disclosure{reg.cite(SourceSpec(kind='verbatim', chunk_ids=(narr.id,), quote=quote))}.\n\n"
    f"| Metric | Value |\n| :--- | ---: |\n"
    f"| Revenue{reg.cite(SourceSpec(kind='derived', chunk_ids=(table.id,), derived_from='Computed from the audited income statement'))} | n/a |\n\n"
    f"Share price{reg.cite(SourceSpec(kind='market', label='Share price', provider='yfinance_live'))}."
)
md, refs = reg.finalize(md)
stale = refs[0].model_copy(update={"n": len(refs) + 1, "chunks": [refs[0].chunks[0].model_copy(update={"chunk_id": "stale-chunk", "document_id": "stale-doc"})]})
md += f"\n\nA stale (re-ingested) reference{marker(stale.n)}."
user = db.query(User).filter(User.username == "finance_user").first()
conv = Conversation(user_id=user.id, title="Citation demo")
db.add(conv)
db.flush()
db.add(ChatMessage(conversation_id=conv.id, role="user", content="citation demo"))
db.add(ChatMessage(conversation_id=conv.id, role="assistant", content=md, sources=[r.model_dump() for r in refs + [stale]]))
db.commit()
print("conversation", conv.id)
```

Then run `bun run dev`, open http://localhost:5173 and select the "Citation demo" conversation. Verify:
- Marker [1] is a sky-blue button. Clicking it opens the panel on the right, scrolled to the Item 1A chunk; the chunk has an amber ring and the quoted sentence is highlighted.
- Marker [2] in the table opens the panel showing "Derived from 1 filing table — Computed from the audited income statement", with the Item 8 table chunk ringed.
- Marker [3] is grey and not clickable, and its tooltip reads "Market data · yfinance_live".
- Marker [4] opens the same Item 1A chunk with the "Approximate location" notice. This exercises the locate fallback.
- The numbered list under the message shows [1]–[4], and clicking an entry behaves like the marker.
- The ToC on the left (≥ 1024px) jumps between items, and the close button hides the panel.
- At under 768px width the panel opens full-screen and closes with the X.

Delete the demo conversation from the sidebar afterwards.

- [ ] **Step 8: Commit**

```bash
git add finance_agent_frontend/src
git commit -m "feat(frontend): clickable filing citations with numbered sources and 10-K side panel

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Full suite, docs and memory

**Files:**
- Modify: `AGENTS.md` (gitignored; local only), sections 2.4 and 3
- Modify: `finance_agent_backend/CLAUDE.md` (API contract)
- Modify: memory files under `C:\Users\satya\.claude\projects\C--Satyam-Projects-finance-agent\memory\`

- [ ] **Step 1: Run the whole backend suite**

Run: `docker exec finance_agent pytest tests/ -q`
Expected: every test passes. If anything fails, fix it before continuing and report the failure honestly.

- [ ] **Step 2: Update AGENTS.md**

- In §2.2, add this sentence to the Lead Synthesizer row's responsibility: `every filing-backed row, quote and prose figure carries an inline [[n]](#cite-n) marker; numbered SourceRef list ships as sources`.
- In §2.4, add these bullets:

```markdown
- `citation_registry.py`: Per-message `CitationRegistry` numbering `SourceSpec`s (verbatim quote / derived from statement chunks / market input) into `[[n]](#cite-n)` markers; `finalize` resolves chunk metadata in one query, drops dead sources with their markers and renumbers; `SourceRef` is the `ChatMessage.sources` contract.
```

- Extend the `figure_verification.py` bullet with `; build_figure_index / annotate_sentences attribute verified prose figures to their sources (best source per figure, max 3 markers per sentence)`.
- In §3, add: `features/filing-viewer/`: numbered source list, `#cite-n` markers in `MarkdownRenderer`, and a right-hand 10-K panel fed by `GET /api/documents/{id}/content` (with `GET /api/documents/locate` fallback for re-ingested filings); quote highlight via the CSS Custom Highlight API.

- [ ] **Step 3: Update `finance_agent_backend/CLAUDE.md` API contract**

Under "API contract", add:

```markdown
- `sources` (sync response, SSE `result`, persisted `ChatMessage.sources`) is a list of `SourceRef` (`app/agents/tools/citation_registry.py`): `{n, kind: verbatim|derived|market, label, document_id, ticker, fiscal_year, chunks: [{chunk_id, document_id, chunk_index, ticker, fiscal_year, item, breadcrumb}], quote, derived_from, provider}`. Report markdown references them as `[[n]](#cite-n)`.
- `GET /api/documents/{id}/content` -> `{document, toc, chunks}` (whole filing in `chunk_index` order); `GET /api/documents/locate?ticker&fiscal_year&breadcrumb&item&chunk_index` -> `{document_id, chunk_id, exact: false}`.
```

- [ ] **Step 4: Update memory**

- In `known-gaps-and-doc-drift.md`, rewrite the `feat/filing-citations` block. Mark the feature as implemented, and record:
  - the empty-pill bug is fixed;
  - conversational answers have no citations (no retrieval);
  - audit/valuation markers depend on `field_provenance`, so LLM-transcription fallback audits get no derived markers.
- In `agent-implementation-notes.md`, add one line under the synthesizer: `ReportCiter` (report_citations.py) + `CitationRegistry`; `cite` is only defined for report rendering, never in the prompt context.
- Keep the `MEMORY.md` index line hooks accurate.

- [ ] **Step 5: Commit**

```bash
git add finance_agent_backend/CLAUDE.md
git commit -m "docs: document SourceRef citation contract and filing viewer endpoints

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

`finance_agent_backend/CLAUDE.md` is currently untracked. If the user prefers to keep it out of git, skip this commit and leave the edit local, as with AGENTS.md.
