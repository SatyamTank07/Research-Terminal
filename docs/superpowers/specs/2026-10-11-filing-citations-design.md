# Filing Citations & 10-K Viewer — Design

Date: 2026-10-11 · Branch: `feat/filing-citations`

## 1. Goal

Every claim or figure in a chat answer that comes from (or is computed from) an SEC filing carries a
clickable numbered marker `[n]`. Clicking a marker, or the matching entry in the source list under the
message, opens a right-hand panel showing the **entire 10-K**, scrolled to the cited chunk, with the
chunk highlighted and (for verbatim quotes) the quoted sentence marked.

### Decisions (agreed in brainstorming)

| Question | Decision |
|---|---|
| Where markers appear | Inline `[n]` next to the claim/figure **and** a numbered source list under the message |
| LLM-written prose | Code attributes figures to chunks deterministically; no LLM-emitted tags, no extra LLM calls |
| Computed figures (ratios, UFCF, DCF) | Marker links to the input statement chunks ("derived from"); market inputs get a non-clickable tag |
| Viewer content | Filing rebuilt from `DocumentChunk` rows in `chunk_index` order, rendered as markdown with an Item ToC |
| Marker encoding | Approach A: per-message `CitationRegistry` + markdown fragment links `[[n]](#cite-n)` |

### Non-goals

- Rendering the original SEC `.htm` / iXBRL (pixel fidelity). Not needed; no re-ingestion.
- Markers on qualitative prose sentences that contain no filing figure (prose attribution is figure-based).
- Changing any LLM prompt or adding LLM calls (budget rule: deterministic-first).

## 2. Current state (what exists)

- Agents already emit `citations: List[Dict]` with `chunk_id, ticker, fiscal_year, item, breadcrumb`.
  Strategist and risk analyst keep **verified** verbatim quotes bound to a `chunk_id`
  (`RiskEvidence`, strategist evidence). The auditor records `field_provenance`
  (`FY<year>.<field> -> statement, row labels, chunk_id`) built in `statement_extraction.py`.
- `LeadSynthesizerAgent._consolidate_citations` flattens these into `Final10KResearchReport.all_citations`,
  which the orchestrator returns as `sources` and `chat_service` persists in `ChatMessage.sources` (JSON).
- The conversational route (`BaseAgent.run`) builds `sources` from tool-message payloads
  (`_extract_citations_from_messages`).
- Frontend `ChatWindow.tsx` renders `sources` as `<a href={src.url}>{src.title}</a>`; the backend never
  sends `url`/`title`, so the pills are currently empty — a latent bug this feature fixes.
- Report body (`report_layout.j2`) has no inline markers. Prose is screened by `figure_verification.py`
  against a **flat** context, so a figure is not traceable to a chunk today.
- Filings are 160–270 chunks / 200–370 KB of markdown each (AAPL, AMZN, NVDA, TSLA in the DB).

## 3. Backend design

### 3.1 Source record (the contract)

Every entry in `ChatMessage.sources` becomes (ordered by `n`, 1-based):

```json
{
  "n": 3,
  "kind": "verbatim | derived | market",
  "label": "Item 7 › Liquidity and Capital Resources",
  "document_id": "uuid | null",
  "ticker": "AAPL",
  "fiscal_year": 2025,
  "chunks": [
    {"chunk_id": "uuid", "chunk_index": 112, "item": "Item 7", "breadcrumb": "Item 7 > Liquidity ..."}
  ],
  "quote": "verbatim excerpt (verbatim kind only) | null",
  "derived_from": "e.g. 'ROIC FY2025 = NOPAT / avg invested capital' (derived kind only) | null",
  "provider": "e.g. 'yfinance (^TNX)' (market kind only) | null"
}
```

- `verbatim`: exactly one chunk + the verified quote.
- `derived`: one or more input chunks (statement tables) + a short formula/description.
- `market`: no chunks, no `document_id`; `provider` names the source tag already produced by
  `market_data_tools.py` / config defaults.
- `chunk_index`, `ticker`, `fiscal_year`, `breadcrumb` are stored so a reference survives re-ingestion
  (see 3.5).

Pydantic model `SourceRef` lives in `app/agents/tools/citation_registry.py`; `chunks` entries are
`SourceChunk`. The old free-form citation dicts and `Final10KResearchReport.all_citations` are
**replaced**, not kept alongside (no compat shims).

### 3.2 `CitationRegistry` (`app/agents/tools/citation_registry.py`)

One registry per assistant message (created by the lead synthesizer for research routes, by the
conversational path for Q&A).

- `cite_quote(chunk, quote) -> str` — returns the marker for a verbatim source.
- `cite_derived(chunks, derived_from) -> str` — returns the marker for a computed figure.
- `cite_market(label, provider) -> str` — returns the marker for a market input. All kinds use the same
  `[[n]](#cite-n)` encoding; the frontend decides navigability from `kind`.
- `marker(n) -> "[[n]](#cite-n)"`; multiple sources on one sentence render as `[[2]](#cite-2)[[5]](#cite-5)`.
- Dedupe key: `(kind, tuple(sorted chunk_ids), normalized quote | derived_from | provider)`; the same
  source cited twice reuses its `n`.
- `sources() -> List[SourceRef]` in `n` order.
- Chunk metadata (`document_id`, `chunk_index`, `item`, `breadcrumb`) is resolved in **one** batched DB
  query (`DocumentChunk.id IN (...)`) at `sources()` time for any chunk the producer did not fully
  describe. Unknown chunk ids are dropped from the source; a source left with no chunks is dropped and
  its markers are removed from the text (never a dangling `[n]`).

### 3.3 Producers (what gets a marker)

All deterministic; no prompt changes.

| Report element | Kind | Chunks come from |
|---|---|---|
| Strategist moat / threat / pricing / concentration evidence quotes | verbatim | verified evidence `chunk_id` + `quote` |
| Strategist segment & revenue-disaggregation tables | derived | `segment_source_chunk_ids`, `revenue_lines_source_chunk_id` |
| Risk matrix rows (evidence, mitigation evidence) | verbatim | `RiskEvidence.chunk_id` + `quote` |
| Auditor statement / ratio / forensic rows | derived | `field_provenance[FY.field].chunk_id` for each input field |
| Forecast baseline rows, MD&A guidance | derived / verbatim | baseline inputs via auditor provenance; guidance via `mdna_guidance` chunk ids |
| Forecast driver overrides with citations | verbatim | `margin/capex/nwc_citation_chunk_id` |
| DCF bridge (debt, cash, NCI, shares), UFCF, fair value | derived | auditor provenance of the input fields |
| WACC inputs: price, beta, Rf, ERP | market | source tags from `market_data_tools.py` / config |
| Synthesizer prose sentences (thesis, summaries, commentary) | derived | per-chunk figure attribution (3.4) |

A small mapping `DERIVED_INPUTS` (e.g. `roic -> [operating_income, income_tax, total_debt, equity, cash]`)
lives next to the code that computes each figure so the input list stays in sync with the formula.

### 3.4 Prose attribution (`figure_verification.py`)

- New `build_figure_index(sources: Dict[source_key, str]) -> FigureIndex`: the existing figure parser run
  **per chunk** (and per derived-figure text block), mapping each known value to the source keys that
  contain it. `collect_known_figures` becomes a view over this index, so verification behaviour is unchanged.
- New `attribute_sentence(sentence, index) -> List[source_key]`: for each figure in a verified sentence,
  the source keys whose values match under the same rounding/scale tolerance (`_matches`). Ties: prefer
  the chunk from the anchor fiscal year, then the statement table over narrative, then lowest
  `chunk_index`; at most 3 markers per sentence.
- The synthesizer's `_screen_prose` keeps verified sentences as today, then appends markers produced by
  `registry.cite_*` for the attributed source keys. Sentences whose figures appear only in computed
  context (e.g. fair value) attribute to that figure's derived source.
- Figures that match a market input (price, beta, Rf, ERP) attribute to that input's `market` source.

### 3.5 Document content API (`app/routers/documents.py`)

`GET /api/documents/{document_id}/content`

```json
{
  "document": {"id": "...", "ticker": "AAPL", "company_name": "...", "fiscal_year": 2025,
               "filing_type": "10-K", "period_end_date": "...", "total_chunks": 165},
  "toc": [{"part": "PART II", "item": "Item 7", "title": "Item 7 › MD&A", "chunk_index": 88}],
  "chunks": [{"id": "...", "chunk_index": 0, "item": "Item 1", "breadcrumb": "...",
              "chunk_type": "narrative|table", "content": "markdown"}]
}
```

- Ordered by `chunk_index`; excludes `embedding`, `content_tsv`, `metadata`.
- `toc`: first chunk of each distinct `(part, item)` in order.
- 404 when the document does not exist.

`GET /api/documents/locate?ticker=&fiscal_year=&breadcrumb=&chunk_index=` — re-ingestion fallback:
returns `{document_id, chunk_id, exact: false}` for the current 10-K of that ticker/year, choosing the
chunk with the same breadcrumb (closest `chunk_index` among them), else the closest `chunk_index` in the
same item; 404 when no filing exists.

### 3.6 Routes

- Research routes: the synthesizer owns the registry; `report_layout.j2` receives markers as
  pre-rendered strings on each row/sentence (template stays dumb). `Final10KResearchReport.sources`
  replaces `all_citations`; orchestrator/chat_service pass it through unchanged.
- Conversational route: `BaseAgent.run` builds a registry from the chunks returned by its RAG tools,
  attributes figures in the final answer (3.4) and returns `registry.sources()` as `sources`.

## 4. Frontend design

### 4.1 Feature folder `src/features/filing-viewer/`

- `types.ts` — `SourceRef`, `SourceChunk`, `FilingContent`, `TocEntry` (mirrors 3.1 / 3.5).
- `filingService.ts` — `getFilingContent(documentId)`, `locateChunk(params)` via `apiClient`.
- `hooks/useFilingViewer.ts` — open/close state, active `SourceRef`, active chunk tab, in-memory cache
  `Map<documentId, FilingContent>` (second reference into the same filing scrolls without refetch).
- `FilingViewerPanel.tsx` — right drawer (≈ 45% width on ≥ lg, full-screen sheet below `md`), header with
  ticker/FY/item and close button, Item ToC column, filing body.

### 4.2 Behaviour

- `MarkdownRenderer` overrides `a`: `href` matching `#cite-(\d+)` renders a small superscript button that
  calls `onCite(n)`; `kind === 'market'` renders a non-clickable tag with a tooltip (`provider`).
  Fragment links pass react-markdown's default `urlTransform`, so no custom scheme handling is needed.
- `ChatWindow` passes each message's `sources` to its renderer and replaces the broken pills with a
  numbered list: `[n] Item 7 › Liquidity …` (verbatim shows a one-line quote preview).
- Opening a source: fetch/cached content → scroll the first source chunk into view → highlight the chunk;
  for `verbatim`, wrap the quote in `<mark>` using whitespace/quote-normalized matching (TS port of the
  `verify_quote` normalization); if not found, chunk highlight only.
- `derived`: header "Derived from N statements — <derived_from>" with chips to jump between input chunks.
- Missing chunk id in the loaded filing → call `locate`; on success open with an
  "approximate location — filing was re-ingested" note; on 404 show "Source filing is no longer available".
- Filing body renders each chunk with the existing `MarkdownRenderer` inside a `data-chunk-id` wrapper;
  ~250 chunks per filing needs no virtualization.

### 4.3 App layout

`App.tsx` renders `FilingViewerPanel` as a sibling of the main content area so chat and filing are
side-by-side on wide screens; the left conversation sidebar is unchanged.

## 5. Error handling

| Failure | Behaviour |
|---|---|
| Producer cites an unknown chunk id | Dropped at `sources()` time; markers removed from text; logged |
| Registry/attribution raises | Synthesizer catches, emits report without markers, appends to `pipeline_notes` (failure isolation) |
| `/content` 404 | Panel inline error, chat unaffected |
| Re-ingested filing | `locate` fallback with "approximate" note |
| Old messages with legacy citation dicts | Rendered as plain unnumbered list entries without click (no backend shim) |

## 6. Testing

Backend (pytest, deterministic, no live LLM calls):
- `test_citation_registry.py`: numbering, dedupe, marker format, multi-marker sentences, unknown-chunk
  drop removes markers, `sources()` ordering, batched metadata resolution (one query).
- `test_figure_verification.py` additions: per-chunk index, figure attributed only to chunks containing it,
  rounding/scale-aware attribution (`$716.9 billion` ↔ `716,924`), tie-break order, max 3 markers,
  existing verification behaviour unchanged.
- Producer tests: strategist/risk produce `verbatim` with quote; auditor rows produce `derived` with the
  provenance chunk ids; market inputs produce `market`.
- Synthesizer test (mock LLM): every `#cite-n` in the markdown has a matching `sources[n-1]` and vice versa.
- Router tests: `/content` ordering, field exclusion, toc, 404; `locate` breadcrumb match, item fallback, 404.

Frontend: `bun run build` (tsc) + `oxlint`; manual check on an existing AAPL FY2025 full report —
click a verbatim, a derived and a market marker, and the bottom list.

## 7. Documentation

Update AGENTS.md §2.4 (new `citation_registry.py`, `figure_verification.py` attribution, documents API),
§3 (filing viewer feature) and project memory after implementation.
