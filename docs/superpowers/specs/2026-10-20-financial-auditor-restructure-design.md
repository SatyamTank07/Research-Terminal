# Financial Auditor Restructure — Design

Date: 2026-10-20 · Branch: `claude/festive-hawking-kowg5d`

## 1. Goal

Rebuild the Financial Auditor agent around **audit areas** — the same units an equity analyst works
through (revenue quality, cash quality, normalization, balance sheet, taxes, …) — so that:

- a developer can find, read, test and extend one area without touching the others;
- the LLM receives small, clearly scoped jobs instead of one long prompt with a tool loop;
- the agent covers everything a 10-K can support, including the checks it does not do today
  (controls & governance, lease / pension / maturity footnotes, tax footnote, screening scores).

### Decisions (agreed)

| Question | Decision |
|---|---|
| Organising unit | One **audit area** = one engine module (numbers) + one area spec (what to fetch, what the LLM judges) |
| LLM steps | Two tool-free calls: **Judge** (normalization, corrections, per-area verdict) then **Write** (summary from the *final* numbers). A separate **Transcribe** call runs only when statement parsing fails |
| Evidence gathering | Code fetches statements **and footnotes** up front; the LLM no longer searches in the normal path |
| Output contract | **Additive.** Every existing `FinancialAuditOutput` field keeps its name, type and meaning; new fields are added beside them |
| Scope | 10-K content only. Proxy, 8-K, Form 4, market data and peers are out of scope and listed as such in the output |
| Delivery | Four phases (§9), each mergeable on its own; phase 2 is a pure restructure with byte-identical output |

### Non-goals

- Changing what downstream agents (lead synthesizer, valuation, forecasting, risk, orchestrator) read.
- Customer concentration (owned by the Business Strategist) and risk-factor analysis (owned by the Risk
  Analyst). The auditor does not duplicate them.
- Peer benchmarking, market-value metrics, or any data outside the company's own 10-K filings.
- Changing the statement-table parser (`statement_extraction.py`) beyond adding new line-item synonyms.

## 2. Current state (what exists)

| Piece | File | Size | What it does |
|---|---|---|---|
| Agent | `specialized/financial_auditor/agent_financial_auditor.py` | 563 lines | Pre-fetch, extraction, mode switch, refine-tool factory, gap-fill, output merge, data quality, restatement notes, citations |
| Math engine | `tools/financial_math_tools.py` | 1,236 lines | Input models, history, ratios, normalization screen, quality-of-earnings analytics, 16 forensic screens, corrections |
| Parser | `tools/statement_extraction.py` | 657 lines | Table grids → standard fields, identity checks, cross-filing differences, provenance |
| Prompts | `prompt_financial_auditor.j2`, `prompt_financial_auditor_query.j2` | 75 + 29 lines | One system prompt covering verify / normalize / footnote lookup / tool call / restatements / 5-paragraph summary |
| Output | `state_financial_auditor.py` | 88 lines | `FinancialAuditOutput`, `AuditDataQuality` |

Flow today: code builds a deterministic baseline → one LangGraph tool loop (≤ 4 calls) where the LLM may
call `refine_financial_audit_tool` (or `audit_financial_metrics_tool` in fallback) and writes the summary
→ `_apply_resolution` overlays the authoritative numbers.

Pain points:

1. No notion of an "area": earnings-quality logic lives in `calculate_quality_of_earnings`,
   `screen_normalization_adjustments`, `detect_forensic_findings` and the prompt at once. Adding one
   check touches 4–5 places.
2. The LLM's single prompt mixes six jobs; failures cannot be attributed to a job.
3. The summary is written *inside* the tool loop, before code applies corrections and gap-fills
   (hence the `summary_may_be_stale` flag).
4. Footnote evidence is only fetched if the LLM decides to search.

Downstream readers (must keep working): `orchestrator.py`, `state.py`, `lead_synthesizer/*` (agent,
`audit_tables.j2`, `financial_audit_context.j2`, `report_blocks.j2`, `report_layout.j2`,
`report_citations.py`), `valuation_specialist/*`, `forecasting_analyst/*`, `risk_analyst/*`,
`business_strategist/*`, `tools/valuation_model.py`, `tools/forecast_baseline.py`, and 10 test modules
that import `financial_math_tools`.

Precedent in this repo for the new LLM pattern: the Risk Analyst, Business Strategist and Lead
Synthesizer already collect evidence in code and make tool-free LLM calls.

## 3. Target architecture

### 3.1 Pipeline

```
 collect ─▶ extract ─▶ [transcribe] ─▶ analyze ─▶ judge ─▶ assemble ─▶ write ─▶ FinancialAuditOutput
  code       code       LLM, only if     code       LLM       code        LLM
                        parsing failed
```

| Stage | Who | Input | Output |
|---|---|---|---|
| **collect** | code | ticker, FY, available years | Statement tables (target + ≤ 2 prior filings), footnote chunks requested by every area, Item 9 / 9A / auditor-report text |
| **extract** | code | statement tables | `ExtractionResult` (unchanged parser) |
| **transcribe** | LLM | raw statement tables + partial extraction | `AuditInputs` JSON; code then gap-fills from the parser (today's `_gap_fill_llm_inputs`) |
| **analyze** | code | inputs + footnotes | One `AreaResult` per area: metrics, findings, surfaced disclosures |
| **judge** | LLM | evidence pack (§5.1) | `JudgeResponse`: normalization adjustments, corrections, per-area verdicts, restatement notes |
| **assemble** | code | baseline + `JudgeResponse` | Screen adjustments / corrections, re-run engine, cap verdicts, build data quality, tables, citations |
| **write** | LLM | final numbers + verdicts | `auditor_summary` (same five labelled paragraphs as today) |

Failure behaviour (unchanged in spirit):

- **judge** fails → baseline numbers, every verdict `not_assessed`, logged in `reconciliation_log`.
- **write** fails → today's fallback summary text.
- **transcribe** fails, or no statements were found → the agent raises; the orchestrator already turns that
  into `financial_audit=None` plus a pipeline note.

### 3.2 Folder layout

Pure numbers stay in the `tools` package, because valuation and forecasting reuse them and the
`specialized` package imports every agent at init (moving math there would create import cycles).
Both sides use the **same area keys**, so a reader always knows where to look.

```
app/agents/tools/
  financial_math_tools.py          ← compatibility shim: re-exports the public names below
  audit_engine/
    __init__.py                    public API: run_audit, apply_input_corrections, models
    inputs.py                      AnnualFinancialInput, BalanceSheetInput, PriorBalanceSheetInput,
                                   NormalizationAdjustment, InputCorrection
    results.py                     result models (YearFinancialsResult, …, ForensicFinding)
    history.py                     multi-year history, growth, FCF
    revenue_quality.py             DSO, receivables vs revenue, deferred revenue / RPO trend
    cash_quality.py                accruals, Sloan, OCF/NI, SBC-adjusted FCF
    normalization.py               screen_normalization_adjustments, normalized earnings
    margins_costs.py               margins, opex vs revenue, bps changes
    balance_sheet.py               liquid cash, debt, net debt, solvency, working capital, leases, pensions
    capital_allocation.py          ROIC/ROE, capex intensity, FCF bridge, share count
    taxes.py                       effective rate, deferred tax, rate-reconciliation items
    scores.py                      Beneish, Piotroski, Altman Z'' (book), Sloan
    engine.py                      run_audit(): calls each module in order, collects findings
  statement_extraction.py          unchanged (new synonyms only)

app/agents/specialized/financial_auditor/
  __init__.py
  agent_financial_auditor.py       thin: audit() runs the pipeline stages in order
  state_financial_auditor.py       FinancialAuditOutput (+ new additive fields)
  pipeline/
    collect.py
    extract.py
    transcribe.py
    analyze.py
    judge.py
    assemble.py
    write.py
  areas/
    __init__.py                    AREAS: ordered registry of every AuditArea
    base.py                        AuditArea, AreaResult, AreaVerdict, FootnoteQuery
    integrity.py
    revenue_quality.py
    cash_quality.py
    normalization.py
    margins_costs.py
    balance_sheet.py
    capital_allocation.py
    taxes.py
    controls_governance.py
    scores.py
  prompt_financial_auditor_judge.j2            role + global rules for the judge call
  prompt_financial_auditor_judge_query.j2      renders the evidence pack from AREAS
  prompt_financial_auditor_write.j2            role + summary contract for the write call
  prompt_financial_auditor_write_query.j2
  prompt_financial_auditor_transcribe.j2       transcription rules (today's llm_fallback protocol)
  prompt_financial_auditor_area_<key>.j2       one short "YOUR TASK" block per area
```

Templates stay flat in the agent folder with the `prompt_financial_auditor_` prefix: `render_prompt`
(`specialized/prompts.py`) searches every agent folder as one namespace, so names must be globally
unique. This matches the existing convention.

## 4. The audit area contract

```python
# areas/base.py
class FootnoteQuery(BaseModel):
    label: str                        # "Debt", shown to the LLM
    keyword: str                      # passed to retrieve_10k_tables(keyword_filter=...) or narrative search
    kind: Literal["table", "narrative"]
    section_item: str = "Item 8"

class AreaVerdict(BaseModel):
    verdict: Literal["clean", "watch", "concern", "not_assessed"]
    comment: str = ""                 # 1–3 sentences, figures only from the area's metrics
    evidence_chunk_ids: List[str] = []

class AreaResult(BaseModel):
    key: str
    title: str
    metrics: Dict[str, Any]           # area-specific, JSON-safe, the numbers the LLM may cite
    findings: List[ForensicFinding]
    disclosures: List[Dict[str, Any]] # code-surfaced footnote excerpts: {chunk_id, label, text}
    verdict: AreaVerdict = AreaVerdict(verdict="not_assessed")

class AuditArea(ABC):
    key: ClassVar[str]
    title: ClassVar[str]
    summary_paragraph: ClassVar[str]  # which of the 5 summary paragraphs it feeds (§6)
    footnotes: ClassVar[List[FootnoteQuery]] = []
    prompt: ClassVar[Optional[str]]   # "prompt_financial_auditor_area_balance_sheet.j2", or None

    @abstractmethod
    def analyze(self, audit: Dict[str, Any], ctx: AuditContext) -> AreaResult: ...
```

`AuditContext` holds the extraction result, the collected footnotes keyed by `FootnoteQuery.label`, and
the target year. `analyze` is pure code: it reads the engine output (and footnotes for the new checks)
and returns metrics + findings. It never calls the LLM.

**Adding a check** = edit one engine module + its area class (+ its prompt block if the LLM should
judge it) + tests. No changes to the agent, the pipeline or other areas.

## 5. LLM contracts

### 5.1 Judge (one call, no tools)

Prompt = `prompt_financial_auditor_judge.j2` + `prompt_financial_auditor_judge_query.j2`. The query renders, in area order:

```
## AREA 6 — Balance Sheet & Liquidity          (key: balance_sheet)
METRICS (authoritative, do not recompute):
  net_debt: -12,400 | current_ratio: 1.8 | lease_liabilities: 3,100 | ...
FINDINGS:
  [medium] DEBT_MATURITY_CONCENTRATION FY2027 — 38% of debt matures within 2 years
DISCLOSURES:
  [chunk:abc123] Debt — "…$4.2 billion of senior notes mature in fiscal 2027…"
YOUR TASK:
  <contents of prompt_financial_auditor_area_balance_sheet.j2>
```

followed by the clean statement tables (as today) and the cross-filing differences.

Response schema (validated with Pydantic; one repair retry on invalid JSON, then fail → §3.1):

```json
{
  "corrections": [InputCorrection, ...],
  "normalization_adjustments": [NormalizationAdjustment, ...],
  "areas": {
    "<area_key>": {"verdict": "clean|watch|concern", "comment": "...", "evidence_chunk_ids": ["..."]}
  },
  "restatement_notes": ["..."]
}
```

Global rules carried over from today's prompt: never compute numbers; corrections only when a value
contradicts a table row, with `source_chunk_id`; the existing normalization rules (discrete items only,
positive `amount`, `direction`, `affects`, no aggregate subtotals, no recurring costs, no tax provision).

### 5.2 Assemble (code) — guards on the LLM

- Corrections → `apply_input_corrections` (unchanged). Adjustments → `screen_normalization_adjustments`
  (unchanged). Engine re-runs once with the accepted set.
- **Verdict caps:** an area with a `high` finding cannot be `clean` (raised to `concern`); an area with a
  `medium` finding cannot be `clean` (raised to `watch`). Every cap is logged.
- `evidence_chunk_ids` not present in that area's disclosures or the statement chunks are dropped.
- Areas the LLM omitted stay `not_assessed`.

### 5.3 Write (one call, no tools)

Input: the **final** engine output, the capped verdicts and comments, and accepted/rejected adjustments.
Output: `auditor_summary`, the same five bold-labelled paragraphs as today (lead synthesizer and tests
depend on the headings). Each area feeds one paragraph (§6). Figures must come from the input; the
existing `figure_verification.py` screen (today used by the lead synthesizer) is applied to the result.

Because the summary is written after all corrections, `summary_may_be_stale` becomes always `false` in
the deterministic path; the field stays for compatibility.

### 5.4 Transcribe (only when parsing fails)

One tool-free call over the pre-fetched raw tables; schema = today's `audit_financial_metrics_tool`
inputs. Code gap-fills from the parser, then the pipeline continues at **analyze** exactly like the
deterministic path (so fallback runs also get judge + write).

**Behaviour change:** today, if pre-fetch returns no tables, the LLM searches with
`retrieve_10k_tables_tool`. In the new flow, collect retries once without the `Item 8` filter
(`retrieve_10k_tables` already supports `section_item=None`); if there are still no tables the agent
raises. `test_06_llm_fallback_without_parsed_statements` is rewritten accordingly.

## 6. Areas

`●` = exists today (moved, unchanged) · `+` = new in phase 4.

| # | Key | Summary paragraph | Engine content | Findings | Footnotes / sections collected |
|---|---|---|---|---|---|
| 1 | `integrity` | Forensic Findings & Monitoring | ● identity checks, completeness, cross-filing differences | ● `DATA_QUALITY` | — |
| 2 | `revenue_quality` | Earnings Quality | ● DSO, receivables vs revenue · + deferred revenue / RPO trend, rev-rec policy excerpt | ● `RECEIVABLES_DIVERGENCE` · + `DEFERRED_REVENUE_DECLINE` | + Revenue (narrative), Contract balances / RPO (table) |
| 3 | `cash_quality` | Cash Flow Durability | ● accruals, Sloan, OCF/NI, FCF conversion, SBC, SBC-adjusted FCF | ● `HIGH_ACCRUALS`, `CHRONIC_ACCRUAL_DEFICIT`, `EARNINGS_QUALITY_DIVERGENCE`, `WEAK_CASH_CONVERSION`, `SBC_INTENSITY` | — |
| 4 | `normalization` | Earnings Quality | ● normalized operating income / net income, non-operating share | ● `NON_CORE_INCOME_DEPENDENCE` | + Restructuring, Impairment, Contingencies (tables) |
| 5 | `margins_costs` | Margins & Cost Structure | ● margins, bps changes, opex vs revenue growth · + capitalized cost & useful-life policy excerpts | ● `MARGIN_COMPRESSION`, `NEGATIVE_OPERATING_LEVERAGE` · + `USEFUL_LIFE_CHANGE` | + Property and equipment, Capitalized software (narrative) |
| 6 | `balance_sheet` | Balance Sheet & Returns | ● liquid cash, debt, net debt, current ratio, D/E, net debt/EBITDA, DSO/DIO/DPO/CCC · + lease liabilities, pension funded status, debt maturity ladder, goodwill % assets | ● `NEGATIVE_BOOK_EQUITY`, `INVENTORY_ACCUMULATION`, `WORKING_CAPITAL_DETERIORATION` · + `DEBT_MATURITY_CONCENTRATION`, `GOODWILL_HEAVY`, `PENSION_UNDERFUNDED` | + Debt, Leases, Pension, Commitments and contingencies, Goodwill (tables) |
| 7 | `capital_allocation` | Balance Sheet & Returns | ● ROIC (avg/ending basis), ROE, capex/revenue, capex/D&A, FCF bridge, share count | ● `CAPEX_DRIVEN_FCF`, `UNDERINVESTMENT`, `SHARE_DILUTION` | + Item 5 repurchase table |
| 8 | `taxes` | Earnings Quality | ● effective rate, deferred tax / NI · + rate-reconciliation items, valuation allowance, cash taxes paid, unrecognized tax benefits | ● `TAX_ANOMALY` · + `VALUATION_ALLOWANCE_RELEASE`, `CASH_TAX_GAP` | + Income taxes (tables) |
| 9 | `controls_governance` | Forensic Findings & Monitoring | + auditor opinion type, critical audit matters, going-concern language, Item 9A conclusion, Item 9 auditor change, newly adopted standards / accounting changes | + `ADVERSE_ICFR`, `MATERIAL_WEAKNESS`, `GOING_CONCERN`, `AUDITOR_CHANGE`, `ACCOUNTING_CHANGE` | + Item 8 auditor report, Item 9, Item 9A (narrative) |
| 10 | `scores` | Forensic Findings & Monitoring | ● Sloan (moved from cash_quality metrics, kept there too) · + Beneish M-Score, Piotroski F-Score, Altman Z'' (book-value, non-manufacturer form) | + `BENEISH_FLAG`, `ALTMAN_DISTRESS` | — |

Notes:

- Existing finding codes, thresholds and messages are **unchanged**; each finding gains an optional
  `area` field (default `None` → filled by the area that produced it).
- Controls & governance detection is code-first: sentence screens over the collected sections (reusing
  `risk_disclosure_tools.split_sentences` / `locate_section_chunks`), with verbatim excerpts. The LLM only
  writes the verdict comment. A finding is raised only on an explicit phrase match (e.g. "material
  weakness", "substantial doubt about … ability to continue as a going concern", "not effective").
- Scores report `null` with a reason when an input is missing (e.g. Piotroski needs three year-end
  balance sheets → requires a prior filing; Beneish needs SG&A and net PP&E).
- New parser synonyms needed (phase 4): `selling_general_administrative`, `property_plant_equipment_net`,
  `retained_earnings`, `goodwill`, `operating_lease_liabilities`, `deferred_revenue`. Missing fields
  degrade only the metric that needs them.

## 7. Output additions (additive)

```python
class ScreeningScores(BaseModel):
    beneish_m_score: Optional[float] = None
    piotroski_f_score: Optional[int] = None
    altman_z_double_prime: Optional[float] = None
    sloan_accruals_pct: Optional[float] = None
    notes: List[str] = []                      # why a score is null

class AuditAreaSummary(BaseModel):
    key: str
    title: str
    verdict: AreaVerdict
    metrics: Dict[str, Any]
    finding_codes: List[str]
    disclosures: List[Dict[str, Any]]

class FinancialAuditOutput(BaseModel):
    ...                                        # every existing field unchanged
    areas: List[AuditAreaSummary] = []
    screening_scores: Optional[ScreeningScores] = None

class AuditDataQuality(BaseModel):
    ...                                        # every existing field unchanged
    not_covered: List[str] = []                # e.g. "Executive compensation (proxy statement)"
```

`ForensicFinding` gains `area: Optional[str] = None`. All additions have defaults, so existing
constructors, fixtures and templates are unaffected. Rendering the new fields in the report (lead
synthesizer templates) is a follow-up, not part of this spec.

`not_covered` is a fixed list: executive compensation and related-party detail (proxy), management
changes and late filings (8-K / NT 10-K), insider transactions (Form 4), market-value metrics and cost of
capital (market data), peer benchmarking.

## 8. Error handling

| Failure | Behaviour |
|---|---|
| Footnote query returns nothing | Area runs with statement data only; `disclosures=[]`; metric that needs it is `null`; warning in `data_quality.warnings` |
| One area's `analyze` raises | Logged; that area is `not_assessed` with a warning; other areas and the engine output are unaffected |
| Judge returns invalid JSON | One repair retry; then baseline numbers, all verdicts `not_assessed` |
| Judge cites unknown chunk ids | Dropped (§5.2) |
| Write fails | Fallback summary text (today's wording) |
| Engine rejects corrected inputs | Corrections reverted to baseline, logged (today's behaviour) |

## 9. Delivery plan

Each phase is one PR, green on the full backend test suite before the next starts.

**Phase 1 — this spec.**

**Phase 2 — pure restructure (no behaviour change).**
1. Before moving code, add a golden test: run the auditor on the `statement_fixtures` inputs with the LLM
   mocked, serialize `FinancialAuditOutput` to JSON, and commit it as the expected output.
2. Split `financial_math_tools.py` into `audit_engine/` modules; keep `financial_math_tools.py` as a
   re-export shim so the other importers keep working.
3. Split the agent into `pipeline/collect.py`, `extract.py`, `assemble.py` (today's `_apply_resolution`
   helpers); the LLM step stays the current tool loop for this phase.
4. Introduce `areas/` with the nine areas that exist today (all except `controls_governance`; `scores`
   holds only Sloan). Their `analyze` only regroups existing engine output and is not yet written to
   the output. The golden test and all existing tests must pass unchanged.

**Phase 3 — new LLM flow.**
Replace the tool loop with transcribe / judge / write (§5), add the `areas` output, `ForensicFinding.area`
and verdict caps.
Rewrite the auditor agent tests that mock the tool loop; add tests for the judge schema, caps, chunk-id
filtering and each failure row in §8.

**Phase 4 — new coverage, one PR per row, in this order:**
`controls_governance` → `balance_sheet` footnotes → `taxes` footnotes → `scores` →
`revenue_quality` footnotes → `margins_costs` policy excerpts.
Each adds parser synonyms if needed, engine functions, the area's prompt block and unit tests on fixture
tables.

## 10. Testing

- **Golden output test** (phase 2) guards the restructure.
- **Engine modules:** existing `test_tool_financial_math.py` cases keep importing from the shim; new
  modules get their own tests (`test_audit_engine_<area>.py`).
- **Areas:** each `AuditArea.analyze` tested with fixture engine output + fixture footnotes.
- **Pipeline:** judge/write/transcribe with a mocked LLM returning valid, invalid and adversarial JSON
  (unknown chunk ids, `clean` verdict on an area with a high finding, rejected adjustments described as
  applied).
- **Contract:** a test asserts every pre-existing `FinancialAuditOutput` field name and type is still
  present, so downstream readers are protected.
- **Full pipeline:** `test_full_pipeline.py` and the lead-synthesizer tests stay green in every phase.

## 11. Open points (to settle while implementing, not blocking)

- Thresholds for the new findings (maturity concentration, goodwill-heavy, cash-tax gap, Beneish and
  Altman cut-offs) — start from the textbook values (Beneish > −1.78, Altman Z'' < 1.1) and record them as
  constants next to the existing thresholds.
- Whether the lead-synthesizer report should render `areas` verdicts — separate follow-up.
