# Business Strategist — Module Structure — Design

Date: 2026-10-18 · Branch: `claude/focused-curie-3kjlay`

## 1. Goal

Reorganise the Business Strategist so that it is easy to understand for people and for the LLM, and
so that the missing 10-K analysis can be added without growing one large file and one large prompt.

The agent is organised the way an equity analyst works: **one module per analyst question**, and
every module has the same four steps — **gather (code) → judge (LLM, optional) → verify (code) →
output block**. Read one module and you understand all of them.

### Decisions (agreed in discussion)

| Question | Decision |
|---|---|
| Sub-agents or modules? | **Modules inside one agent.** The orchestrator still sees one `business_strategist` node; no new graph nodes, no tool-calling LLM loops |
| Who decides what to read | Code. Each module knows which 10-K sections it needs; the LLM only judges what it is given |
| LLM calls | Grouped by filing input: **2 call groups** (Item 1 group, MD&A group) instead of 1 call today; each prompt is focused |
| Cost | LLM cost is **not** a design constraint: quality settings (stronger model, more output tokens, full filing text, repair and skeptic passes) are on by default, see §3.8 |
| Verification | Existing rules kept unchanged (verbatim quotes, mechanism relevance, "Wide" needs 2 quotes, numbers must appear in source) and extended to the new fields |
| Output compatibility | Existing `BusinessMoatOutput` field names stay; new data is added as new optional fields |
| Order of work | Phase 1 is a behaviour-preserving restructure; new analysis comes in later phases |

### Non-goals

- Turning modules into autonomous sub-agents or adding graph nodes.
- Data outside SEC filings: industry size, market share, peer data, primary research, web search.
- Risk-factor analysis (owned by the Risk Analyst) and statement auditing (owned by the Financial Auditor).
- Changing the Lead Synthesizer's report layout beyond rendering the new fields (separate spec if needed).

## 2. Current state (what exists)

- `agent_business_strategist.py` (838 lines) does everything: locate Item 1, load the filing, parse
  segment / revenue tables, screen concentration and pricing sentences, run **one** tool-free LLM
  call, verify every claim, assemble `BusinessMoatOutput` and citations.
- One system prompt (`prompt_business_strategist.j2`) and one query prompt cover all topics at once.
- Coverage of the ~25 business-strategy inputs a 10-K offers: 10 used, 7 partly, 8 unused.
  Main gaps:
  - **Free-text prose is not verified**: `business_summary`, `revenue_architecture`, rationales,
    segment descriptions and growth drivers are accepted as written.
  - **Moat is quote-based only**: never checked against ROIC / margin history, which the Financial
    Auditor already computes (ROIC for the latest year only).
  - **MD&A is not read**: no segment explanations, KPIs, or strategy priorities.
  - **Output not used by the Forecaster**, although `segment_details` is documented as feeding it.
  - Single filing only, so `moat_trajectory` is usually Stable / Indeterminate.
  - Unused: deferred revenue / RPO, buyback and dividend cash, M&A outcome, R&D intensity,
    suppliers, seasonality, backlog.
- Pipeline: `supervisor → {business_strategist, financial_auditor, risk_analyst}` run in parallel and
  join at `forecasting_analyst`. In the `business_moat_only` route the auditor does not run.
- Consumers of `BusinessMoatOutput`: `lead_synthesizer` (prose fallbacks, `report_layout.j2`) and
  `report_citations.py` (evidence, `segment_financials`, `revenue_lines`). No frontend dependency.

## 3. Design

### 3.1 Module contract

Every module implements the same small interface (`shared/base_module.py`):

```python
class StrategyModule(Protocol):
    name: str                                   # "m1_business_model"
    call_group: Optional[str]                   # "item1" | "mdna" | None (code-only)

    def gather(self, ctx: FilingContext) -> ModuleInput: ...          # code: read its 10-K part
    def prompt_section(self, inp: ModuleInput) -> Optional[str]: ...  # its block of the LLM prompt
    def verify(self, inp, raw: dict, ev: EvidencePool) -> ModuleResult: ...  # code: keep/drop/downgrade
```

- `FilingContext` is built once per run (filing chunks, Item 1, MD&A, inferred items, extracted tables)
  and shared by all modules, so nothing is loaded twice.
- `ModuleResult` carries the module's output fields, its verified evidence, and its own
  `status` (`ok | partial | not_assessed`) and notes.
- A module never raises: a failure marks only that module `not_assessed`, the rest still run.

### 3.2 The modules

Each module answers one analyst question. "Existing" = moved from today's code; "New" = added in a later phase.

#### M1 — Business Model · *What does it sell, to whom, how does it make money?*
- **Gather:** Item 1; segment note; revenue disaggregation (existing extraction).
- **Judge (item1 group):** business summary, revenue model, product-line descriptions and growth
  drivers (existing); **customers & channels** (new).
- **Verify:** names match code-extracted segments / lines (existing). New: summary, revenue model,
  descriptions and drivers must each cite ≥1 verified quote; an unsupported sentence is removed, and
  a fully unsupported field falls back to the deterministic text.
- **Output:** `business_summary`, `revenue_architecture`, `reportable_segments`,
  `primary_product_segments`, `segment_details`, `segment_financials`, `revenue_lines`, both markdown
  tables (existing); `customers_and_channels` (new).

#### M2 — Segment Performance & Drivers · *What is growing, why, and which numbers matter?*
- **Gather:** multi-year segment and revenue-line figures (existing); MD&A segment / results
  discussion, located like Item 1 (new).
- **Judge (mdna group):** per segment / line, why revenue changed; company KPIs (name, value, period);
  price-vs-volume statements — each with a quote.
- **Verify:** quote verification; every KPI value and every figure in an explanation must appear in
  the quoted chunk (`numbers_supported`); segment names must match M1.
- **Output (new):** `segment_drivers: List[SegmentDriver]`, `operating_kpis: List[OperatingKpi]`,
  `price_volume_notes: List[str]`.
- **Downstream (new):** the Forecaster reads `segment_drivers` / segment growth as context for its
  growth drivers (still clamped by its existing code bounds).

#### M3 — Competition & Moat · *Who does it compete with, and what protects it?*
- **Gather:** Item 1 (existing); deferred revenue / RPO from the statements and notes (new);
  prior-year Item 1 when available (new).
- **Judge (item1 group):** primary / secondary / segment moats, durability, competitive threats
  (existing); trajectory using the prior-year comparison (new).
- **Verify:** all existing moat and threat rules unchanged. New: `Expanding` / `Deteriorating`
  trajectory requires a quote showing the change versus the prior year, else `Stable` /
  `Indeterminate`.
- **Output:** all existing moat and threat fields; `contracted_revenue` (new: deferred revenue and
  RPO figures with citations).

#### M4 — Moat Numbers Check · *Do the numbers prove the moat?* (code only)
- **Inputs:** M3's moat rating + Financial Auditor output (`multi_year_history`, ROIC, margins).
- **Rules (deterministic):**
  - "Wide" requires ROIC above a fixed hurdle in each available year and no sustained margin
    compression finding; otherwise capped at "Narrow".
  - A verified moat with ROIC below the hurdle in every available year is flagged
    `numbers_contradict` (rating kept, flag shown in the report).
  - No auditor output (e.g. `business_moat_only` route) → `not_assessed`.
- **Output (new):** `moat_numbers_check: MoatNumbersCheck` (status, years checked, ROIC / margin
  series, rule outcomes, adjustments made).
- **Prerequisite:** the Auditor must expose ROIC per year (today: latest year only).
- **Where it runs:** see §3.7 — it needs both agents' outputs, so it cannot run inside the
  strategist's parallel branch.

#### M5 — Market Power · *Can it set prices, and who is it dependent on?*
- **Gather:** code-screened concentration and pricing sentences (existing); supplier, seasonality and
  backlog sentences (new, same screening approach).
- **Judge (item1 group):** pricing power, customer concentration (existing); supplier dependence,
  seasonality, backlog (new).
- **Verify:** existing pricing / concentration rules and deterministic fallback; new fields follow the
  concentration pattern (verified quote or the screened sentence verbatim, else "Not explicitly
  disclosed in 10-K").
- **Output:** `pricing_power_assessment`, `customer_concentration` (existing);
  `supplier_dependence`, `seasonality`, `backlog` (new).

#### M6 — Capital Allocation · *What does management do with the cash?* (code only)
- **Gather:** cash flow statement (buybacks, dividends, acquisitions), goodwill and impairments,
  R&D expense, executive officers list (Items 10–14 when present).
- **Judge:** none in this spec; text is a deterministic template from the figures.
- **Verify:** figures must come from parsed statement rows with a source chunk.
- **Output (new):** `capital_allocation: CapitalAllocation` (multi-year buybacks, dividends, M&A
  spend, R&D % of revenue, goodwill impairments, summary line, citations).

### 3.3 LLM call groups

Modules that read the same filing section share one call, so the filing text is sent once:

| Call | Modules | Filing input | Notes |
|---|---|---|---|
| `item1` | M1, M3, M5 | Item 1 + screened sentences + tables | Today's call, prompt split into one labelled section per module |
| `mdna` | M2 | MD&A + segment / revenue tables | New in Phase 3 |
| — | M4, M6 | none | Code only |

- Each module owns its **prompt section** (`modules/<m>/prompt.j2`) and its **slice of the JSON
  answer** (`"business_model": {...}`, `"moat": {...}`, `"market_power": {...}`). The call group
  concatenates sections under one shared system prompt (role + verification rules + quote format).
- The two calls are independent and run concurrently.
- One call failing marks only its modules `not_assessed`; deterministic parts are still returned.

### 3.4 Shared layer (`shared/`)

| File | Content (moved from today's agent unless noted) |
|---|---|
| `context.py` | `FilingContext`: Item 1 locate + RAG fallback, filing load, inferred items, table extraction, MD&A locate (new) |
| `evidence.py` | `EvidencePool` and `verify_quotes` (today's `_verify`), quote / number checks |
| `llm.py` | `run_call_group(system, sections, callbacks)` — the single tool-free JSON call (today's `_run_judgement_llm`) |
| `base_module.py` | `StrategyModule` protocol, `ModuleInput`, `ModuleResult` |
| `assembly.py` | Maps module results to `BusinessMoatOutput`, builds citations and `data_quality` |

### 3.5 Folder layout

```
business_strategist/
├── agent_business_strategist.py   # analyze(): build context → run call groups → verify modules → assemble
├── state_business_strategist.py   # BusinessMoatOutput (+ new optional fields)
├── prompt_business_strategist.j2  # shared system prompt: role, quote rules, output envelope
├── shared/
│   ├── context.py
│   ├── evidence.py
│   ├── llm.py
│   ├── base_module.py
│   └── assembly.py
└── modules/
    ├── m1_business_model/        # module.py · prompt.j2 · schema.py
    ├── m2_performance_drivers/
    ├── m3_competition_moat/
    ├── m4_moat_numbers_check/    # module.py · schema.py (no prompt)
    ├── m5_market_power/
    └── m6_capital_allocation/    # module.py · schema.py (no prompt)
```

`agent_business_strategist.py` keeps the registry name, class name and `analyze()` signature, so the
orchestrator is unchanged in Phase 1.

### 3.6 Output schema and compatibility

- All current `BusinessMoatOutput` fields keep their names, types and meaning.
- New fields are optional with empty defaults: `customers_and_channels`, `segment_drivers`,
  `operating_kpis`, `price_volume_notes`, `contracted_revenue`, `moat_numbers_check`,
  `supplier_dependence`, `seasonality`, `backlog`, `capital_allocation`.
- `StrategyEvidence.claim` gains the new claim kinds (e.g. `business_summary`, `segment_driver`,
  `kpi`, `trajectory`, `supplier_dependence`).
- `StrategistDataQuality` gains `module_status: Dict[str, str]`; existing counters stay.
- Lead Synthesizer and `report_citations.py` keep working untouched; rendering the new fields is a
  follow-up per phase.

### 3.7 Pipeline integration

- **M4 placement.** Strategist and auditor run in parallel and join at `forecasting_analyst_node`.
  Recommended: `forecasting_analyst_node` calls the pure function
  `moat_numbers_check(business_moat, financial_audit)` first and writes the result back into
  `business_moat`, so the Forecaster and Synthesizer both see it. In `business_moat_only` the
  `lead_synthesizer_node` calls it with `financial_audit=None` → `not_assessed`.
- **Forecaster link.** `forecasting_analyst_node` passes `business_moat` to the Forecaster; the
  Forecaster adds segment growth and drivers to its MD&A context. Its code bounds on growth and margins
  are unchanged.
- **Prior-year Item 1.** The orchestrator passes `available_fiscal_years` to the strategist (as it
  already does for the auditor and risk analyst).

### 3.8 Quality mode (cost is not a constraint)

Today's limits were set to keep cost low, and several of them reduce quality:

| Limit today | Where | Effect |
|---|---|---|
| `gpt-4o-mini` | default in `agents/base.py` | Small model for subtle moat / durability judgement |
| 2,000 output tokens | `JUDGEMENT_MAX_TOKENS` | Crowded answer; truncated JSON fails to parse and **all** qualitative fields become "Not assessed" |
| Item 1 cut to 90,000 chars | `ITEM1_CHAR_BUDGET` | Later paragraphs (often competition, IP) are lost on long filings |
| Single attempt | — | A rejected claim is dropped, never fixed |
| ≤ 2 quotes per claim, 2 secondary moats, 4 threats | `MAX_*` constants | Less evidence; "Wide" harder to support |
| Keyword-only relevance check | `claim_relevance.py` | Real but differently worded quotes are dropped |

With cost removed, each call group runs this pipeline (every step still passes the same code checks):

```
judge (strong model, ~8k out, full text) → verify → repair rejected → verify
      → skeptic pass (moat only) → verify → final
```

| Setting | Change | Gain |
|---|---|---|
| **Model** | Stronger reasoning model for the judgement calls (configurable per call group) | Better moat, durability and pricing-power judgement |
| **Output tokens** | 2,000 → ~8,000 | No truncated JSON; room for more quotes, threats and segment details |
| **Input** | Full Item 1, full MD&A, prior-year Item 1; no char budget | Nothing cut off; trajectory possible |
| **Evidence limits** | 3–4 quotes per claim, 3 secondary moats, 6 threats | Stronger evidence trail |
| **Repair pass** | Rejected claims sent back once with the rejection reason; re-verified with the same rules | Claims with one bad quote survive. Same pattern as the Risk Analyst (`_repair_coverage`, `prompt_risk_analyst_repair.j2`) |
| **Skeptic pass** | One extra call argues against the moat using filing quotes (competition, price pressure, customer losses); verified counter-evidence lowers durability | The analyst's bear-case check; counters over-generous ratings |
| **Agreement (optional)** | Judgement run 3×; moat type kept only if 2 of 3 agree; durability takes the most cautious value | Stable ratings between runs |
| **LLM relevance check** | Cheap LLM call judges quote–claim support; keyword check stays as a floor, verbatim check unchanged | Fewer good quotes discarded |

What does **not** change with a higher budget: code verification of every quote and number,
temperature 0, and numbers coming only from code. More tokens do not prevent hallucination; the checks do.

**Configuration.** All settings live in one `StrategistQualityConfig` (model per call group, max output
tokens, input budgets, evidence limits, `repair`, `skeptic`, `agreement_runs`), so quality can be tuned
without touching module code. Defaults are the quality values above; a `lean` preset reproduces today's
behaviour (used by the Phase 1 equivalence tests).

**Estimated calls per run:**

| Mode | Calls |
|---|---|
| Today | 1 |
| Modules, lean | 2 |
| Modules, quality (default) | ~5–6 (2 groups × judge + repair, + 1 skeptic, + relevance checks) |
| Quality + agreement (3×) | ~10–12 |

## 4. Phases

| Phase | Scope | Done when |
|---|---|---|
| **1. Restructure** | Shared layer + M1, M3, M5 from existing code; one `item1` call with sectioned prompt; `StrategistQualityConfig` with `lean` preset; no new fields | All existing strategist tests pass unchanged in assertions under `lean`; output identical on fixtures; agent file ≤ ~150 lines |
| **2. Trust** | Quote-backed prose for M1; Auditor per-year ROIC; M4 + orchestrator hook; quality defaults (model, output tokens, full input, evidence limits) + repair pass | Unsupported prose removed in tests; "Wide" capped when ROIC below hurdle; `not_assessed` without auditor; repaired claims pass the same checks |
| **3. Drivers** | M2 (`mdna` call), Forecaster link | KPIs / drivers verified against chunks; Forecaster receives segment context |
| **4. Coverage** | M6; RPO / deferred revenue in M3; prior-year trajectory; M5 suppliers / seasonality / backlog; skeptic pass; optional agreement runs and LLM relevance check | Each new field has verified evidence or a "not disclosed" value; trajectory rule tested; verified counter-evidence lowers durability |

Each phase is a separate PR.

## 5. Error handling

- A module failure (exception, missing section, no tables) → that module `not_assessed` with a note;
  other modules and the deterministic tables are still returned.
- A call-group failure or malformed JSON → its modules `not_assessed` (as today's `llm_status`).
- M4 without auditor output → `not_assessed`, never a silent pass.
- The agent as a whole never raises on data problems (unchanged contract).

## 6. Testing

- **Phase 1 safety net:** existing `tests/test_agent_business_strategist.py` keeps all assertions.
  Its patch targets (`locate_section_chunks`, `load_filing_chunks`, `retrieve_10k_narrative`,
  `_run_judgement_llm`) move to the new locations; patch paths are the only test edits.
- **Per-module unit tests** (`tests/business_strategist/test_m*.py`): `verify()` driven by a canned
  LLM slice and a small evidence pool — no DB, no LLM.
- **Call-group test:** sections concatenate in a fixed order; the JSON answer splits back to modules.
- **Integration:** `test_full_pipeline.py` still passes; Phase 2+ add cases for the M4 hook and the
  Forecaster link.

## 7. Open questions

1. **ROIC hurdle for M4:** fixed value (e.g. 10%) or the Valuation Specialist's WACC? WACC is computed
   after the join, so a fixed hurdle is simpler; WACC is more correct.
2. **M6 commentary:** keep code-only templates, or add a small LLM comment later?
3. **Report rendering:** should the new fields get their own report sections, or extend Pillar 1?
4. **Judgement model:** which stronger model to use for the call groups, and whether agreement runs (3×)
   are on by default or opt-in.
