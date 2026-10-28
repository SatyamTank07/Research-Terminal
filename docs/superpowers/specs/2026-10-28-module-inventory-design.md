# Module Inventory — Design

Date: 2026-10-28 · Branch: `claude/festive-hawking-kowg5d` · Status: complete, all 7 agents approved

Companion to `2026-10-28-codebase-top-down-structure-design.md`. That spec defines the shape of every agent
(levels, steps, module and agent anatomy, pipeline). This document lists **which modules each agent has**.

For each module: the analyst question it answers, what it calculates or judges, whether it has a judge step,
and which earlier modules of the same agent it reads.

Legend: **+** = new coverage (not in today's code); everything else exists today and moves.
"Code only" = no `judge.j2` / `verify.py`.

Order of agents: Auditor → Strategist → Risk → Forecaster → Valuation → Synthesizer → Supervisor.

---

## 1. Financial Auditor

Source inputs: `2026-10-20-financial-auditor-restructure-design.md` (areas), `financial_math_tools.py`,
`statement_extraction.py`, `agent_financial_auditor.py`.

### Agent level

| Piece | Content |
|---|---|
| `context/` | Folder (the parser is too large for one file): fetch statement tables (target + up to 2 prior filings), extract line items, transcribe fallback (LLM reads the tables only when parsing fails; output flags it) |
| judge batch | 1 call covering every module with a judge |
| `assemble.py` | Rank all findings by severity, markdown statement tables, citations, `data_quality` |
| write | 1 call: the 5-paragraph `auditor_summary`, written over final numbers, then figure check |

LLM calls per run: 2, plus 1 only when parsing fails.

### Modules

| # | Module | Analyst question | Calculates (finding codes) | Judge | Reads |
|---|---|---|---|---|---|
| 1 | `statement_integrity` | Can I trust the extracted numbers? | Identity checks, completeness, cross-filing restatements, field provenance | Yes: proposes input corrections (each must cite a table row) and restatement notes | — |
| 2 | `financial_history` | What did the last 3–5 years look like? | Revenue, EBIT, NI, OCF, capex, FCF, growth, FCF conversion | Code only | — |
| 3 | `margins_and_costs` | How profitable is each dollar of revenue, and is that changing? | Gross / operating / net margins, bps change, opex vs revenue growth (`MARGIN_COMPRESSION`, `NEGATIVE_OPERATING_LEVERAGE`) · + capitalised-cost and useful-life policy excerpts | Yes: verdict | 2 |
| 4 | `working_capital` | Is cash getting tied up in operations? | DSO, DIO, DPO, cash conversion cycle (`INVENTORY_ACCUMULATION`, `WORKING_CAPITAL_DETERIORATION`) | Yes: verdict | 2 |
| 5 | `revenue_quality` | Is reported revenue real and collectable? | Receivables vs revenue (`RECEIVABLES_DIVERGENCE`) · + deferred revenue / RPO trend, revenue-recognition policy excerpt | Yes: verdict | 2, 4 |
| 6 | `cash_conversion` | Are profits backed by cash? | Accruals, Sloan, OCF/NI, SBC, SBC-adjusted FCF (`HIGH_ACCRUALS`, `CHRONIC_ACCRUAL_DEFICIT`, `EARNINGS_QUALITY_DIVERGENCE`, `WEAK_CASH_CONVERSION`, `SBC_INTENSITY`) | Yes: verdict | 2 |
| 7 | `earnings_normalization` | What are earnings without one-offs? | Normalised EBIT / NI, non-operating share (`NON_CORE_INCOME_DEPENDENCE`) · + restructuring / impairment footnotes | Yes: proposes one-off adjustments (cited, discrete), screened by today's rules | 2 |
| 8 | `tax_quality` | Is the tax rate sustainable? | Effective rate, deferred tax / NI (`TAX_ANOMALY`) · + rate reconciliation, valuation allowance, cash taxes | Yes: verdict | 2 |
| 9 | `balance_sheet_strength` | Can it pay its debts? | Liquid cash, debt, net debt, current ratio, D/E, net debt / EBITDA (`NEGATIVE_BOOK_EQUITY`) · + leases, pension, maturity ladder, goodwill share of assets | Yes: verdict | 2 |
| 10 | `capital_allocation` | Does management earn more than the capital costs, and where does the cash go? | ROIC, ROE, capex / revenue, capex / D&A, FCF bridge, dilution (`CAPEX_DRIVEN_FCF`, `UNDERINVESTMENT`, `SHARE_DILUTION`) · + buybacks, dividends, M&A spend, R&D intensity, goodwill impairments | Yes: verdict | 2, 8, 9 |
| 11 | `controls_and_governance` + | Did the auditor or management flag problems? | Opinion type, critical audit matters, material weakness, going concern, auditor change, accounting changes (finding only on an explicit phrase match in code) | Yes: verdict | — |
| 12 | `screening_scores` + | Do textbook distress and manipulation scores flag it? | Beneish M, Piotroski F, Altman Z'' (book form), Sloan | Code only | 2, 4, 6, 9 |

An accepted input correction (module 1) re-runs calculate for every module; an accepted one-off adjustment
(module 7) re-runs calculate from module 7 down.

### Changes vs the 2026-10-20 auditor spec

1. `balance_sheet` split into `working_capital` and `balance_sheet_strength`: two analyst questions, and the
   Forecaster reads working-capital intensity on its own.
2. `financial_history` added as a code-only base module (eight modules read it).
3. `screening_scores` has no judge: fixed formulas with textbook cut-offs.
4. No `forensic_flags` module: each finding comes from the module that owns its metric; `assemble` ranks them.
5. `capital_allocation` absorbs the Strategist's former M6 (buybacks, dividends, M&A, R&D, impairments).
6. `capital_allocation` outputs ROIC **per audited year** (today: latest year only); the Synthesizer's
   `moat_vs_returns_check` needs the series.

---

## 2. Business Strategist

Source inputs: `2026-10-18-business-strategist-modules-design.md` (branch `claude/focused-curie-3kjlay`),
`agent_business_strategist.py`, `business_disclosure_tools.py`, `claim_relevance.py`.

### Agent level

| Piece | Content |
|---|---|
| `context.py` | Locate Item 1 (RAG fallback), load filing, infer 10-K items · + locate MD&A, prior-year Item 1 |
| judge batch | 2 concurrent calls grouped by filing input: Item 1 call (modules 2–4), MD&A call (module 5) |
| `claim_relevance.py` | Agent-level: mechanism-relevance rules used by the verify steps of modules 3 and 4 |
| `assemble.py` | `BusinessMoatOutput`, citations, `data_quality` |
| write | None: prose is quote-verified judge output |

LLM calls per run: 2 (today: 1).

### Modules

| # | Module | Analyst question | Calculates / judges | Judge | Reads |
|---|---|---|---|---|---|
| 1 | `revenue_mix` | Where does revenue come from, in numbers? | Segment revenue, profit, growth, share; revenue lines; reconciliation to consolidated revenue; business vs geographic basis | Code only | — |
| 2 | `business_model` | What does it sell, to whom, and how does it make money? | Summary, revenue model, segment descriptions and growth drivers · + customers and channels | Yes (Item 1 call). Verify: names match module 1; + every prose field cites a verified quote, else deterministic text | 1 |
| 3 | `competition_and_moat` | Who does it compete with, and what protects it? | Primary, secondary and segment moats; durability; threats · + trajectory vs prior-year Item 1 | Yes (Item 1 call). Verify: verbatim quotes, mechanism relevance, "Wide" needs 2 quotes; + trajectory needs a quote showing the change | 1 |
| 4 | `market_power` | Can it set prices, and who does it depend on? | Pricing power, customer concentration · + supplier dependence, seasonality, backlog | Yes (Item 1 call). Verify: today's rules, code-screened sentence fallback | — |
| 5 | `segment_drivers` + | What is growing, why, and which KPIs matter? | Per-segment change drivers, company KPIs, price vs volume | Yes (MD&A call). Verify: quotes; every figure appears in the quoted chunk; names match module 1 | 1 |

### Changes vs the 2026-10-18 strategist spec

1. Old M1 split into `revenue_mix` (code, ~700 lines of table parsing) and `business_model` (judge).
2. Old M4 "Moat Numbers Check" moves to the **Synthesizer**: it compares two agents' outputs. The hurdle is the
   WACC when the valuation ran, else a fixed rate.
3. Old M6 "Capital Allocation" merges into the Auditor's `capital_allocation` (statement numbers belong to one
   owner). The executive-officers list is dropped (usually in the proxy, out of scope).
4. Contracted revenue (RPO / deferred revenue) is dropped here; the Auditor's `revenue_quality` owns it.
5. Quality settings from that spec (agreement runs, skeptic pass, repair pass, stronger model, larger budgets)
   are agent-level judge policy, decided in the migration plan, not modules.

---

## 3. Risk Analyst

Source inputs: `agent_risk_analyst.py`, `risk_disclosure_tools.py`, the three risk prompts. No earlier spec.

Today the agent still runs a LangGraph tool loop: the LLM may call `retrieve_10k_narrative_tool` for a legal
follow-up (Item 3) or as the Item 1A fallback. In this design both become code gather.

### Agent level

| Piece | Content |
|---|---|
| `context.py` | Locate Item 1A (label → heading boundary → RAG fallback); "not provided" detection (smaller reporting companies); prior-year Item 1A; section budget |
| judge batch | 1 call (module 5) |
| repair pass | Agent-level judge policy (shared with the Strategist): 1 extra call only when a material quantified disclosure stays uncited; repaired risks pass the same verify |
| `assemble.py` | Risk matrix, evidence table, citations, `data_quality` |
| write | None: risk summaries are quote-verified judge output |

LLM calls per run: 1, plus 1 repair call when needed. The tool loop is removed.

### Modules

| # | Module | Analyst question | Calculates / judges | Judge | Reads |
|---|---|---|---|---|---|
| 1 | `disclosure_changes` | What did management add, expand or drop vs last year? | Year-over-year Item 1A diff: new and removed passages, sentence similarity index | Code only | — |
| 2 | `quantified_exposures` | Which risks does the company put a number on? | Sentences stating a % or currency figure, each with a materiality flag | Code only | — |
| 3 | `legal_proceedings` | What lawsuits and investigations are pending, and for how much? | Item 3 passages and the legal-contingencies note, with stated amounts (replaces today's LLM tool call) | Code only | — |
| 4 | `market_risk_sensitivity` + | How much do rates, FX and commodities move earnings? | Item 7A stated sensitivities (e.g. "a 100 bp rise would reduce interest income by $X") | Code only | — |
| 5 | `risk_inventory` | What are the 5–8 material risks, how likely, how damaging, and how do they reach cash flow? | LLM proposes category, title, summary, likelihood, impact, financial transmission, quantified exposure, mitigation, signposts, quotes | Yes. Verify: verbatim quotes; impact caps (immaterial or legal-outside-1A → Medium, boilerplate → Low, unsubstantiated High → Medium); exposure figures appear in the quote; mitigation needs a quote; duplicate merge; change label from module 1; coverage check against module 2 (gaps trigger the repair pass) | 1, 2, 3, 4 |
| 6 | `risk_profile` | Which risk matters most, and how risky is the company overall? | Severity from the likelihood × impact matrix, ranking, 5–8 cap, primary threat, overall profile | Code only (runs in the calculate re-run after verify) | 5 |

### Notes

1. The four code modules run before the judge so the LLM receives every fact up front instead of searching.
2. `risk_profile` is a module, not part of `assemble`: its rules (severity matrix, "primary threat must be among
   the most severe", profile from severities) are finance rules with thresholds.
3. `market_risk_sensitivity` is the only new coverage: Item 7A is a required 10-K section that nothing reads today.

---

## 4. Forecasting Analyst

Source inputs: `agent_forecasting_analyst.py`, `forecast_baseline.py`, `forecast_drivers.py`, `forecast_model.py`,
`forecast_tools.py`, `mdna_guidance.py`.

Upstream: `needs` `financial_audit`; `uses` `business_moat`, `risk_audit`. Terminal growth comes from `RunContext`.

Today the agent runs a tool loop: the LLM calls `propose_forecast_drivers_tool`, sees the screen's clamps and may
re-propose. In this design the LLM proposes once and code screens.

### Agent level

| Piece | Content |
|---|---|
| `context.py` | Locate Item 7 (MD&A) through the shared section locator; RAG fallback |
| judge batch | 1 call (module 3) |
| repair pass | Agent-level judge policy: 1 retry with the screen's rejection reasons (replaces tool-loop self-correction) |
| `assemble.py` | `ForecastOutput`, citations |
| write | 1 call: growth and margin rationale over the final screened drivers, then figure check. The reinvestment rationale stays a code template |

LLM calls per run: 2, plus 1 repair call when needed.

### Modules

| # | Module | Analyst question | Calculates / judges | Judge | Reads |
|---|---|---|---|---|---|
| 1 | `audited_baseline` | What is the audited starting point? | Normalised margin and trend, median tax rate, recency-weighted CapEx / D&A / NWC / SBC intensity, steady-state RONIC (8–30%), historical CAGR, history table | Code only | — |
| 2 | `management_guidance` | What does management say about the future? | Forward-looking and quantified Item 7 sentences (excludes safe-harbor, FX-only and single-quarter text), margin-evidence chunks, guided vs fade decision | Code only | — |
| 3 | `revenue_and_margin_drivers` | How fast will revenue grow, and where will margins go? | Calculate: historical fade (default and fallback), growth anchor, margin corridor. Judge: 5-year growth and margin path, cited CapEx / NWC overrides, a reason per driver; context includes the Strategist's segment drivers and the top revenue / margin risks | Yes. Verify: year-1 band, ±3 pp margin steps, evidence-gated corridor around the audited trend, overrides cited and within bounds, decimal-reading check; rejected drivers fall back to the fade | 1, 2 |
| 4 | `reinvestment_path` | How much must it reinvest to fund that growth? | CapEx fade to the steady state where net reinvestment = g / RONIC (maintenance fade when NOPAT ≤ 0), D&A, NWC on incremental revenue, SBC memo line, 10-year extension when year-5 growth > 3% | Code only | 1, 3 |
| 5 | `free_cash_flow_schedule` | What cash does the business throw off each year? | UFCF = NOPAT + D&A − CapEx − ΔNWC, revenue CAGR, cumulative FCF, schedule table with audited history | Code only | 1, 3, 4 |
| 6 | `scenarios` | What if things go better or worse? | Bull / base / bear growth shifts (2–10 pp, scaled ×1.5 / 1.0 / 0.75 by the risk profile) and margin shifts (+1 / −1.5 pp), 25 / 50 / 25 weights, schedule re-run per scenario | Code only | 3, 4, 5 |
| 7 | `forecast_consistency` | Does the forecast hang together? | Growth vs reinvestment support gap, implied ROIC and RONIC, CapEx-fade warning | Code only | 1, 4, 5 |

### Notes

1. Tool loop removed; the repair pass gives the LLM one correction with explicit reasons.
2. Write step added: today the LLM's rationale can describe a number the screen later clamped.
3. MD&A gather reads the located Item 7 section instead of a 6-chunk keyword search; RAG stays as fallback.
4. Reinvestment and free cash flow are separate modules, matching the separate build steps of a model.
5. Public door for the Valuation Specialist: `project_with_overrides` (module 5), `SCENARIO_PROBABILITIES`
   (module 6), `MARGIN_FLOOR` (module 1).

---

## 5. Valuation Specialist

Source inputs: `agent_valuation_specialist.py`, `valuation_model.py` (`build_valuation`), `wacc_tools.py`,
`dcf_tools.py`, `market_data_tools.py`, the valuation node in `orchestrator.py`.

Upstream: `needs` `financial_audit` **and** `forecast` (changed: see note 4). Overrides (WACC, beta, price),
terminal growth and valuation date come from `RunContext`.

### Agent level

| Piece | Content |
|---|---|
| `context.py` | Market data (price, market cap, raw beta, live 10Y Treasury, each with a source tag; config defaults offline), balance-sheet date, user overrides applied with source tags (all moved from today's orchestrator node) |
| judge batch | None: every number is code |
| `assemble.py` | `DCFValuationOutput`; collects each module's data-quality warnings |
| write | 1 call: commentary, template fallback, then figure check |

LLM calls per run: 1.

### Modules

| # | Module | Analyst question | Calculates | Judge | Reads |
|---|---|---|---|---|---|
| 1 | `cost_of_capital` | What return do investors require? | CAPM: Rf, implied ERP, Blume-adjusted beta (default 1.0), cost of equity; cost of debt = interest / average debt inside the Rf−1pp..15% band, else Rf + spread; marginal-rate tax shield; market-value weights (book-equity fallback); a user override replaces the rate and keeps the components visible | Code only | — |
| 2 | `intrinsic_value` | What is the business worth today? | Mid-year 2-stage DCF; value-driver terminal value NOPAT × (1 − g/RONIC) at the marginal tax rate; EV → equity bridge (net debt, NCI); roll-forward to today (EV at WACC, claims at Rf); fair value per share; TV-concentration warning; implied terminal EV/EBITDA (5–30x) and EV/NOPAT (6–35x) bands | Code only | 1 |
| 3 | `scenario_values` | What is it worth if the bull or bear case plays out? | A DCF per forecast scenario with the same WACC and bridge; probability-weighted value | Code only | 1, 2 |
| 4 | `market_implied_expectations` | What is the current price already assuming? | Reverse DCF: implied growth, implied margin shift, implied discount rate; gap warning against the forecast | Code only | 1, 2 |
| 5 | `sensitivity` | How much does the value move if WACC, g or RONIC is off? | WACC × g grid, WACC × RONIC grid | Code only | 1, 2 |
| 6 | `valuation_stance` | Undervalued, fairly valued or overvalued? | Upside against price; ±10% band → stance | Code only | 2, 3 |

### Notes

1. `build_valuation` (one 220-line function with numbered comment sections) splits into these six questions.
2. The sensitivity grids leave `calculate_dcf_with_sensitivity`: value and fragility are separate questions.
3. `valuation_stance` is its own module: the ±10% band is a finance threshold the Synthesizer's dashboard reads.
4. **Behaviour change:** the forecast becomes a `need` and the naive fallback (last audited FCF grown 5% a year)
   is removed. Without a forecast the valuation is skipped with a note, instead of publishing a fair value and a
   stance built on a placeholder growth rate.

---

## 6. Lead Synthesizer

Source inputs: `agent_lead_synthesizer.py`, `report_citations.py`, `report_layout.j2`, `report_blocks.j2` and the
context templates; the Strategist spec's M4 (moved here).

Upstream: `uses` all five outputs; `needs_any` `business_moat`, `financial_audit`, `risk_audit`; reads the run notes.

### Agent level

| Piece | Content |
|---|---|
| `context.py` | None: reads upstream outputs only |
| judge batch | None: the stance comes from the Valuation Specialist; the LLM only writes |
| write | 1 call: the route's prose fields (3-pillar thesis, executive summary, bottom line, section commentaries). Screen: drop sentences with untraceable figures, date or year-over-year claims, and layout leakage; code fallback text for empty fields |
| `report_citations.py` | Agent level: maps every table row and prose figure to its source, on top of `shared/citations.py` |
| `assemble.py` | Renders the report and builds `Final10KResearchReport` |
| `report/` | One layout template per route (`full_10k_report.j2`, `dcf_valuation_only.j2`, …) matching `ROUTES` 1:1, replacing the per-route if-blocks of `report_layout.j2`; `report_blocks.j2` keeps the shared table blocks |

LLM calls per run: 1.

### Modules

| # | Module | Analyst question | Calculates | Judge | Reads |
|---|---|---|---|---|---|
| 1 | `report_coverage` | What can this report cover, and what is missing? | Effective layout (full layout when a single-topic route lost its agent), sections present vs skipped, run notes grouped by kind (skipped / failed / parameter), year-substitution note | Code only | — |
| 2 | `moat_vs_returns_check` + | Do the numbers back the claimed moat? | "Wide" requires ROIC above the hurdle in every audited year and no `MARGIN_COMPRESSION` finding, else it is flagged as capped to "Narrow"; a verified moat with ROIC below the hurdle in every year is flagged `numbers_contradict`; hurdle = WACC when the valuation ran, else a fixed 10%; no audit → `not_assessed` | Code only | — |

### Notes

1. The moat check flags and never edits the Strategist's output (an agent never edits another agent's output).
   The report shows both views, e.g. "Moat: Wide (Strategist) · numbers support Narrow (ROIC 7–9% vs 9.1% WACC)".
2. Needs ROIC per audited year from the Auditor's `capital_allocation` (Auditor note 6).
3. The Synthesizer is the home for future cross-agent checks (e.g. a forecast that expands margins while the
   Auditor flagged `MARGIN_COMPRESSION`). None are added now.

---

## 7. Supervisor

Source inputs: `agent_supervisor.py`, `state_supervisor.py`, `prompt_supervisor.j2`, `agents/query_parameters.py`.

The Supervisor runs before the route table. It turns a chat message into the run's facts: `RunContext`, route,
`parameter` notes and the session update (top-level spec §4.3).

Today it keeps its own `AGENT_EXECUTION_PLANS` dict, a second route table next to the orchestrator's routing.
In this design `ROUTES` is the only one.

### Agent level

| Piece | Content |
|---|---|
| `context.py` | Session state (minus `pending_action`), recent messages, catalog hint (up to 25 ingested companies) |
| judge batch | 1 call: intent, ticker and year extraction; for a conversational turn the same call writes the reply (splitting it would double the cost of every chat message) |
| `assemble.py` | `RunContext`, route, `parameter` notes, session update. `RoutingPlan.active_agents` is dropped (`ROUTES` answers it) |
| write | None (the conversational reply comes from the judge call) |

LLM calls per run: 1.

### Modules

| # | Module | Question | Calculates / judges | Judge | Reads |
|---|---|---|---|---|---|
| 1 | `query_intent` | What is the user asking for? | Report type, mentioned ticker and year, conversational reply | Yes. Verify: the type is a `ROUTES` key or `conversational`, else conversational | — |
| 2 | `company_resolution` | Which company? | Explicit argument → extracted ticker → session ticker; none → clarification reply | Code only | 1 |
| 3 | `filing_resolution` | Which 10-K? | Requested year → session year (a follow-up naming neither company nor year keeps it) → latest audited; catalog lookup: document id, available years, year substitution | Code only | 2 |
| 4 | `valuation_parameters` | Did the user set WACC, g, beta or price? | Range-checked parsing (WACC 3–25%, g −2–5%, beta 0.05–4), "use CAPM / reset", per-ticker carry-over (dropped on a company switch); terminal growth resolved here as the user value or the house default | Code only | 2 |
| 5 | `session_continuity` | What should the desk remember for the next question? | Active ticker, company, fiscal year, available years, last query type, valuation overrides | Code only | 1–4 |

### Notes

1. Every turn, research or conversational, goes through `session_continuity`. Today two hand-written builders
   (`route` and `_conversational_plan`) must both be updated for a new session key (the AGENTS.md §4.2 trap).
2. Conversational turns run the same modules: 2 and 3 return `not_assessed`, 5 still runs, so overrides given in
   chat are remembered the same way.
3. `query_parameters.py` splits: parsing → `valuation_parameters`, per-ticker session binding → `session_continuity`.
4. `tools/catalog_tools.py` is confirmed unused: the Supervisor does its own catalog lookup.

---

## 8. Summary

| Agent | Modules | With judge | New (+) | LLM calls per run |
|---|---|---|---|---|
| Financial Auditor | 12 | 10 | 2 | 2 (+1 parsing fallback) |
| Business Strategist | 5 | 4 | 1 | 2 |
| Risk Analyst | 6 | 1 | 1 | 1 (+1 repair) |
| Forecasting Analyst | 7 | 1 | 0 | 2 (+1 repair) |
| Valuation Specialist | 6 | 0 | 0 | 1 |
| Lead Synthesizer | 2 | 0 | 1 | 1 |
| Supervisor | 5 | 1 | 0 | 1 |
| **Total** | **43** | **17** | **5** | |

Cross-agent moves decided here:

| What | From | To |
|---|---|---|
| Moat numbers check | Strategist spec M4 | Synthesizer `moat_vs_returns_check` |
| Capital allocation (buybacks, dividends, M&A, R&D, impairments) | Strategist spec M6 | Auditor `capital_allocation` |
| Contracted revenue (RPO, deferred revenue) | Strategist spec M3 | Auditor `revenue_quality` |
| Market data, balance-sheet date | Orchestrator valuation node | Valuation `context.py` |
| Terminal growth default | Orchestrator `_terminal_growth` | Supervisor `valuation_parameters` |

Behaviour changes decided here (all others are pure restructure or new coverage):

1. Valuation `needs` the forecast; the naive 5% FCF fallback is removed.
2. The Risk Analyst and Forecasting Analyst tool loops are removed (code gather + one judge + repair pass).
3. The Forecasting Analyst gains a write step over the final screened drivers.
4. Item 7 is read as a located section instead of a 6-chunk keyword search.

Agent-level judge policy (repair pass on/off per agent, agreement runs, skeptic pass, model choice, token budgets)
is decided in the migration plan.
