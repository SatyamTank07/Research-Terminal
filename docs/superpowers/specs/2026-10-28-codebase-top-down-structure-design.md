# Codebase Top-Down Structure — Design

Date: 2026-10-28 · Branch: `claude/festive-hawking-kowg5d` · Status: approved in conversation, Breaks 1–5

## 1. Goal

Break the backend agent code into small units that each pair a piece of code with the finance concept behind it.
Each unit must be understandable on its own, and the units must add up to an agent, then to the whole pipeline.
The structure must be easy to read for a strong engineer with thin finance background and for LLMs working on
the code, and it must hold up long term (reliability, maintainability, scalability).

This spec defines the **shape** every agent and the pipeline follow. It does not list each agent's modules
(that is the next step, §13) and it is not an implementation plan.

### Decisions (agreed)

| # | Question | Decision |
|---|---|---|
| 0 | How does the code explain finance? | Through its organisation only. No concept cards, no learning docs in the repo, no finance-teaching docstrings. Folder, file, function and type names carry the meaning |
| 1 | Rebuild or evolve? | Evolve. Keep `agents/specialized/<agent>/`; every agent folder owns all code it uses |
| 2 | Levels | L0 pipeline → L1 agent → L2 module (one analyst question) → L3 step |
| 3 | Step vocabulary | gather → calculate → judge (LLM, optional) → verify (required if judge ran); `write` is agent-level |
| 4 | Module anatomy | One folder per module: `schema.py`, `gather.py`, `calculate.py`, `judge.j2`, `verify.py`, `thresholds.py` |
| 5 | Agent anatomy | Triad kept, plus `context.py`, `assemble.py`, `write.py`/`write.j2`, `modules/__init__.py` with ordered `MODULES` |
| 6 | Routes | Explicit route table; agents declare `needs` / `uses` |
| 7 | Agent contract | Each agent exports one `AgentSpec`; every `run` has the signature `run(run: RunContext, upstream: Upstream)` |
| 8 | Node | One generic `run_agent_node(spec)` for skip / run / fail / milestone |
| 9 | Orchestrator | Thin: no finance, no agent names; run state with 6 slots that imports no agent schemas |

### Non-goals

- Changing any agent's output contract (`FinancialAuditOutput`, `ForecastOutput`, … keep fields and meaning).
- Changing the HTTP / SSE contract (`POST /chat`, event types and shapes). The frontend is untouched.
- Changing ingestion, the database schema or the services layer (`chat_service`, `langfuse_service`).
- Deciding each agent's module list (§13) or the migration order (§13). The two older per-agent specs
  (`2026-10-18-business-strategist-modules-design.md` on `claude/focused-curie-3kjlay`,
  `2026-10-20-financial-auditor-restructure-design.md`) are inputs to that step, translated into this vocabulary (§10).

## 2. Current state (problems this design removes)

| Problem | Where |
|---|---|
| `tools/` is a shared drawer of 20 files; ownership is invisible and cross-agent coupling is silent | `agents/tools/` |
| A few cross-agent internal imports | `valuation_model` → forecaster's `project_with_overrides`, `SCENARIO_PROBABILITIES`, `MARGIN_FLOOR`; `forecast_baseline` → auditor's `STATUTORY_TAX_RATE`; strategist and `business_disclosure_tools` → generic helpers in `risk_disclosure_tools` |
| Duplicated logic | Quote verification in Strategist `_verify` and Risk `_verify_evidence`; three near-copy LLM JSON helpers (`_run_judgement_llm`, `_run_repair_llm`, `_run_commentary_llm`) |
| Orchestrator does finance work | `valuation_specialist_node` fetches yfinance, builds `MarketInputs`, queries the balance-sheet date; `_terminal_growth` applies the house default for g |
| Bespoke agent entry points | `analyze`, `audit`, `forecast(… 7 args)`, `value(… 10 args)`, `synthesize(… 14 args)` |
| Dependencies hidden in node bodies | Only the code says the forecaster *needs* the audit but merely *uses* moat and risk |
| Routes spread over four if/else predicates | `route_from_supervisor`, `route_from_business_strategist`, `route_from_financial_auditor`, `route_from_risk_analyst` |
| Failure handling copied six times | try / except / `None` / hand-typed note in every node |
| Per-agent status text in the orchestrator | `_node_status`, `_risk_status_message`, `_forecast_status_message`, `_valuation_status_message` |
| Central import knot | `state.py` imports every agent's schemas; `orchestrator.py` is 662 lines with five jobs |
| Dead registrations | All 7 specialists are in `AgentRegistry`, but the frontend never sends `agent_type`, and a specialist cannot run from bare `messages` |

## 3. The four levels

```
L0 Pipeline   agents/pipeline/                   which agents run, in what order, what they hand over
L1 Agent      agents/specialized/<agent>/        one analyst role; owns all its code
L2 Module       modules/<analyst_question>/      one question the analyst answers
L3 Step           gather → calculate → judge → verify
```

### Ownership rule

- Code used by **one agent** lives inside that agent's folder.
- Code used by **two or more agents** that makes **no finance judgement** lives in `agents/shared/`.
- Code that makes a finance judgement is owned by exactly one agent, even if another agent needs its result.
- Agents talk only through their outputs, downstream only. An agent's `__init__.py` is its only door; it
  exports its `AGENT` spec, its output schema types, and an explicit, short list of pure functions or
  constants that downstream agents need (§9).

## 4. L0 — Pipeline (Break 5)

### 4.1 Folder

```
agents/
  pipeline/
    contract.py      AgentSpec, RunContext, Upstream, RunNote
    routes.py        ROUTES
    node.py          run_agent_node(spec)
    graph.py         build_graph(ROUTES)   — the only LangGraph-aware code
    run_state.py     ResearchRunState
    orchestrator.py  MultiAgentOrchestrator: start a run, stream events, shape the result
  shared/            §8
  specialized/<agent>/
```

`agents/orchestrator.py` and `agents/state.py` are replaced by this folder.

### 4.2 Route table

```python
ROUTES = {
    "full_10k_report":      [[business_strategist, financial_auditor, risk_analyst],
                             forecasting_analyst, valuation_specialist, lead_synthesizer],
    "dcf_valuation_only":   [financial_auditor, forecasting_analyst, valuation_specialist, lead_synthesizer],
    "financial_audit_only": [financial_auditor, lead_synthesizer],
    "business_moat_only":   [business_strategist, lead_synthesizer],
    "risk_factors_only":    [risk_analyst, lead_synthesizer],
}
```

- Steps run in order; a nested list is a parallel group that joins before the next step.
- `conversational` is not in the table. The Supervisor answers it and the graph ends.
- `graph.py` builds the LangGraph edges from this table; no hand-written edges or route predicates remain.
- A test asserts that for every route, each agent's `needs` are produced earlier in that route.

### 4.3 Agent contract

```python
@dataclass(frozen=True)
class AgentSpec:
    name: str                       # "valuation_specialist"
    output_key: str                 # "dcf_valuation" — its slot in outputs
    needs: frozenset[str]           # required upstream output keys
    uses: frozenset[str]            # optional upstream output keys
    needs_any: frozenset[str]       # at least one must be present (empty = no rule)
    run: Callable[[RunContext, Upstream], BaseModel]
    milestone: Callable[[Optional[BaseModel]], StatusLine]   # StatusLine = (message, optional details dict)
```

Example declarations:

| Agent | `output_key` | `needs` | `uses` | `needs_any` |
|---|---|---|---|---|
| business_strategist | `business_moat` | — | — | — |
| financial_auditor | `financial_audit` | — | — | — |
| risk_analyst | `risk_audit` | — | — | — |
| forecasting_analyst | `forecast` | `financial_audit` | `business_moat`, `risk_audit` | — |
| valuation_specialist | `dcf_valuation` | `financial_audit` | `forecast` | — |
| lead_synthesizer | `final_report` | — | all five above | `business_moat`, `financial_audit`, `risk_audit` |

- `run` keeps a finance-verb name (`audit_financial_statements`, `forecast_cash_flows`, `value_equity`).
- **`RunContext`** (immutable, identical for every agent): ticker, company name, fiscal year, available fiscal
  years, document id, user query, year-substituted flag, valuation overrides, terminal growth rate, valuation
  date, callbacks.
- **`Upstream`** exposes only the declared outputs. A `needs` output is guaranteed present; a `uses` output may be
  `None`; reading an undeclared key raises. The Synthesizer's `Upstream` also carries the run's notes.
- Outside data an agent needs is fetched by that agent's `context.py` (yfinance and the balance-sheet date move
  into `valuation_specialist/context.py`).
- **Supervisor** is outside the route table. It runs first and produces the `RunContext`, the route, the session
  update and `parameter` notes. It resolves terminal growth once (user value or house default) so the forecast's
  CapEx fade and the valuation's terminal value use the same g. The default constant and the range checks live in
  `supervisor/query_parameters.py` (moved from `agents/query_parameters.py`).

### 4.4 Generic node

`run_agent_node(spec)` runs every agent in the route table with the same four steps:

1. **Check:** if a `needs` output is `None`, or `needs_any` is set and all are `None` → output `None`, write a
   `skipped` note.
2. **Run:** `await asyncio.to_thread(spec.run, run_ctx, upstream)`.
3. **Fail:** on exception → output `None`, write a `failed` note, log with traceback.
4. **Milestone:** `spec.milestone(output)` → SSE `status` event (also for `None`, so the UI shows the gap).

Two failure levels:

| Level | Trigger | Result |
|---|---|---|
| Module | A module's step fails inside an agent | Module marked `not_assessed` in the agent's output; the agent succeeds |
| Agent | The agent raises, or a declared requirement is missing | Output `None` + note; dependants that `need` it skip, those that `use` it continue |

```python
class RunNote(BaseModel):
    agent: str
    kind: Literal["skipped", "failed", "parameter"]
    message: str
```

The wrapper writes `skipped` / `failed`; the Supervisor writes `parameter` (e.g. "WACC override 9% applied").
The Synthesizer discloses notes in the report. The run fails only when the Synthesizer's `needs_any` is not met.

### 4.5 Run state

| Slot | Written by | Holds | Reducer |
|---|---|---|---|
| `conversation` | caller | user query, recent messages, incoming session state | — |
| `run` | Supervisor | `RunContext`, route, session update | — |
| `outputs` | each agent (via node) | `{output_key: output or None}` | dict merge (parallel-safe) |
| `notes` | node, Supervisor | `list[RunNote]` | append (parallel-safe) |
| `final` | Synthesizer or Supervisor | report, or conversational reply | — |
| `error` | orchestrator | message when nothing was produced | — |

The Synthesizer's output (`final_report`) goes to `final`, not `outputs`, because nothing reads it downstream.
`run_state.py` imports no agent schema; typed access is through `Upstream`. Tests import schemas from each
agent's `__init__` instead of `agents/state.py`.

### 4.6 Orchestrator

It does four things only:

1. Build the initial state, in one place. Today it is duplicated in `arun` and `astream_run`.
2. Run the graph.
3. Turn each finished node into an SSE `status` event via that agent's `milestone`. The orchestrator has no
   per-agent code.
4. Shape the final `result` or `error` event, in exactly today's wire shapes.

`AgentRegistry` keeps only the orchestrator (`multi_agent`, `orchestrator`). Specialists are `AgentSpec`s, not
registry entries.

## 5. L1 — Agent anatomy (Break 4)

```
specialized/<agent>/
  __init__.py                 the only door: AGENT, output schemas, listed public functions
  agent_<name>.py             the checklist: runs context → modules → judge batch → verify → assemble → write
  state_<name>.py             the agent's output schema
  prompt_<name>.j2            role and global rules shared by its judge calls
  context.py                  loads filings / outside data once for all modules
  assemble.py                 merges module results into the output schema
  write.py, write.j2          the agent-level prose step
  modules/
    __init__.py               MODULES = [ordered list]
    <analyst_question>/       §6
```

Run order: `context` → every module runs gather and calculate → the agent batches all judge sections into 1..N
LLM calls → each module's verify → re-run calculate once for modules whose judge proposed accepted numbers →
`assemble` → `write` → figure check.

Rules:

- **Modules never call the LLM.** The agent owns every LLM call, which keeps call count and cost under control.
- **A module reads only modules listed before it** in `MODULES`. There are no cycles.
- **An agent never edits another agent's output.** A cross-check between two agents' results belongs to a
  downstream agent. Example: the Strategist spec's check of moat claims against financial numbers moves downstream.

## 6. L2 — Module anatomy (Break 3)

```
modules/<analyst_question>/
  schema.py       the module's result types
  gather.py       step: fetch inputs (code)
  calculate.py    step: compute every number (code)
  judge.j2        step: LLM judgement prompt section   — only if the module judges
  verify.py       step: check the judgement             — required if judge.j2 exists
  thresholds.py   named constants for every rule cut-off
```

A code-only module has no `judge.j2` or `verify.py`. Files stay under about 200 lines; a bigger file means the
module answers more than one question and should split.

## 7. L3 — Step vocabulary (Break 2)

| Step | Who | Does |
|---|---|---|
| gather | code | Fetch the inputs: statement lines, filing passages, upstream outputs |
| calculate | code | Produce every number the module reports |
| judge | LLM, optional | Judge the numbers and passages; may propose a number with a cited reason |
| verify | code, required after judge | Check quotes exist (claimed chunk first, then any chunk), check numbers match calculate, apply bands |
| write | LLM, agent-level | Turn verified results into prose; a figure check drops sentences with untraceable numbers |

Invariants:

1. Every number in an output comes from `calculate`.
2. The LLM appears only in `judge` and `write`. The one exception is the auditor's transcribe fallback (LLM reads a
   statement table when parsing fails); the output flags it.
3. Nothing from `judge` reaches the output without passing `verify`.
4. `write` changes no number and no verdict.
5. A failing module becomes `not_assessed`; it never fails the agent.

## 8. Shared code

```
agents/shared/
  filing/        hybrid narrative search, table search, section locating, sentence splitting
  quotes.py      verify a quote: claimed chunk first, then any chunk of the filing
  figures.py     figure index and figure check for write steps
  citations.py   CitationRegistry, SourceRef (also used by chat_service)
  llm_json.py    one structured-LLM-call helper replacing the three near-copies
  prompts.py     render_prompt, resolving templates by agent and module path
```

`claim_relevance` stays in the Strategist because it makes a finance judgement (does this quote name the claimed
moat mechanism?).

## 9. Ownership map (where today's files go)

| Today | Destination |
|---|---|
| `tools/business_disclosure_tools.py` | `business_strategist/` |
| `tools/claim_relevance.py` | `business_strategist/` |
| `tools/financial_math_tools.py` | `financial_auditor/` (door exports `STATUTORY_TAX_RATE` for the forecaster) |
| `tools/statement_extraction.py` | `financial_auditor/` |
| `tools/risk_disclosure_tools.py` | generic section / sentence helpers → `shared/filing/`; quote verification → `shared/quotes.py`; Item 1A location and YoY diff → `risk_analyst/` |
| `tools/forecast_baseline.py`, `forecast_drivers.py`, `forecast_model.py`, `forecast_tools.py`, `mdna_guidance.py` | `forecasting_analyst/` (door exports `project_with_overrides`, `SCENARIO_PROBABILITIES`, `MARGIN_FLOOR` for the reverse DCF) |
| `tools/dcf_tools.py`, `wacc_tools.py`, `valuation_model.py` | `valuation_specialist/` |
| `tools/market_data_tools.py` | `valuation_specialist/context.py` |
| `tools/rag_narrative_tools.py`, `rag_table_tools.py` | `shared/filing/` |
| `tools/citation_registry.py` | `shared/citations.py` |
| `tools/figure_verification.py` | `shared/figures.py` |
| `specialized/prompts.py` | `shared/prompts.py` |
| `agents/query_parameters.py` | `supervisor/` |
| `agents/orchestrator.py`, `agents/state.py` | `agents/pipeline/` |
| `tools/catalog_tools.py`, `tools/tavily_search.py` | imported by nothing except `tools/__init__.py`; removal confirmed in the plan |

Within each agent, the moved files are further split into modules during that agent's migration (§13).

## 10. Naming rules

- Module folders are named after the analyst's question (`earnings_quality`, `cost_of_capital`).
  `tools`, `utils`, `helpers` and `engine` are not allowed as names.
- Functions are a finance verb plus an object (`compute_accruals_ratio`, `value_equity`).
- Units go in the name: `_rate` = decimal, `_pct` = percent, `_x` = multiple, `_bps`, `_days`.
- Thresholds are named constants in the module's `thresholds.py` (e.g. `ACCRUALS_RED_FLAG_RATE`).
- Types are finance objects prefixed by their module.
- Acronyms come from an allow-list kept in `finance_agent_backend/CLAUDE.md`, with no definitions.

Translation of the older per-agent specs into this vocabulary:

| Older spec term | This design |
|---|---|
| Auditor "audit area", Strategist "module" | module |
| Auditor collect + extract | agent-level `context` + module `gather` |
| Auditor analyze | `calculate` |
| Auditor assemble guards | `verify` |
| Auditor / Strategist "output block" | module result (`schema.py`), merged by `assemble` |
| `audit_engine/` package | dissolved into the auditor's modules |

## 11. Testing

- Tests mirror the structure: `tests/<agent>/<module>/test_<step>.py`, plus `tests/pipeline/`.
- Test names state the finance rule (`test_accruals_above_10pct_flags_red`); the test suite is the rulebook.
- Pipeline tests:
  - Every route satisfies `needs` ordering.
  - The `needs` cascade skips dependants and keeps `uses` dependants running.
  - `needs_any` failure ends the run with an `error` event.
  - SSE event shapes are unchanged (golden test against today's shapes).
- The suite stays hermetic: no live OpenAI, yfinance or Tavily calls.

## 12. Consequences for the implementation plan

- `render_prompt` today uses one global template namespace; it must resolve by agent and module path.
- The eager imports in `specialized/__init__.py` must be ordered so the pipeline can import every `AGENT` without
  cycles.
- `StructuredAgent`'s tool loop is not used by tool-free modules; its fate is decided per agent in the plan.
- AGENTS.md (§2.3, §2.4) and backend CLAUDE.md (layout, conventions, "adding an agent") must be updated with each
  migration step.

## 13. Next steps

1. **Module inventory:** list the modules for each of the 7 agents. Start from the two older specs, translated
   with §10.
2. **Migration plan:** order of work, one agent or pipeline layer at a time, each step mergeable on its own with
   output contracts and SSE shapes unchanged.
