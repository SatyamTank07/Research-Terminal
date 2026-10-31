# Codebase Restructure — Migration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the backend from shared `tools/` files and a hand-wired orchestrator to the pipeline → agent → module → step structure, without changing any output, then apply the agreed behaviour changes, then add the new finance coverage.

**Architecture:** Three phases, 27 PRs. Phase 1 restructures only; a golden snapshot of every route must stay byte-identical. Phase 2 makes one behaviour change per PR and regenerates the golden snapshot in its own commit. Phase 3 adds one new module or coverage item per PR. This document holds the roadmap for all 27 PRs and full task detail for **PR 1 (golden safety net)** and **PR 2 (pipeline L0)**. Every later PR gets its own plan file, written with superpowers:writing-plans right before it starts, because each move shifts file paths and line numbers.

**Tech Stack:** Python 3.11, FastAPI, LangGraph 1.2, LangChain / langchain-openai, Pydantic v2, SQLAlchemy + PostgreSQL 16 (pgvector), pytest (unittest-style classes), Docker (`finance_agent`, `finance_postgres`).

**Spec:** `docs/superpowers/specs/2026-10-28-codebase-top-down-structure-design.md` and `docs/superpowers/specs/2026-10-28-module-inventory-design.md`. Read both before any task.

## Global Constraints

- Phase 1 changes no agent output contract (`FinancialAuditOutput`, `ForecastOutput`, `DCFValuationOutput`, `BusinessMoatOutput`, `RiskAuditOutput`, `Final10KResearchReport`, `RoutingPlan`): same fields, same meaning, same values.
- The HTTP / SSE contract is unchanged: `POST /chat`, event types `status` / `token` / `result` / `error` and their keys. SSE `node` names stay `supervisor`, `business_strategist`, `financial_auditor`, `risk_analyst`, `forecasting_analyst`, `valuation_specialist`, `lead_synthesizer`.
- No backward-compatibility shims: when code moves, every importer and every test is updated in the same commit (user preference).
- Tests are hermetic: no live OpenAI, yfinance or Tavily calls. `tests/conftest.py` blocks ports 80/443.
- No live LLM run without the user's explicit go-ahead (very limited AI budget). Validate with mocked tests and offline golden runs.
- Never hardcode tickers or years in app code. Tickers in tests and golden scenarios are fine.
- LangGraph nodes are `async`; synchronous agent / LLM / DB / yfinance work runs through `asyncio.to_thread`.
- New code follows the spec's naming rules (§10): folders named after the analyst question, finance verb + object for functions, units in names (`_rate` decimal, `_pct` percent, `_x`, `_bps`, `_days`), no `utils` / `helpers` / `engine`.
- No file-length cap: a file may run to 2000, 5000 or more lines as long as everything in it belongs to one thing. Split a file only when it mixes more than one concern.
- All commands run inside Docker: `docker exec finance_agent pytest tests/ -q` (omit `-it` when non-interactive). `pytest` is installed ad hoc in the container and is not in `requirements.txt`; do not rebuild the image unless `requirements.txt` changes.
- Phase 1 gate for every PR: full suite green **and** `docker exec finance_agent pytest tests/golden -q` green with no snapshot change.
- Phase 2 / 3 gate: full suite green; snapshots regenerated in a separate commit `test(golden): regenerate for <change>` whose diff the PR description explains.
- Each PR updates `AGENTS.md` and `finance_agent_backend/CLAUDE.md` where layout or conventions changed, plus the two specs if a decision changed.
- `finance_agent_backend/CLAUDE.md` is untracked in git today. Edit it, but `git add` it only after the user decides to track it; until then, leave it out of every commit below.

## Review Focus

1. **Parallel Phase-1 agents writing at once** (full report): all three outputs must survive the join; a reducer that overwrites instead of merging silently drops two sections. Pinned by the parallel-graph test in Task 9 and the `full_*` golden scenarios.
2. **Turns with no research run** (conversational answer, "which company?" clarification): the graph must end after the Supervisor without touching any agent, and override notes given in chat must still stream. Pinned by the conversational-graph test in Task 9 and the `chat_what_is_dcf` golden scenario.
3. **A valuation override remembered from an earlier turn**: the forecaster's CapEx fade and the valuation's terminal value must use the same terminal growth. Pinned by the shared-`RunContext` test in Task 7 and the `followup_dcf_aapl_session` golden scenario.
4. **Every research agent failing**: the run must end with an `error` event carrying today's message, never a silent empty result. Pinned by the needs_any test in Task 4 and the orchestrator error test in Task 10.
5. **Non-deterministic snapshots** (DB row order, float formatting, wall clock): a flaky golden is worse than none. Pinned by the run-twice step in Task 2 and the pinned clock in Task 1.

---

## Roadmap

### Phase 1: pure restructure (golden snapshots unchanged)

| PR | Scope | Depends on |
|---|---|---|
| 1 | **Golden safety net**: offline fakes, 10 scenarios, committed snapshots. No app change. (Tasks 1–2 below) | — |
| 2 | **Pipeline L0**: `agents/pipeline/` (contract, run state, node, routes, graph, orchestrator); every agent exports `AGENT`; valuation `context.py` takes market data and the balance-sheet date from the orchestrator; terminal growth and valuation date move into `RunContext`; `agents/orchestrator.py` and `agents/state.py` deleted. (Tasks 3–10 below) | 1 |
| 3 | **Shared code**: `agents/shared/` with `filing/` (rag_narrative, rag_table, section locate, sentence split), `quotes.py`, `figures.py`, `citations.py`, `llm_json.py`, `prompts.py` (templates resolved by agent and module path) | 2 |
| 4 | **Valuation Specialist → 6 modules** (`cost_of_capital`, `intrinsic_value`, `scenario_values`, `market_implied_expectations`, `sensitivity`, `valuation_stance`); `dcf_tools`, `wacc_tools`, `valuation_model`, `market_data_tools` move in | 3 |
| 5 | **Forecasting Analyst → 7 modules**; `forecast_*` and `mdna_guidance` move in; default-`terminal_growth` parameters removed (g always comes from `RunContext`); the tool loop stays at agent level | 3 |
| 6 | **Financial Auditor → `context/` + 10 existing modules**; `financial_math_tools` and `statement_extraction` move in; the tool loop stays | 3 |
| 7 | **Business Strategist → 4 existing modules** (`revenue_mix`, `business_model`, `competition_and_moat`, `market_power`); `business_disclosure_tools` and `claim_relevance` move in; the single judgement call stays | 3 |
| 8 | **Risk Analyst → 4 existing modules** (`disclosure_changes`, `quantified_exposures`, `risk_inventory`, `risk_profile`); the tool loop stays | 3 |
| 9 | **Lead Synthesizer**: `report_coverage` module, one layout per route under `report/`, `report_citations.py` kept at agent level | 3 |
| 10 | **Supervisor → 5 modules**; one `session_continuity` path for research and chat turns; `query_parameters.py` split; `AGENT_EXECUTION_PLANS` removed | 2 |
| 11 | **Cleanup**: delete what is left of `agents/tools/` (incl. unused `catalog_tools.py`, `tavily_search.py`); `AgentRegistry` keeps only the orchestrator; prune unused `StructuredAgent` code; final docs pass | 4–10 |

### Phase 2: behaviour changes (one per PR, snapshots regenerated)

| PR | Change | Why |
|---|---|---|
| 12 | **Fix: chat turns lose the requested and the session fiscal year.** The orchestrator turns a missing `fiscal_year` into `0`, so `SupervisorAgent.route` never uses the LLM-extracted year or the session-year carry-over (AGENTS.md §4.2). PR 1's `followup_dcf_aapl_session` scenario pins today's wrong answer (FY2025 instead of the session's FY2024) | Bug found while writing this plan |
| 13 | **Valuation `needs` the forecast**; the naive 5% FCF fallback is removed | Inventory §5 |
| 14 | **Forecaster reads the located Item 7 section** (shared locator, RAG fallback) instead of a 6-chunk keyword search | Inventory §4 |
| 15 | **Forecaster: tool loop → judge batch + repair pass + write step** | Inventory §4 |
| 16 | **Risk Analyst: tool loop → code gather (`legal_proceedings` module) + judge batch + repair pass** | Inventory §3 |
| 17 | **Financial Auditor: tool loop → judge batch + write step** (summary written over final numbers) | Inventory §1; missing from the inventory's behaviour-change summary, corrected in the same commit as this plan |

Agent-level judge policy (repair pass on/off, model choice, token budgets) is decided in each PR's own plan; the default stays gpt-4o-mini because of the budget.

### Phase 3: new coverage (one per PR, snapshots regenerated)

| PR | Adds |
|---|---|
| 18 | Auditor `capital_allocation` outputs ROIC per audited year; Synthesizer `moat_vs_returns_check` |
| 19 | Auditor `controls_and_governance` |
| 20 | Auditor `screening_scores` (Beneish, Piotroski, Altman Z'', Sloan) |
| 21 | Risk `market_risk_sensitivity` (Item 7A) |
| 22 | Strategist `segment_drivers` (MD&A judge call) and the forecaster's judge context link |
| 23 | Auditor footnotes: `balance_sheet_strength` (leases, pension, maturity ladder, goodwill) |
| 24 | Auditor footnotes: `tax_quality`, `earnings_normalization` |
| 25 | Auditor: `revenue_quality` (deferred revenue / RPO, revenue-recognition excerpt), `margins_and_costs` (policy excerpts) |
| 26 | Auditor: `capital_allocation` cash uses (buybacks, dividends, M&A, R&D intensity, impairments) |
| 27 | Strategist: customers and channels, quote-backed prose, trajectory vs prior year, suppliers / seasonality / backlog |

Each Phase 3 PR also renders its new fields in the report (or says why not) and ends with at most one live validation run, only with the user's go-ahead.

### Known quirks Phase 1 preserves on purpose

- `fiscal_year=0` sentinel from the orchestrator to the Supervisor (fixed in PR 12).
- A Synthesizer exception ends the run with `error` "Research pipeline failed: …"; it is not recorded as a note (kept via `AgentSpec.writes_final`).
- Specialists stay registered in `AgentRegistry` until PR 11 (tests assert their registration).
- `forecast_drivers` keeps a default `terminal_growth` parameter, imported from `query_parameters`, until PR 5.

---

# PR 1 — Golden safety net

Branch: `refactor/golden-safety-net` from `main`.

### Task 1: Offline fakes for every external boundary

**Files:**
- Create: `finance_agent_backend/tests/golden/__init__.py` (empty)
- Create: `finance_agent_backend/tests/golden/offline.py`
- Test: `finance_agent_backend/tests/golden/test_offline.py`

**Interfaces:**
- Consumes: `langchain_openai.ChatOpenAI`, `langchain_openai.OpenAIEmbeddings`, `yfinance.Ticker` (library classes, patched in place).
- Produces: `offline(structured_responses: Mapping[str, dict]) -> ContextManager[None]`, `GOLDEN_DATE: date`, `FakeDate`, `fake_vector(text: str) -> list[float]`, `CLOCK_TARGETS: tuple[str, ...]`, `MARKET: dict`.

- [ ] **Step 1: Write the failing test**

`finance_agent_backend/tests/golden/test_offline.py`:

```python
"""The offline fakes must answer exactly like the services they replace, with fixed values."""

import unittest

import yfinance
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

from tests.golden.offline import EMBEDDING_DIM, GOLDEN_DATE, FakeDate, offline


class _Answer(BaseModel):
    text: str


def _chat() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", api_key="sk-offline")


class TestOfflineFakes(unittest.TestCase):
    def test_chat_model_replies_with_empty_json(self):
        with offline({}):
            reply = _chat().invoke("hello")
        self.assertEqual(reply.content, "{}")

    def test_structured_output_returns_the_canned_answer_for_its_schema(self):
        with offline({"_Answer": {"text": "canned"}}):
            answer = _chat().with_structured_output(_Answer).invoke("question")
        self.assertEqual(answer, _Answer(text="canned"))

    def test_structured_output_without_a_canned_answer_raises(self):
        with offline({}):
            with self.assertRaises(RuntimeError):
                _chat().with_structured_output(_Answer).invoke("question")

    def test_embeddings_are_deterministic_and_non_zero(self):
        with offline({}):
            client = OpenAIEmbeddings(model="text-embedding-3-small", api_key="sk-offline")
            first, again = client.embed_query("revenue"), client.embed_query("revenue")
            other = client.embed_documents(["margin"])[0]
        self.assertEqual(first, again)
        self.assertEqual(len(first), EMBEDDING_DIM)
        self.assertNotEqual(first, other)
        self.assertTrue(any(value != 0.0 for value in first))

    def test_quotes_are_fixed_per_ticker(self):
        with offline({}):
            apple = yfinance.Ticker("aapl")
            treasury = yfinance.Ticker("^TNX").history(period="5d")
            unknown = yfinance.Ticker("ZZZZ")
        self.assertEqual(apple.fast_info.last_price, 230.0)
        self.assertEqual(apple.fast_info.market_cap, 3.45e12)
        self.assertEqual(apple.info["beta"], 1.2)
        self.assertEqual(treasury["Close"].iloc[-1], 4.3)
        self.assertIsNone(unknown.fast_info)
        self.assertEqual(unknown.info, {})

    def test_clock_is_pinned(self):
        self.assertEqual(FakeDate.today(), GOLDEN_DATE)

    def test_patches_are_removed_on_exit(self):
        with offline({}):
            pass
        self.assertNotEqual(yfinance.Ticker.__name__, "FakeTicker")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec finance_agent pytest tests/golden/test_offline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.golden.offline'`

- [ ] **Step 3: Write the implementation**

`finance_agent_backend/tests/golden/__init__.py`: empty file.

`finance_agent_backend/tests/golden/offline.py`:

```python
"""Offline fakes for golden snapshot runs.

Every boundary that leaves the process is replaced: chat models, embeddings, market quotes and the clock
that dates the valuation. The fakes patch library classes (langchain_openai, yfinance) rather than app
modules, so snapshots keep working while app code moves between files. The one exception is
CLOCK_TARGETS: update it when the module that stamps the valuation date moves.
"""

import contextlib
import hashlib
import random
from datetime import date
from typing import Any, Dict, Iterator, List, Mapping, Optional
from unittest.mock import patch

import pandas as pd
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

GOLDEN_DATE = date(2026, 10, 28)
EMBEDDING_DIM = 1536
TREASURY_10Y_PCT = 4.3
CHAT_REPLY = "{}"
MARKET: Dict[str, Dict[str, float]] = {
    "AAPL": {"last_price": 230.0, "market_cap": 3.45e12, "beta": 1.2},
    "AMZN": {"last_price": 220.0, "market_cap": 2.3e12, "beta": 1.3},
    "NVDA": {"last_price": 180.0, "market_cap": 4.4e12, "beta": 1.7},
    "TSLA": {"last_price": 250.0, "market_cap": 8.0e11, "beta": 2.0},
}
# Modules whose `date.today()` stamps the valuation date; each is patched to GOLDEN_DATE.
CLOCK_TARGETS = ("app.agents.orchestrator.date", "app.agents.tools.valuation_model.date")


class FakeDate(date):
    """`date` with a pinned today(), so the DCF roll-forward does not change from day to day."""

    @classmethod
    def today(cls) -> date:
        return GOLDEN_DATE


def _fake_generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=CHAT_REPLY))])


async def _fake_agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
    return _fake_generate(self, messages, stop, run_manager, **kwargs)


class FakeStructuredModel:
    """Stands in for `ChatOpenAI.with_structured_output(schema)`: returns the canned answer for that schema."""

    def __init__(self, schema: Any, responses: Mapping[str, Dict[str, Any]]):
        self.schema = schema
        self.responses = responses

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        canned = self.responses.get(self.schema.__name__)
        if canned is None:
            raise RuntimeError(f"No canned structured response for {self.schema.__name__}")
        return self.schema.model_validate(canned)


def fake_vector(text: str) -> List[float]:
    """Deterministic, non-zero pseudo-embedding (a zero vector makes cosine distance undefined)."""
    rng = random.Random(hashlib.sha256(text.encode("utf-8")).digest())
    return [rng.uniform(-1.0, 1.0) for _ in range(EMBEDDING_DIM)]


class _FastInfo:
    def __init__(self, quote: Dict[str, float]):
        self.last_price = quote["last_price"]
        self.market_cap = quote["market_cap"]


class FakeTicker:
    """yfinance.Ticker stand-in: fixed price, market cap and beta per ticker, fixed 10Y Treasury yield."""

    def __init__(self, symbol: str):
        self.symbol = symbol.strip().upper()

    @property
    def fast_info(self) -> Optional[_FastInfo]:
        quote = MARKET.get(self.symbol)
        return _FastInfo(quote) if quote else None

    @property
    def info(self) -> Dict[str, float]:
        quote = MARKET.get(self.symbol)
        return {"beta": quote["beta"]} if quote else {}

    def history(self, period: str = "5d") -> pd.DataFrame:
        if self.symbol == "^TNX":
            return pd.DataFrame({"Close": [TREASURY_10Y_PCT]})
        return pd.DataFrame({"Close": []})


@contextlib.contextmanager
def offline(structured_responses: Mapping[str, Dict[str, Any]]) -> Iterator[None]:
    """Runs the block with every external boundary faked; patches are removed on exit."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(ChatOpenAI, "_generate", _fake_generate))
        stack.enter_context(patch.object(ChatOpenAI, "_agenerate", _fake_agenerate))
        stack.enter_context(patch.object(
            ChatOpenAI, "with_structured_output",
            lambda self, schema, **kwargs: FakeStructuredModel(schema, structured_responses),
        ))
        stack.enter_context(patch.object(OpenAIEmbeddings, "embed_query", lambda self, text: fake_vector(text)))
        stack.enter_context(patch.object(
            OpenAIEmbeddings, "embed_documents",
            lambda self, texts, chunk_size=None: [fake_vector(text) for text in texts],
        ))
        stack.enter_context(patch("yfinance.Ticker", FakeTicker))
        for target in CLOCK_TARGETS:
            stack.enter_context(patch(target, FakeDate))
        yield
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec finance_agent pytest tests/golden/test_offline.py -v`
Expected: 7 passed.

If `test_chat_model_replies_with_empty_json` fails because the installed langchain-openai routes `invoke` through `_stream`, also patch `ChatOpenAI._stream` to yield one `ChatGenerationChunk(message=AIMessageChunk(content=CHAT_REPLY))`, and rerun.

- [ ] **Step 5: Commit**

```bash
git add finance_agent_backend/tests/golden/__init__.py finance_agent_backend/tests/golden/offline.py finance_agent_backend/tests/golden/test_offline.py
git commit -m "test(golden): offline fakes for chat, embeddings, market quotes and clock"
```

### Task 2: Golden scenarios, harness and committed snapshots

**Files:**
- Create: `finance_agent_backend/tests/golden/scenarios.py`
- Create: `finance_agent_backend/tests/golden/harness.py`
- Create: `finance_agent_backend/tests/golden/test_golden_pipeline.py`
- Create (generated): `finance_agent_backend/tests/golden/snapshots/*.json` (10 files)
- Modify: `finance_agent_backend/CLAUDE.md` (Testing rules section)

**Interfaces:**
- Consumes: `offline()` (Task 1); `MultiAgentOrchestrator.arun(user_query, ticker=None, fiscal_year=None, messages=None, session_state=None, callbacks=None)` and `.astream_run(...)` (same keywords) from `app.agents`.
- Produces: `SCENARIOS: dict[str, Scenario]`; `run_scenario(Scenario) -> dict`; `extract_outputs(state) -> dict` (the one adapter PR 2 updates); snapshot JSON shape `{"scenario", "outputs": {business_moat, financial_audit, risk_audit, forecast, dcf_valuation, final_report, conversational_response, updated_session_state, pipeline_notes}, "events": [...]}`.

- [ ] **Step 1: Write the scenarios**

`finance_agent_backend/tests/golden/scenarios.py`:

```python
"""Golden scenarios: one chat turn each, with the canned structured LLM answers that turn needs.

The Supervisor's routing and the Synthesizer's prose are the only LLM answers with content; every other
LLM call gets "{}", so agents take their deterministic paths (the code being moved).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

# One sentence carries a figure found in no filing, so the figure screen must drop it.
SYNTHESIS_PROSE: Dict[str, str] = {
    "executive_summary": "The filing supports a balanced view of the business. Revenue reached $999,999 million last year.",
    "bottom_line": "The stance depends on whether margins hold.",
    "pillar_1_business_moat": "The business relies on its installed base and its brand.",
    "pillar_2_financial_durability": "Cash generation covers reinvestment needs.",
    "pillar_3_valuation_asymmetry": "The valuation case rests on the base-case cash flows.",
    "business_commentary": "Segment disclosure frames the revenue mix.",
    "earnings_quality_commentary": "Reported and normalized earnings are compared year by year.",
    "cash_flow_balance_sheet_commentary": "Free cash flow funds buybacks and dividends.",
    "forecast_commentary": "Growth fades toward the terminal rate.",
    "valuation_commentary": "The discount rate drives most of the value range.",
    "risk_commentary": "Risk disclosures focus on competition and regulation.",
    "bear_case_commentary": "A demand slowdown is the main bear case.",
    "disclosure_change_commentary": "Year-over-year risk changes are limited.",
}


@dataclass(frozen=True)
class Scenario:
    name: str
    query: str
    extraction: Dict[str, Any]
    session_state: Optional[Dict[str, Any]] = None

    def structured_responses(self) -> Dict[str, Dict[str, Any]]:
        return {"SupervisorExtraction": self.extraction, "ResearchSynthesisPayload": SYNTHESIS_PROSE}


def _research(query_type: str, ticker: Optional[str], year: Optional[int]) -> Dict[str, Any]:
    return {
        "query_type": query_type,
        "extracted_ticker": ticker,
        "extracted_year": year,
        "routing_reasoning": "golden scenario",
    }


# Pins today's behaviour: the session year (FY2024) is lost because the orchestrator passes fiscal_year=0,
# so the run uses FY2025. PR 12 fixes this and regenerates the snapshot.
_AAPL_SESSION = {
    "active_ticker": "AAPL",
    "active_company": "Apple Inc.",
    "active_fiscal_year": 2024,
    "available_fiscal_years": [2025, 2024],
    "last_query_type": "full_10k_report",
    "valuation_overrides": {"ticker": "AAPL", "terminal_growth": 0.03},
}

SCENARIOS: Dict[str, Scenario] = {
    scenario.name: scenario
    for scenario in [
        Scenario("full_aapl_2025", "Write a full 10-K research report on Apple for FY2025",
                 _research("full_10k_report", "AAPL", 2025)),
        Scenario("full_amzn_2025", "Write a full 10-K research report on Amazon for FY2025",
                 _research("full_10k_report", "AMZN", 2025)),
        Scenario("full_nvda_2026", "Write a full 10-K research report on NVIDIA for FY2026",
                 _research("full_10k_report", "NVDA", 2026)),
        Scenario("full_tsla_2025", "Write a full 10-K research report on Tesla for FY2025",
                 _research("full_10k_report", "TSLA", 2025)),
        Scenario("dcf_aapl_2025_overrides", "Run a DCF on Apple with a WACC of 9% and terminal growth of 3%",
                 _research("dcf_valuation_only", "AAPL", 2025)),
        Scenario("audit_aapl_2025", "Audit Apple's FY2025 financial statements",
                 _research("financial_audit_only", "AAPL", 2025)),
        Scenario("moat_nvda_2026", "What is NVIDIA's business model and economic moat?",
                 _research("business_moat_only", "NVDA", 2026)),
        Scenario("risk_tsla_2025", "What are Tesla's main risk factors?",
                 _research("risk_factors_only", "TSLA", 2025)),
        Scenario("followup_dcf_aapl_session", "Now run the DCF again with a beta of 1.4",
                 _research("dcf_valuation_only", None, None), session_state=_AAPL_SESSION),
        Scenario("chat_what_is_dcf", "What is a discounted cash flow valuation?", {
            "query_type": "conversational",
            "extracted_ticker": None,
            "extracted_year": None,
            "conversational_response": "A DCF values a business as the present value of its future free cash flows.",
            "routing_reasoning": "golden scenario",
        }),
    ]
}
```

- [ ] **Step 2: Write the harness**

`finance_agent_backend/tests/golden/harness.py`:

```python
"""Runs a golden scenario through the orchestrator offline and turns the result into a JSON snapshot."""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from app.agents import MultiAgentOrchestrator
from tests.golden.offline import offline
from tests.golden.scenarios import Scenario

SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"
UPDATE_ENV = "UPDATE_GOLDEN"
OUTPUT_KEYS = ("business_moat", "financial_audit", "risk_audit", "forecast", "dcf_valuation")


def _dump(value: Any) -> Any:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


def extract_outputs(state: Dict[str, Any]) -> Dict[str, Any]:
    """Agent outputs and turn results from the final run state.

    This is the only place that knows the state's shape; update it when the shape changes, never the snapshots.
    """
    return {
        **{key: _dump(state.get(key)) for key in OUTPUT_KEYS},
        "final_report": _dump(state.get("final_report")),
        "conversational_response": state.get("conversational_response"),
        "updated_session_state": state.get("updated_session_state"),
        "pipeline_notes": list(state.get("pipeline_notes") or []),
    }


def _messages(scenario: Scenario) -> List[Dict[str, str]]:
    return [{"role": "user", "content": scenario.query}]


async def _collect(scenario: Scenario) -> Dict[str, Any]:
    orchestrator = MultiAgentOrchestrator()
    turn = dict(user_query=scenario.query, messages=_messages(scenario), session_state=scenario.session_state)
    state = await orchestrator.arun(**turn)
    # The result event's final_report duplicates outputs.final_report, so it is left out.
    events = [
        {key: value for key, value in event.items() if key != "final_report"}
        async for event in orchestrator.astream_run(**turn)
    ]
    return {"scenario": scenario.name, "outputs": extract_outputs(state), "events": events}


def run_scenario(scenario: Scenario) -> Dict[str, Any]:
    with offline(scenario.structured_responses()):
        snapshot = asyncio.run(_collect(scenario))
    return json.loads(json.dumps(snapshot, sort_keys=True, default=str))  # plain JSON types only


def snapshot_path(name: str) -> Path:
    return SNAPSHOT_DIR / f"{name}.json"


def write_snapshot(name: str, snapshot: Dict[str, Any]) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    text = json.dumps(snapshot, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    snapshot_path(name).write_text(text, encoding="utf-8")


def read_snapshot(name: str) -> Dict[str, Any]:
    return json.loads(snapshot_path(name).read_text(encoding="utf-8"))


def first_difference(expected: Any, actual: Any, path: str = "$") -> Optional[str]:
    """Path and values of the first mismatch between two JSON values, or None when they are equal."""
    if type(expected) is not type(actual):
        return f"{path}: type {type(expected).__name__} != {type(actual).__name__} ({expected!r:.200} vs {actual!r:.200})"
    if isinstance(expected, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                return f"{path}.{key}: unexpected key (value {actual[key]!r:.200})"
            if key not in actual:
                return f"{path}.{key}: missing key (expected {expected[key]!r:.200})"
            diff = first_difference(expected[key], actual[key], f"{path}.{key}")
            if diff:
                return diff
        return None
    if isinstance(expected, list):
        for index, (want, got) in enumerate(zip(expected, actual)):
            diff = first_difference(want, got, f"{path}[{index}]")
            if diff:
                return diff
        if len(expected) != len(actual):
            return f"{path}: length {len(expected)} != {len(actual)}"
        return None
    if expected != actual:
        return f"{path}: {expected!r:.300} != {actual!r:.300}"
    return None
```

- [ ] **Step 3: Write the golden test, including content sanity checks on the snapshots**

`finance_agent_backend/tests/golden/test_golden_pipeline.py`:

```python
"""Golden snapshots: the whole pipeline, run offline, must reproduce the committed JSON exactly.

Regenerate only when an output change is intended, in its own commit:
    docker exec -e UPDATE_GOLDEN=1 finance_agent pytest tests/golden -q
Regenerate on main after re-ingesting any filing, before starting a PR.
"""

import os
import unittest

from tests.golden.harness import (
    OUTPUT_KEYS,
    UPDATE_ENV,
    first_difference,
    read_snapshot,
    run_scenario,
    snapshot_path,
    write_snapshot,
)
from tests.golden.scenarios import SCENARIOS


class TestGoldenPipeline(unittest.TestCase):
    def _check(self, name: str) -> None:
        actual = run_scenario(SCENARIOS[name])
        if os.environ.get(UPDATE_ENV) == "1":
            write_snapshot(name, actual)
            return
        self.assertTrue(snapshot_path(name).exists(), f"No snapshot for {name}; run with {UPDATE_ENV}=1 first")
        diff = first_difference(read_snapshot(name), actual)
        self.assertIsNone(diff, f"Golden mismatch in {name}: {diff}")

    def test_full_aapl_2025(self):
        self._check("full_aapl_2025")

    def test_full_amzn_2025(self):
        self._check("full_amzn_2025")

    def test_full_nvda_2026(self):
        self._check("full_nvda_2026")

    def test_full_tsla_2025(self):
        self._check("full_tsla_2025")

    def test_dcf_aapl_2025_overrides(self):
        self._check("dcf_aapl_2025_overrides")

    def test_audit_aapl_2025(self):
        self._check("audit_aapl_2025")

    def test_moat_nvda_2026(self):
        self._check("moat_nvda_2026")

    def test_risk_tsla_2025(self):
        self._check("risk_tsla_2025")

    def test_followup_dcf_aapl_session(self):
        self._check("followup_dcf_aapl_session")

    def test_chat_what_is_dcf(self):
        self._check("chat_what_is_dcf")

    def test_every_scenario_has_a_test(self):
        tested = {name[len("test_"):] for name in dir(self) if name.startswith("test_")}
        self.assertEqual(set(SCENARIOS) - tested, set())


@unittest.skipIf(os.environ.get(UPDATE_ENV) == "1", "snapshots are being regenerated")
class TestGoldenSnapshotContent(unittest.TestCase):
    """Guards against snapshots that are byte-stable but meaningless (e.g. a broken fake nulls every output)."""

    def test_full_report_has_every_section(self):
        outputs = read_snapshot("full_aapl_2025")["outputs"]
        for key in OUTPUT_KEYS + ("final_report",):
            self.assertIsNotNone(outputs[key], key)

    def test_research_turns_start_with_the_supervisor_and_end_with_a_result(self):
        for name in ("full_aapl_2025", "audit_aapl_2025", "risk_tsla_2025"):
            events = read_snapshot(name)["events"]
            self.assertEqual((events[0]["type"], events[0]["node"]), ("status", "supervisor"), name)
            self.assertEqual(events[-1]["type"], "result", name)

    def test_dcf_route_skips_qualitative_agents_and_applies_the_wacc_override(self):
        outputs = read_snapshot("dcf_aapl_2025_overrides")["outputs"]
        self.assertIsNone(outputs["business_moat"])
        self.assertIsNone(outputs["risk_audit"])
        self.assertEqual(outputs["dcf_valuation"]["wacc_audit"]["wacc_source"], "user_override")
        self.assertAlmostEqual(outputs["dcf_valuation"]["wacc_audit"]["wacc_pct"], 9.0)

    def test_follow_up_carries_session_overrides(self):
        outputs = read_snapshot("followup_dcf_aapl_session")["outputs"]
        self.assertIn("Applied terminal_growth specified earlier in the conversation.", outputs["pipeline_notes"])
        self.assertEqual(outputs["updated_session_state"]["valuation_overrides"]["beta"], 1.4)

    def test_fabricated_figure_is_screened_out_of_the_report(self):
        report = read_snapshot("full_aapl_2025")["outputs"]["final_report"]["full_markdown_report"]
        self.assertNotIn("999,999", report)

    def test_conversational_turn_runs_no_agent(self):
        outputs = read_snapshot("chat_what_is_dcf")["outputs"]
        self.assertTrue(outputs["conversational_response"].startswith("A DCF values a business"))
        for key in OUTPUT_KEYS + ("final_report",):
            self.assertIsNone(outputs[key], key)
```

- [ ] **Step 4: Run it to verify it fails without snapshots**

Run: `docker exec finance_agent pytest tests/golden/test_golden_pipeline.py -q`
Expected: the 10 `TestGoldenPipeline` scenario tests FAIL with `No snapshot for <name>`; the `TestGoldenSnapshotContent` tests error with `FileNotFoundError`.

- [ ] **Step 5: Generate the snapshots**

Run: `docker exec -e UPDATE_GOLDEN=1 finance_agent pytest tests/golden/test_golden_pipeline.py -q`
Expected: 11 passed, 6 skipped; `tests/golden/snapshots/` holds 10 JSON files.

- [ ] **Step 6: Verify the snapshots are deterministic and meaningful**

Run twice: `docker exec finance_agent pytest tests/golden -q`
Expected both times: all passed (7 + 11 + 6). Then `git status --short finance_agent_backend/tests/golden/snapshots` shows only new, unmodified files.

If a scenario passes on the generating run but fails on the next, the pipeline is non-deterministic for that input. Find the varying path in the failure message and make the harness stable at the boundary (for example, a clock read missing from `CLOCK_TARGETS`). Never loosen the comparison.

If a content check fails (for example, `full_aapl_2025` has a `null` section), a fake is not reaching an agent. Open the snapshot, find the failure note in `outputs.pipeline_notes`, fix the fake, and regenerate.

- [ ] **Step 7: Run the full suite**

Run: `docker exec finance_agent pytest tests/ -q`
Expected: all passed (the existing suite plus the golden tests).

- [ ] **Step 8: Document the golden workflow**

In `finance_agent_backend/CLAUDE.md`, section `## Testing rules`, add after the "hermetic" bullet:

```markdown
- **Golden snapshots** (`tests/golden/`): ten chat turns run the real pipeline on the real DB with every LLM, embedding, yfinance and clock call faked (`tests/golden/offline.py`), and must reproduce `tests/golden/snapshots/*.json` exactly. A restructure must not change them. An intended output change regenerates them in its own commit: `docker exec -e UPDATE_GOLDEN=1 finance_agent pytest tests/golden -q`. Regenerate on `main` after re-ingesting a filing.
```

- [ ] **Step 9: Commit**

```bash
git add finance_agent_backend/tests/golden   # + finance_agent_backend/CLAUDE.md once it is tracked
git commit -m "test(golden): pipeline snapshots for 10 chat turns as the restructure safety net"
```

---

# PR 2 — Pipeline L0

Branch: `refactor/pipeline-l0` from `main` after PR 1 merged. Gate: full suite green, golden unchanged.

### Task 3: Pipeline contract

**Files:**
- Create: `finance_agent_backend/app/agents/pipeline/__init__.py` (docstring only, no imports)
- Create: `finance_agent_backend/app/agents/pipeline/contract.py`
- Create: `finance_agent_backend/tests/pipeline/__init__.py` (empty)
- Create: `finance_agent_backend/tests/pipeline/fixtures.py`
- Test: `finance_agent_backend/tests/pipeline/test_contract.py`

**Interfaces:**
- Produces: `StatusLine = Tuple[Optional[str], Optional[Dict[str, Any]]]`; `RunNote(agent: str, kind: Literal["skipped","failed","parameter"], message: str)`; `Conversation(user_query, messages, session_state, ticker: str, fiscal_year: int, run_date: date, callbacks=None)`; `RunContext(ticker, company_name, fiscal_year, available_fiscal_years: Tuple[int, ...], document_id, user_query, route, year_substituted, valuation_overrides: Mapping, terminal_growth_rate: float, valuation_date: date, callbacks=None)`; `Upstream(outputs, declared, notes=())` with `.get(key)` and `.notes`; `UndeclaredUpstreamError(KeyError)`; `AgentSpec(name, label, output_key, run, milestone, needs=frozenset(), uses=frozenset(), needs_any=frozenset(), skip_message="", writes_final=False)` with `.declared`. Test helper `make_run_context(**changes) -> RunContext`.

- [ ] **Step 1: Write the failing test**

`finance_agent_backend/tests/pipeline/fixtures.py`:

```python
"""Shared builders for pipeline tests."""

from datetime import date
from typing import Any

from app.agents.pipeline.contract import RunContext

RUN_DATE = date(2026, 10, 28)


def make_run_context(**changes: Any) -> RunContext:
    fields = dict(
        ticker="AAPL",
        company_name="Apple Inc.",
        fiscal_year=2025,
        available_fiscal_years=(2025, 2024),
        document_id="doc-aapl-2025",
        user_query="Write a full report on AAPL",
        route="full_10k_report",
        year_substituted=False,
        valuation_overrides={},
        terminal_growth_rate=0.025,
        valuation_date=RUN_DATE,
        callbacks=None,
    )
    fields.update(changes)
    return RunContext(**fields)
```

`finance_agent_backend/tests/pipeline/test_contract.py`:

```python
import dataclasses
import unittest

from app.agents.pipeline.contract import AgentSpec, RunNote, UndeclaredUpstreamError, Upstream
from tests.pipeline.fixtures import make_run_context


def _noop_run(run, upstream):
    return None


def _noop_milestone(output):
    return None, None


class TestUpstream(unittest.TestCase):
    def test_declared_output_is_readable_and_missing_one_is_none(self):
        upstream = Upstream({"financial_audit": "audit"}, frozenset({"financial_audit", "forecast"}))
        self.assertEqual(upstream.get("financial_audit"), "audit")
        self.assertIsNone(upstream.get("forecast"))

    def test_reading_an_undeclared_output_raises(self):
        upstream = Upstream({"financial_audit": "audit", "risk_audit": "risk"}, frozenset({"financial_audit"}))
        with self.assertRaises(UndeclaredUpstreamError):
            upstream.get("risk_audit")

    def test_notes_are_carried(self):
        note = RunNote(agent="supervisor", kind="parameter", message="WACC override 9% applied")
        self.assertEqual(Upstream({}, frozenset(), (note,)).notes, (note,))


class TestAgentSpec(unittest.TestCase):
    def test_declared_is_the_union_of_needs_uses_and_needs_any(self):
        spec = AgentSpec(
            name="lead_synthesizer", label="Lead Synthesizer", output_key="final_report",
            run=_noop_run, milestone=_noop_milestone,
            needs=frozenset({"a"}), uses=frozenset({"b"}), needs_any=frozenset({"c"}),
        )
        self.assertEqual(spec.declared, frozenset({"a", "b", "c"}))


class TestRunContext(unittest.TestCase):
    def test_run_context_is_immutable(self):
        ctx = make_run_context()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ctx.ticker = "MSFT"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec finance_agent pytest tests/pipeline/test_contract.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.agents.pipeline'`

- [ ] **Step 3: Write the implementation**

`finance_agent_backend/app/agents/pipeline/__init__.py`:

```python
"""L0 pipeline: which agents run, in what order, and what they hand each other.

Kept import-free so agents can import `app.agents.pipeline.contract` without cycles.
"""
```

`finance_agent_backend/app/agents/pipeline/contract.py`:

```python
"""Pipeline contract: what every agent declares, what it receives, and what the run records."""

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Dict, FrozenSet, List, Literal, Mapping, Optional, Tuple

from pydantic import BaseModel

StatusLine = Tuple[Optional[str], Optional[Dict[str, Any]]]


class RunNote(BaseModel):
    """A disclosure the report carries: an agent was skipped or failed, or a user parameter was applied."""

    agent: str
    kind: Literal["skipped", "failed", "parameter"]
    message: str


@dataclass(frozen=True)
class Conversation:
    """The chat turn the Supervisor resolves. `ticker` is "" and `fiscal_year` 0 when the caller named none."""

    user_query: str
    messages: Optional[List[Dict[str, str]]]
    session_state: Optional[Dict[str, Any]]
    ticker: str
    fiscal_year: int
    run_date: date
    callbacks: Optional[List[Any]] = None


@dataclass(frozen=True)
class RunContext:
    """Facts the Supervisor resolved for one research run; every agent receives the same instance."""

    ticker: str
    company_name: str
    fiscal_year: int
    available_fiscal_years: Tuple[int, ...]
    document_id: Optional[str]
    user_query: str
    route: str
    year_substituted: bool
    valuation_overrides: Mapping[str, Any]
    terminal_growth_rate: float
    valuation_date: date
    callbacks: Optional[List[Any]] = None


class UndeclaredUpstreamError(KeyError):
    """An agent read an upstream output it did not declare in needs, uses or needs_any."""


class Upstream:
    """Read-only view of the upstream outputs one agent declared, plus the run's notes so far."""

    def __init__(self, outputs: Mapping[str, Any], declared: FrozenSet[str], notes: Tuple[RunNote, ...] = ()):
        self._outputs = {key: outputs.get(key) for key in declared}
        self.notes = notes

    def get(self, key: str) -> Any:
        if key not in self._outputs:
            raise UndeclaredUpstreamError(key)
        return self._outputs[key]


@dataclass(frozen=True)
class AgentSpec:
    """An agent's door into the pipeline: what it reads, what it writes, how it runs and reports."""

    name: str
    label: str
    output_key: str
    run: Callable[[RunContext, Upstream], Any]
    milestone: Callable[[Optional[Any]], StatusLine]
    needs: FrozenSet[str] = frozenset()
    uses: FrozenSet[str] = frozenset()
    needs_any: FrozenSet[str] = frozenset()
    skip_message: str = ""
    writes_final: bool = False

    @property
    def declared(self) -> FrozenSet[str]:
        return self.needs | self.uses | self.needs_any
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec finance_agent pytest tests/pipeline/test_contract.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add finance_agent_backend/app/agents/pipeline finance_agent_backend/tests/pipeline
git commit -m "feat(pipeline): agent contract (AgentSpec, RunContext, Upstream, RunNote)"
```

### Task 4: Run state and the generic agent node

**Files:**
- Create: `finance_agent_backend/app/agents/pipeline/run_state.py`
- Create: `finance_agent_backend/app/agents/pipeline/node.py`
- Test: `finance_agent_backend/tests/pipeline/test_node.py`

**Interfaces:**
- Consumes: Task 3 contract.
- Produces: `merge_outputs(current, update) -> dict`; `ResearchRunState` TypedDict (`conversation: Conversation`, `run: Any` (the Supervisor's `TurnResolution`), `outputs: Annotated[dict, merge_outputs]`, `notes: Annotated[list[RunNote], operator.add]`, `final: Any`); `failure_note(label, exc) -> str`; `RunAborted(RuntimeError)`; `make_agent_node(spec) -> async (state) -> dict`. Node update shape: `{"outputs": {spec.output_key: output}}` or `{"final": output}` when `writes_final`, plus `"notes": [RunNote]` when there is a note.

- [ ] **Step 1: Write the failing test**

`finance_agent_backend/tests/pipeline/test_node.py`:

```python
import asyncio
import unittest
from types import SimpleNamespace

from app.agents.pipeline.contract import AgentSpec, RunNote
from app.agents.pipeline.node import RunAborted, failure_note, make_agent_node
from app.agents.pipeline.run_state import merge_outputs
from tests.pipeline.fixtures import make_run_context


def _milestone(output):
    return None, None


def _spec(run, **fields) -> AgentSpec:
    base = dict(name="forecasting_analyst", label="Forecasting Analyst", output_key="forecast",
                run=run, milestone=_milestone)
    base.update(fields)
    return AgentSpec(**base)


def _state(outputs=None, notes=None):
    return {"run": SimpleNamespace(run_context=make_run_context()), "outputs": outputs or {}, "notes": notes or []}


def _call(spec, state):
    return asyncio.run(make_agent_node(spec)(state))


class TestMergeOutputs(unittest.TestCase):
    def test_parallel_updates_merge_instead_of_overwriting(self):
        merged = merge_outputs(merge_outputs({}, {"business_moat": "m"}), {"financial_audit": "a"})
        self.assertEqual(merged, {"business_moat": "m", "financial_audit": "a"})

    def test_none_current_is_treated_as_empty(self):
        self.assertEqual(merge_outputs(None, {"risk_audit": "r"}), {"risk_audit": "r"})


class TestAgentNode(unittest.TestCase):
    def test_runs_the_agent_with_context_and_declared_upstream(self):
        seen = {}

        def run(ctx, upstream):
            seen["ticker"], seen["audit"] = ctx.ticker, upstream.get("financial_audit")
            return "forecast"

        update = _call(_spec(run, needs=frozenset({"financial_audit"})), _state({"financial_audit": "audit"}))
        self.assertEqual(update, {"outputs": {"forecast": "forecast"}})
        self.assertEqual(seen, {"ticker": "AAPL", "audit": "audit"})

    def test_missing_need_skips_with_the_agents_note(self):
        def run(ctx, upstream):
            raise AssertionError("must not run")

        spec = _spec(run, needs=frozenset({"financial_audit"}),
                     skip_message="Forecast skipped: no audited financial statements were available.")
        update = _call(spec, _state({"financial_audit": None}))
        self.assertEqual(update["outputs"], {"forecast": None})
        self.assertEqual(update["notes"], [RunNote(agent="forecasting_analyst", kind="skipped",
                                                   message="Forecast skipped: no audited financial statements were available.")])

    def test_failure_is_recorded_not_raised(self):
        def run(ctx, upstream):
            raise ValueError("base_revenue must be positive")

        update = _call(_spec(run), _state())
        self.assertEqual(update["outputs"], {"forecast": None})
        self.assertEqual(update["notes"][0].kind, "failed")
        self.assertEqual(
            update["notes"][0].message,
            "Forecasting Analyst failed (ValueError: base_revenue must be positive); its section is omitted from this report.",
        )

    def test_final_writer_puts_its_output_in_final(self):
        update = _call(_spec(lambda ctx, up: "report", writes_final=True), _state())
        self.assertEqual(update, {"final": "report"})

    def test_final_writer_failure_ends_the_run(self):
        def run(ctx, upstream):
            raise RuntimeError("template error")

        with self.assertRaises(RuntimeError):
            _call(_spec(run, writes_final=True), _state())

    def test_unmet_needs_any_aborts_with_todays_message(self):
        notes = [RunNote(agent="financial_auditor", kind="failed", message="Financial Auditor failed (X: y).")]
        spec = _spec(lambda ctx, up: "report", writes_final=True,
                     needs_any=frozenset({"business_moat", "financial_audit", "risk_audit"}))
        with self.assertRaises(RunAborted) as caught:
            _call(spec, _state({"financial_audit": None}, notes))
        self.assertEqual(str(caught.exception),
                         "No research output could be produced for AAPL FY2025. Financial Auditor failed (X: y).")


class TestFailureNote(unittest.TestCase):
    def test_long_messages_are_collapsed_and_cut(self):
        note = failure_note("Risk Analyst", ValueError("a\n  b " + "x" * 300))
        self.assertTrue(note.startswith("Risk Analyst failed (ValueError: a b xxx"))
        self.assertTrue(note.endswith("; its section is omitted from this report."))

    def test_empty_message_falls_back_to_the_exception_type(self):
        self.assertEqual(failure_note("Risk Analyst", KeyError()),
                         "Risk Analyst failed (KeyError: KeyError); its section is omitted from this report.")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec finance_agent pytest tests/pipeline/test_node.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.agents.pipeline.node'`

- [ ] **Step 3: Write the implementation**

`finance_agent_backend/app/agents/pipeline/run_state.py`:

```python
"""The state one research run carries through the graph; it imports no agent schema."""

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from app.agents.pipeline.contract import Conversation, RunNote


def merge_outputs(current: Optional[Dict[str, Any]], update: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Parallel agents each add their own key; merging keeps every one of them."""
    return {**(current or {}), **(update or {})}


class ResearchRunState(TypedDict, total=False):
    conversation: Conversation
    run: Any  # the Supervisor's TurnResolution: route, RunContext, session update
    outputs: Annotated[Dict[str, Any], merge_outputs]
    notes: Annotated[List[RunNote], operator.add]
    final: Any  # Final10KResearchReport, or the conversational reply
```

`finance_agent_backend/app/agents/pipeline/node.py`:

```python
"""The one node every routed agent runs through: check its inputs, run it, record a failure, report."""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List

from app.agents.pipeline.contract import AgentSpec, RunNote, Upstream
from app.agents.pipeline.run_state import ResearchRunState

logger = logging.getLogger("finance_agent.agents.pipeline")


class RunAborted(RuntimeError):
    """No research output exists, so no report can be written; the run ends with an error event."""


def failure_note(label: str, exc: BaseException) -> str:
    detail = " ".join(str(exc).split())[:240] or type(exc).__name__
    return f"{label} failed ({type(exc).__name__}: {detail}); its section is omitted from this report."


def _update(spec: AgentSpec, output: Any, notes: List[RunNote]) -> Dict[str, Any]:
    update: Dict[str, Any] = {"final": output} if spec.writes_final else {"outputs": {spec.output_key: output}}
    if notes:
        update["notes"] = notes
    return update


def make_agent_node(spec: AgentSpec) -> Callable[[ResearchRunState], Awaitable[Dict[str, Any]]]:
    async def agent_node(state: ResearchRunState) -> Dict[str, Any]:
        run_ctx = state["run"].run_context
        outputs = state.get("outputs") or {}
        notes = tuple(state.get("notes") or [])
        if any(outputs.get(key) is None for key in spec.needs):
            return _update(spec, None, [RunNote(agent=spec.name, kind="skipped", message=spec.skip_message)])
        if spec.needs_any and all(outputs.get(key) is None for key in spec.needs_any):
            raise RunAborted(
                f"No research output could be produced for {run_ctx.ticker} FY{run_ctx.fiscal_year}. "
                + " ".join(note.message for note in notes)
            )
        try:
            output = await asyncio.to_thread(spec.run, run_ctx, Upstream(outputs, spec.declared, notes))
        except Exception as exc:
            if spec.writes_final:
                raise
            logger.exception(f"[{spec.name}] {run_ctx.ticker} FY{run_ctx.fiscal_year} failed")
            return _update(spec, None, [RunNote(agent=spec.name, kind="failed", message=failure_note(spec.label, exc))])
        return _update(spec, output, [])

    agent_node.__name__ = f"{spec.name}_node"
    return agent_node
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec finance_agent pytest tests/pipeline/test_node.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add finance_agent_backend/app/agents/pipeline/run_state.py finance_agent_backend/app/agents/pipeline/node.py finance_agent_backend/tests/pipeline/test_node.py
git commit -m "feat(pipeline): run state and the generic agent node (skip, run, fail, abort)"
```

### Task 5: Supervisor door — resolve a turn into a RunContext

**Files:**
- Modify: `finance_agent_backend/app/agents/query_parameters.py` (add `DEFAULT_TERMINAL_GROWTH`, `resolve_terminal_growth`)
- Modify: `finance_agent_backend/app/agents/tools/forecast_drivers.py:25` (import the constant instead of defining it)
- Modify: `finance_agent_backend/app/agents/specialized/supervisor/state_supervisor.py` (add `TurnResolution`)
- Modify: `finance_agent_backend/app/agents/specialized/supervisor/agent_supervisor.py` (add `resolve_turn`, `supervisor_milestone`)
- Modify: `finance_agent_backend/app/agents/specialized/supervisor/__init__.py` (export them)
- Modify: `finance_agent_backend/app/agents/pipeline/node.py` (add `supervisor_node`)
- Test: `finance_agent_backend/tests/test_agent_supervisor.py` (new class), `finance_agent_backend/tests/pipeline/test_node.py` (new class), `finance_agent_backend/tests/test_query_parameters.py` (new class)

**Interfaces:**
- Consumes: `Conversation`, `RunContext`, `RunNote`, `StatusLine` (Task 3); `SupervisorAgent.route(user_query, ticker, fiscal_year, session_state, messages, callbacks) -> RoutingPlan` (existing).
- Produces: `DEFAULT_TERMINAL_GROWTH = 0.025`; `resolve_terminal_growth(overrides: Mapping) -> float`; `TurnResolution(route, plan, run_context: Optional[RunContext], conversational_response, session_update, parameter_notes: Tuple[str, ...])`; `resolve_turn(conversation: Conversation) -> TurnResolution`; `supervisor_milestone(turn: Optional[TurnResolution]) -> StatusLine`; `supervisor_node(state) -> {"run": TurnResolution, "notes"?: [...], "final"?: str}`.

- [ ] **Step 1: Write the failing tests**

Append to `finance_agent_backend/tests/test_query_parameters.py`:

```python
class TestTerminalGrowthResolution(unittest.TestCase):
    def test_user_value_wins(self):
        from app.agents.query_parameters import resolve_terminal_growth
        self.assertAlmostEqual(resolve_terminal_growth({"terminal_growth": 0.03}), 0.03)

    def test_house_default_when_absent(self):
        from app.agents.query_parameters import DEFAULT_TERMINAL_GROWTH, resolve_terminal_growth
        self.assertEqual(resolve_terminal_growth({}), DEFAULT_TERMINAL_GROWTH)
        self.assertEqual(DEFAULT_TERMINAL_GROWTH, 0.025)

    def test_forecaster_uses_the_same_default(self):
        from app.agents.query_parameters import DEFAULT_TERMINAL_GROWTH
        from app.agents.tools import forecast_drivers
        self.assertIs(forecast_drivers.DEFAULT_TERMINAL_GROWTH, DEFAULT_TERMINAL_GROWTH)
```

(If `unittest` is not imported at the top of that file, add `import unittest`.)

Append to `finance_agent_backend/tests/test_agent_supervisor.py`:

```python
from datetime import date as _date
from unittest.mock import patch as _patch

from app.agents.pipeline.contract import Conversation
from app.agents.specialized.supervisor import resolve_turn, supervisor_milestone


def _plan(**changes) -> RoutingPlan:
    fields = dict(
        ticker="AAPL", company_name="Apple Inc.", fiscal_year=2025, year_requested=2025, year_substituted=False,
        available_fiscal_years=[2025, 2024], document_id="doc-aapl-2025", query_type="dcf_valuation_only",
        active_agents=[], routing_provenance="llm_inferred", conversational_response=None,
        valuation_overrides={"wacc": 0.09, "terminal_growth": 0.03}, parameter_notes=["WACC override 9% applied"],
        updated_session_state={"active_ticker": "AAPL"},
    )
    fields.update(changes)
    return RoutingPlan(**fields)


def _conversation(query: str = "DCF on AAPL") -> Conversation:
    return Conversation(user_query=query, messages=None, session_state=None, ticker="", fiscal_year=0,
                        run_date=_date(2026, 10, 28))


class TestResolveTurn(unittest.TestCase):
    def test_research_turn_builds_the_run_context(self):
        with _patch.object(SupervisorAgent, "route", return_value=_plan()) as route:
            turn = resolve_turn(_conversation())
        self.assertEqual(route.call_args.kwargs["fiscal_year"], 0)  # today's sentinel, fixed in PR 12
        ctx = turn.run_context
        self.assertEqual((ctx.ticker, ctx.fiscal_year, ctx.route), ("AAPL", 2025, "dcf_valuation_only"))
        self.assertEqual(ctx.available_fiscal_years, (2025, 2024))
        self.assertAlmostEqual(ctx.terminal_growth_rate, 0.03)
        self.assertEqual(ctx.valuation_date, _date(2026, 10, 28))
        self.assertEqual(ctx.valuation_overrides, {"wacc": 0.09, "terminal_growth": 0.03})
        self.assertEqual(turn.parameter_notes, ("WACC override 9% applied",))
        self.assertEqual(turn.session_update, {"active_ticker": "AAPL"})

    def test_terminal_growth_defaults_when_not_overridden(self):
        with _patch.object(SupervisorAgent, "route", return_value=_plan(valuation_overrides={})):
            turn = resolve_turn(_conversation())
        self.assertEqual(turn.run_context.terminal_growth_rate, 0.025)

    def test_company_name_falls_back_to_ticker(self):
        with _patch.object(SupervisorAgent, "route", return_value=_plan(company_name=None)):
            turn = resolve_turn(_conversation())
        self.assertEqual(turn.run_context.company_name, "AAPL")

    def test_conversational_turn_has_no_run_context(self):
        plan = _plan(query_type="conversational", conversational_response="A DCF discounts cash flows.",
                     ticker="", fiscal_year=0, parameter_notes=[])
        with _patch.object(SupervisorAgent, "route", return_value=plan):
            turn = resolve_turn(_conversation("what is a DCF?"))
        self.assertIsNone(turn.run_context)
        self.assertEqual(turn.conversational_response, "A DCF discounts cash flows.")


class TestSupervisorMilestone(unittest.TestCase):
    def test_research_turn_milestone(self):
        with _patch.object(SupervisorAgent, "route", return_value=_plan()):
            message, details = supervisor_milestone(resolve_turn(_conversation()))
        self.assertEqual(message, "Resolved filing: AAPL FY2025 | Route: dcf_valuation_only")
        self.assertEqual(details["valuation_overrides"], {"wacc": 0.09, "terminal_growth": 0.03})

    def test_conversational_turn_has_no_milestone(self):
        plan = _plan(query_type="conversational", conversational_response="hi", parameter_notes=[])
        with _patch.object(SupervisorAgent, "route", return_value=plan):
            self.assertEqual(supervisor_milestone(resolve_turn(_conversation())), (None, None))
```

Append to `finance_agent_backend/tests/pipeline/test_node.py`:

```python
from datetime import date
from unittest.mock import patch

from app.agents.pipeline.contract import Conversation
from app.agents.pipeline.node import supervisor_node


def _turn(run_context, notes=(), reply=None):
    return SimpleNamespace(run_context=run_context, parameter_notes=notes, conversational_response=reply,
                           route=run_context.route if run_context else "conversational")


class TestSupervisorNode(unittest.TestCase):
    STATE = {"conversation": Conversation(user_query="q", messages=None, session_state=None, ticker="",
                                          fiscal_year=0, run_date=date(2026, 10, 28))}

    def test_research_turn_records_the_turn_and_parameter_notes(self):
        turn = _turn(make_run_context(), notes=("WACC override 9% applied",))
        with patch("app.agents.pipeline.node.resolve_turn", return_value=turn):
            update = asyncio.run(supervisor_node(self.STATE))
        self.assertIs(update["run"], turn)
        self.assertEqual(update["notes"], [RunNote(agent="supervisor", kind="parameter", message="WACC override 9% applied")])
        self.assertNotIn("final", update)

    def test_conversational_turn_writes_the_reply_as_final(self):
        turn = _turn(None, reply="A DCF discounts cash flows.")
        with patch("app.agents.pipeline.node.resolve_turn", return_value=turn):
            update = asyncio.run(supervisor_node(self.STATE))
        self.assertEqual(update["final"], "A DCF discounts cash flows.")
        self.assertNotIn("notes", update)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker exec finance_agent pytest tests/test_query_parameters.py tests/test_agent_supervisor.py tests/pipeline/test_node.py -q`
Expected: FAIL with `ImportError: cannot import name 'resolve_terminal_growth'` (and `resolve_turn`, `supervisor_node`).

- [ ] **Step 3: Implement terminal-growth resolution**

In `finance_agent_backend/app/agents/query_parameters.py`, change the typing import to `from typing import Any, Dict, List, Mapping, Optional` and add after `BETA_BAND = (0.05, 4.0)`:

```python
DEFAULT_TERMINAL_GROWTH = 0.025  # house perpetual growth when the user gives none


def resolve_terminal_growth(overrides: Mapping[str, Any]) -> float:
    """User-specified perpetual growth (already range-checked), else the house default."""
    value = overrides.get("terminal_growth")
    return float(value) if value is not None else DEFAULT_TERMINAL_GROWTH
```

In `finance_agent_backend/app/agents/tools/forecast_drivers.py`, replace line 25 `DEFAULT_TERMINAL_GROWTH = 0.025` with:

```python
from app.agents.query_parameters import DEFAULT_TERMINAL_GROWTH  # one house default for g (PR 5 removes the defaults)
```

Move that import up into the import block at the top of the file.

- [ ] **Step 4: Implement `TurnResolution`, `resolve_turn`, `supervisor_milestone`**

In `finance_agent_backend/app/agents/specialized/supervisor/state_supervisor.py`, change the imports to:

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from app.agents.pipeline.contract import RunContext
```

and append:

```python
@dataclass(frozen=True)
class TurnResolution:
    """What the Supervisor hands the pipeline for one chat turn."""

    route: str
    plan: RoutingPlan
    run_context: Optional[RunContext]
    conversational_response: Optional[str]
    session_update: Optional[Dict[str, Any]]
    parameter_notes: Tuple[str, ...]
```

In `finance_agent_backend/app/agents/specialized/supervisor/agent_supervisor.py`, add the contract import and extend two existing import lines:

```python
from app.agents.pipeline.contract import Conversation, RunContext, StatusLine
from app.agents.query_parameters import SESSION_KEY, parse_valuation_overrides, resolve_session_overrides, resolve_terminal_growth
from app.agents.specialized.supervisor.state_supervisor import QueryType, RoutingPlan, TurnResolution
```

Append at the end of the module:

```python
def resolve_turn(conversation: Conversation) -> TurnResolution:
    """Routes one chat turn and, for a research route, fixes the facts every agent will share."""
    plan = SupervisorAgent().route(
        user_query=conversation.user_query,
        ticker=conversation.ticker,
        fiscal_year=conversation.fiscal_year,
        session_state=conversation.session_state,
        messages=conversation.messages,
        callbacks=conversation.callbacks,
    )
    logger.info(
        f"[supervisor] Resolved {plan.ticker} FY{plan.fiscal_year} "
        f"route={plan.query_type} (year_substituted={plan.year_substituted})"
    )
    run_context = None
    if plan.query_type != "conversational":
        run_context = RunContext(
            ticker=plan.ticker,
            company_name=plan.company_name or plan.ticker,
            fiscal_year=plan.fiscal_year,
            available_fiscal_years=tuple(plan.available_fiscal_years),
            document_id=plan.document_id,
            user_query=conversation.user_query,
            route=plan.query_type,
            year_substituted=plan.year_substituted,
            valuation_overrides=dict(plan.valuation_overrides),
            terminal_growth_rate=resolve_terminal_growth(plan.valuation_overrides),
            valuation_date=conversation.run_date,
            callbacks=conversation.callbacks,
        )
    return TurnResolution(
        route=plan.query_type,
        plan=plan,
        run_context=run_context,
        conversational_response=plan.conversational_response,
        session_update=plan.updated_session_state,
        parameter_notes=tuple(plan.parameter_notes),
    )


def supervisor_milestone(turn: Optional[TurnResolution]) -> StatusLine:
    """SSE status line for a resolved research turn; conversational turns show none."""
    if turn is None or turn.run_context is None:
        return None, None
    plan = turn.plan
    sub_note = " (substituted year)" if plan.year_substituted else ""
    return (
        f"Resolved filing: {plan.ticker} FY{plan.fiscal_year}{sub_note} | Route: {plan.query_type}",
        {
            "ticker": plan.ticker,
            "fiscal_year": plan.fiscal_year,
            "query_type": plan.query_type,
            "year_substituted": plan.year_substituted,
            "valuation_overrides": plan.valuation_overrides,
        },
    )
```

Replace `finance_agent_backend/app/agents/specialized/supervisor/__init__.py` with:

```python
from app.agents.specialized.supervisor.agent_supervisor import (
    SupervisorAgent,
    SupervisorExtraction,
    resolve_turn,
    supervisor_milestone,
)
from app.agents.specialized.supervisor.state_supervisor import (
    QueryType,
    RoutingPlan,
    TurnResolution,
)

__all__ = [
    "SupervisorAgent",
    "SupervisorExtraction",
    "RoutingPlan",
    "QueryType",
    "TurnResolution",
    "resolve_turn",
    "supervisor_milestone",
]
```

- [ ] **Step 5: Implement `supervisor_node`**

In `finance_agent_backend/app/agents/pipeline/node.py`, add to the imports:

```python
from app.agents.specialized.supervisor import resolve_turn
```

and append:

```python
async def supervisor_node(state: ResearchRunState) -> Dict[str, Any]:
    """Resolves the chat turn; a conversational turn writes its reply as the run's final answer."""
    turn = await asyncio.to_thread(resolve_turn, state["conversation"])
    update: Dict[str, Any] = {"run": turn}
    if turn.parameter_notes:
        update["notes"] = [RunNote(agent="supervisor", kind="parameter", message=m) for m in turn.parameter_notes]
    if turn.run_context is None:
        update["final"] = turn.conversational_response
    return update
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `docker exec finance_agent pytest tests/test_query_parameters.py tests/test_agent_supervisor.py tests/pipeline -q`
Expected: all passed. Then run the full suite: `docker exec finance_agent pytest tests/ -q` — all passed, golden unchanged (the old orchestrator still runs the graph).

- [ ] **Step 7: Commit**

```bash
git add finance_agent_backend/app/agents/query_parameters.py finance_agent_backend/app/agents/tools/forecast_drivers.py finance_agent_backend/app/agents/specialized/supervisor finance_agent_backend/app/agents/pipeline/node.py finance_agent_backend/tests/test_query_parameters.py finance_agent_backend/tests/test_agent_supervisor.py finance_agent_backend/tests/pipeline/test_node.py
git commit -m "feat(pipeline): supervisor resolves a turn into a RunContext; one house default for g"
```

### Task 6: AGENT specs for the Strategist, Auditor and Risk Analyst

**Files:**
- Modify: `finance_agent_backend/app/agents/specialized/business_strategist/agent_business_strategist.py` (append run + milestone)
- Modify: `finance_agent_backend/app/agents/specialized/business_strategist/__init__.py` (add `AGENT`)
- Modify: `finance_agent_backend/app/agents/specialized/financial_auditor/agent_financial_auditor.py` and `__init__.py`
- Modify: `finance_agent_backend/app/agents/specialized/risk_analyst/agent_risk_analyst.py` and `__init__.py`
- Test: `finance_agent_backend/tests/pipeline/test_agent_specs.py`

**Interfaces:**
- Consumes: `AgentSpec`, `RunContext`, `Upstream`, `StatusLine` (Task 3).
- Produces: `analyze_business(run, upstream) -> BusinessMoatOutput`, `summarize_moat(moat) -> StatusLine`; `audit_financial_statements(run, upstream) -> FinancialAuditOutput`, `summarize_audit(audit) -> StatusLine`; `analyze_risks(run, upstream) -> RiskAuditOutput`, `summarize_risks(risk) -> StatusLine`; `AGENT` in each door with names `business_strategist` / `financial_auditor` / `risk_analyst`, labels `Business Strategist` / `Financial Auditor` / `Risk Analyst`, output keys `business_moat` / `financial_audit` / `risk_audit`.

- [ ] **Step 1: Write the failing test**

`finance_agent_backend/tests/pipeline/test_agent_specs.py`:

```python
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.agents.pipeline.contract import Upstream
from app.agents.specialized.business_strategist import AGENT as STRATEGIST
from app.agents.specialized.business_strategist import BusinessStrategistAgent
from app.agents.specialized.financial_auditor import AGENT as AUDITOR
from app.agents.specialized.financial_auditor import FinancialAuditorAgent
from app.agents.specialized.risk_analyst import AGENT as RISK
from app.agents.specialized.risk_analyst import RiskAnalystAgent
from tests.forecast_fixtures import make_aapl_audit
from tests.pipeline.fixtures import make_run_context


def _empty(spec):
    return Upstream({}, spec.declared)


class TestPhaseOneSpecs(unittest.TestCase):
    def test_declarations(self):
        for spec, name, key in ((STRATEGIST, "business_strategist", "business_moat"),
                                (AUDITOR, "financial_auditor", "financial_audit"),
                                (RISK, "risk_analyst", "risk_audit")):
            self.assertEqual((spec.name, spec.output_key), (name, key))
            self.assertEqual(spec.declared, frozenset())
            self.assertFalse(spec.writes_final)

    def test_strategist_runs_on_the_filing(self):
        ctx = make_run_context(callbacks=["cb"])
        with patch.object(BusinessStrategistAgent, "analyze", return_value="moat") as analyze:
            self.assertEqual(STRATEGIST.run(ctx, _empty(STRATEGIST)), "moat")
        self.assertEqual(analyze.call_args.kwargs, {"ticker": "AAPL", "fiscal_year": 2025, "callbacks": ["cb"]})

    def test_auditor_gets_the_available_years_as_a_list(self):
        with patch.object(FinancialAuditorAgent, "audit", return_value="audit") as audit:
            AUDITOR.run(make_run_context(), _empty(AUDITOR))
        self.assertEqual(audit.call_args.kwargs["available_fiscal_years"], [2025, 2024])

    def test_risk_analyst_gets_the_available_years_as_a_list(self):
        with patch.object(RiskAnalystAgent, "analyze", return_value="risk") as analyze:
            RISK.run(make_run_context(), _empty(RISK))
        self.assertEqual(analyze.call_args.kwargs["available_fiscal_years"], [2025, 2024])

    def test_milestones_when_unavailable(self):
        self.assertEqual(STRATEGIST.milestone(None), ("Economic moat analysis unavailable", None))
        self.assertEqual(AUDITOR.milestone(None), ("Financial statement audit unavailable", None))
        self.assertEqual(RISK.milestone(None), ("Analyzed Item 1A risks & threats (no risk output)", None))

    def test_strategist_milestone_names_the_moat(self):
        self.assertEqual(STRATEGIST.milestone(SimpleNamespace(economic_moat_type="Network Effects"))[0],
                         "Completed economic moat analysis (Moat: Network Effects)")

    def test_auditor_milestone_counts_high_and_medium_findings(self):
        message, _ = AUDITOR.milestone(make_aapl_audit())
        self.assertRegex(message, r"^Audited financial statements & ratios \(\d+ high/medium forensic findings, \w+ extraction\)$")

    def test_risk_milestone_for_a_filing_without_risk_factors(self):
        risk = SimpleNamespace(data_quality=SimpleNamespace(coverage_mode="not_provided"))
        self.assertEqual(RISK.milestone(risk)[0],
                         "Analyzed Item 1A risks & threats (filing provides no risk factors; nothing rated)")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec finance_agent pytest tests/pipeline/test_agent_specs.py -v`
Expected: FAIL with `ImportError: cannot import name 'AGENT' from 'app.agents.specialized.business_strategist'`

- [ ] **Step 3: Implement the three run functions and milestones**

Append to `agent_business_strategist.py` (and add `from app.agents.pipeline.contract import RunContext, StatusLine, Upstream` to its imports):

```python
def analyze_business(run: RunContext, upstream: Upstream) -> BusinessMoatOutput:
    """Pipeline entry: Item 1 business model and moat for the run's filing."""
    return BusinessStrategistAgent().analyze(ticker=run.ticker, fiscal_year=run.fiscal_year, callbacks=run.callbacks)


def summarize_moat(moat: Optional[BusinessMoatOutput]) -> StatusLine:
    if moat is None:
        return "Economic moat analysis unavailable", None
    return f"Completed economic moat analysis (Moat: {moat.economic_moat_type})", None
```

Append to `agent_financial_auditor.py` (same contract import):

```python
def audit_financial_statements(run: RunContext, upstream: Upstream) -> FinancialAuditOutput:
    """Pipeline entry: multi-year Item 8 audit anchored on the run's filing."""
    return FinancialAuditorAgent().audit(
        ticker=run.ticker,
        fiscal_year=run.fiscal_year,
        available_fiscal_years=list(run.available_fiscal_years),
        callbacks=run.callbacks,
    )


def summarize_audit(audit: Optional[FinancialAuditOutput]) -> StatusLine:
    if audit is None:
        return "Financial statement audit unavailable", None
    severe_cnt = sum(1 for f in audit.forensic_findings if f.severity in ("high", "medium"))
    return (
        f"Audited financial statements & ratios ({severe_cnt} high/medium forensic findings, "
        f"{audit.data_quality.extraction_mode} extraction)",
        None,
    )
```

Append to `agent_risk_analyst.py` (same contract import):

```python
def analyze_risks(run: RunContext, upstream: Upstream) -> RiskAuditOutput:
    """Pipeline entry: verified Item 1A risk inventory with the prior-year comparison."""
    return RiskAnalystAgent().analyze(
        ticker=run.ticker,
        fiscal_year=run.fiscal_year,
        available_fiscal_years=list(run.available_fiscal_years),
        callbacks=run.callbacks,
    )


def summarize_risks(risk: Optional[RiskAuditOutput]) -> StatusLine:
    """SSE status line summarizing the verified risk inventory."""
    if risk is None:
        return "Analyzed Item 1A risks & threats (no risk output)", None
    if risk.data_quality.coverage_mode == "not_provided":
        return "Analyzed Item 1A risks & threats (filing provides no risk factors; nothing rated)", None
    severe = sum(1 for r in risk.identified_risks if r.severity == "Severe")
    coverage = risk.data_quality.coverage_mode.replace("_", "-")
    if risk.data_quality.section_located_by == "heading_boundary":
        coverage += " (located by headings)"
    if risk.disclosure_changes:
        new_cnt = sum(1 for r in risk.identified_risks if r.disclosure_change in ("New", "Expanded"))
        change = f"{new_cnt} new/expanded vs FY{risk.disclosure_changes.prior_fiscal_year}"
    else:
        change = "no prior-year baseline"
    return (
        f"Analyzed Item 1A risks & threats ({len(risk.identified_risks)} verified risks, {severe} severe; "
        f"{coverage} read; {change})",
        None,
    )
```

Make sure `Optional` is imported in each of the three modules.

- [ ] **Step 4: Export `AGENT` from each door**

Add to `business_strategist/__init__.py` (and `"AGENT"` to `__all__`):

```python
from app.agents.pipeline.contract import AgentSpec
from app.agents.specialized.business_strategist.agent_business_strategist import analyze_business, summarize_moat

AGENT = AgentSpec(
    name="business_strategist",
    label="Business Strategist",
    output_key="business_moat",
    run=analyze_business,
    milestone=summarize_moat,
)
```

Add to `financial_auditor/__init__.py`:

```python
from app.agents.pipeline.contract import AgentSpec
from app.agents.specialized.financial_auditor.agent_financial_auditor import audit_financial_statements, summarize_audit

AGENT = AgentSpec(
    name="financial_auditor",
    label="Financial Auditor",
    output_key="financial_audit",
    run=audit_financial_statements,
    milestone=summarize_audit,
)
```

Add to `risk_analyst/__init__.py`:

```python
from app.agents.pipeline.contract import AgentSpec
from app.agents.specialized.risk_analyst.agent_risk_analyst import analyze_risks, summarize_risks

AGENT = AgentSpec(
    name="risk_analyst",
    label="Risk Analyst",
    output_key="risk_audit",
    run=analyze_risks,
    milestone=summarize_risks,
)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `docker exec finance_agent pytest tests/pipeline/test_agent_specs.py -v` → 8 passed. Then `docker exec finance_agent pytest tests/ -q` → all passed.

- [ ] **Step 6: Commit**

```bash
git add finance_agent_backend/app/agents/specialized/business_strategist finance_agent_backend/app/agents/specialized/financial_auditor finance_agent_backend/app/agents/specialized/risk_analyst finance_agent_backend/tests/pipeline/test_agent_specs.py
git commit -m "feat(pipeline): AGENT specs for the strategist, auditor and risk analyst"
```

### Task 7: AGENT specs for the Forecaster and Valuation (market data moves into the agent)

**Files:**
- Modify: `finance_agent_backend/app/agents/specialized/forecasting_analyst/agent_forecasting_analyst.py` and `__init__.py`
- Create: `finance_agent_backend/app/agents/specialized/valuation_specialist/context.py`
- Modify: `finance_agent_backend/app/agents/specialized/valuation_specialist/agent_valuation_specialist.py` and `__init__.py`
- Modify: `finance_agent_backend/tests/test_agent_forecasting_analyst.py` (replace `TestForecastNodeWiring`; drop the orchestrator import)
- Modify: `finance_agent_backend/tests/test_agent_valuation.py` (replace `TestValuationNodeWiring` and `test_07_status_message`; drop the orchestrator import)

**Interfaces:**
- Consumes: Task 3 contract; `make_agent_node` (Task 4) for skip / failure tests; `parse_valuation_overrides` (existing).
- Produces: `forecast_cash_flows(run, upstream) -> ForecastOutput`, `summarize_forecast(fc) -> StatusLine`; `valuation_specialist.context.market_inputs(ticker: str, overrides: Mapping) -> MarketInputs`, `valuation_specialist.context.balance_sheet_date(ticker: str, fiscal_year: int) -> Optional[str]`; `value_equity(run, upstream) -> DCFValuationOutput`, `summarize_valuation(val) -> StatusLine`; `AGENT` in both doors (`forecasting_analyst` needs `financial_audit`, uses `business_moat`, `risk_audit`; `valuation_specialist` needs `financial_audit`, uses `forecast`).

- [ ] **Step 1: Replace the forecaster node-wiring tests (failing first)**

In `tests/test_agent_forecasting_analyst.py`, delete line 12 (`from app.agents.orchestrator import MultiAgentOrchestrator, forecasting_analyst_node`) and replace the whole `class TestForecastNodeWiring` with:

```python
class TestForecastSpec(unittest.TestCase):
    def test_passes_risk_and_the_runs_terminal_growth(self):
        from app.agents.pipeline.contract import Upstream
        from app.agents.specialized.forecasting_analyst import AGENT
        from tests.pipeline.fixtures import make_run_context

        risk = SimpleNamespace(overall_risk_profile="High", identified_risks=[])
        upstream = Upstream({"financial_audit": make_aapl_audit(), "business_moat": None, "risk_audit": risk},
                            AGENT.declared)
        with patch.object(ForecastingAnalystAgent, "forecast", return_value="forecast") as mock_forecast:
            self.assertEqual(AGENT.run(make_run_context(terminal_growth_rate=0.03), upstream), "forecast")
        kwargs = mock_forecast.call_args.kwargs
        self.assertIs(kwargs["risk_audit"], risk)
        self.assertAlmostEqual(kwargs["terminal_growth"], 0.03)

    def test_skips_without_an_audit(self):
        from app.agents.pipeline.node import make_agent_node
        from app.agents.specialized.forecasting_analyst import AGENT
        from tests.pipeline.fixtures import make_run_context

        state = {"run": SimpleNamespace(run_context=make_run_context()), "outputs": {"financial_audit": None}}
        update = asyncio.run(make_agent_node(AGENT)(state))
        self.assertIsNone(update["outputs"]["forecast"])
        self.assertIn("no audited financial statements", update["notes"][0].message)

    def test_failure_is_recorded_not_raised(self):
        from app.agents.pipeline.node import make_agent_node
        from app.agents.specialized.forecasting_analyst import AGENT
        from tests.pipeline.fixtures import make_run_context

        state = {"run": SimpleNamespace(run_context=make_run_context()), "outputs": {"financial_audit": make_aapl_audit()}}
        with patch.object(ForecastingAnalystAgent, "forecast", side_effect=ValueError("base_revenue must be positive")):
            update = asyncio.run(make_agent_node(AGENT)(state))
        self.assertIsNone(update["outputs"]["forecast"])
        self.assertIn("Forecasting Analyst failed (ValueError: base_revenue must be positive)", update["notes"][0].message)

    def test_status_message(self):
        from app.agents.specialized.forecasting_analyst import AGENT

        message, _ = AGENT.milestone(make_forecast_output())
        self.assertIn("5-year UFCF forecast", message)
        self.assertIn("historical fade", message)
        self.assertEqual(AGENT.milestone(None), ("Forecast unavailable", None))
```

(Keep `asyncio`, `SimpleNamespace`, `patch`, `make_aapl_audit`, `make_forecast_output` imports; they are already used in this file.)

- [ ] **Step 2: Replace the valuation node-wiring tests (failing first)**

In `tests/test_agent_valuation.py`, delete line 14 (`from app.agents.orchestrator import MultiAgentOrchestrator, valuation_specialist_node`). Replace `test_07_status_message` with:

```python
    def test_07_status_message(self):
        from app.agents.specialized.valuation_specialist import AGENT

        out = make_valuation_output()
        msg, _ = AGENT.milestone(out)
        self.assertIn(f"${out.implied_fair_value_per_share:.2f}/share", msg)
        self.assertIn(f"bear ${out.bear_fair_value:.2f}", msg)
        self.assertIn("CAPM WACC", msg)
        self.assertEqual(AGENT.milestone(None), ("Valuation unavailable", None))
```

Replace the whole `class TestValuationNodeWiring` with:

```python
class TestValuationSpecWiring(unittest.TestCase):
    CONTEXT = "app.agents.specialized.valuation_specialist.context"
    AGENT_MODULE = "app.agents.specialized.valuation_specialist.agent_valuation_specialist"

    def _run(self, query: str, market_data: dict, **ctx_changes):
        from app.agents.pipeline.contract import Upstream
        from app.agents.specialized.valuation_specialist import AGENT
        from tests.pipeline.fixtures import make_run_context

        ctx = make_run_context(valuation_overrides=parse_valuation_overrides(query).values(), **ctx_changes)
        upstream = Upstream({"financial_audit": make_aapl_audit(), "forecast": make_forecast_output()}, AGENT.declared)
        with patch(f"{self.CONTEXT}.fetch_market_context", return_value=market_data), \
                patch(f"{self.AGENT_MODULE}.balance_sheet_date", return_value=AAPL_BALANCE_SHEET_DATE), \
                patch.object(ValuationSpecialistAgent, "value", return_value="sentinel") as mock_value:
            self.assertEqual(AGENT.run(ctx, upstream), "sentinel")
        return ctx, mock_value.call_args.kwargs

    def test_user_overrides_take_precedence(self):
        live = {"share_price": 230.0, "market_cap": 3_450_000.0, "beta": 1.1, "beta_source": "yfinance_raw",
                "risk_free_rate": 0.044, "risk_free_rate_source": "us_10y_treasury_live",
                "market_data_source": "yfinance_live"}
        ctx, kwargs = self._run("value AAPL with beta 1.3, share price 200 and WACC of 9%", live)
        market: MarketInputs = kwargs["market"]
        self.assertEqual((market.beta, market.beta_source), (1.3, "user_provided"))
        self.assertEqual((market.share_price, market.share_price_source), (200.0, "user_provided"))
        self.assertIsNone(market.market_cap)  # live cap is inconsistent with a user price
        self.assertEqual(market.risk_free_rate, 0.044)
        self.assertAlmostEqual(kwargs["wacc_override"], 0.09)
        self.assertEqual(kwargs["balance_sheet_date"], AAPL_BALANCE_SHEET_DATE)
        self.assertEqual(kwargs["valuation_date"], ctx.valuation_date)
        self.assertIsNotNone(kwargs["forecast"])

    def test_live_market_data_passes_through(self):
        live = {"share_price": 230.0, "market_cap": 3_450_000.0, "beta": None, "beta_source": "unavailable",
                "risk_free_rate": 0.042, "risk_free_rate_source": "house_default_config",
                "market_data_source": "yfinance_live"}
        _, kwargs = self._run("DCF for AAPL", live)
        market: MarketInputs = kwargs["market"]
        self.assertEqual(market.share_price_source, "yfinance_live")
        self.assertEqual(market.market_cap, 3_450_000.0)
        self.assertIsNone(market.beta)
        self.assertEqual(market.risk_free_rate_source, "house_default_config")

    def test_forecast_and_valuation_share_one_terminal_growth(self):
        from app.agents.pipeline.contract import Upstream
        from app.agents.specialized.forecasting_analyst import AGENT as FORECASTER
        from app.agents.specialized.forecasting_analyst import ForecastingAnalystAgent
        from tests.pipeline.fixtures import make_run_context

        live = {"share_price": 230.0, "market_cap": 3_450_000.0, "beta": 1.1, "beta_source": "yfinance_raw",
                "risk_free_rate": 0.044, "risk_free_rate_source": "us_10y_treasury_live",
                "market_data_source": "yfinance_live"}
        ctx, valuation_kwargs = self._run("DCF for AAPL", live, terminal_growth_rate=0.031)
        upstream = Upstream({"financial_audit": make_aapl_audit()}, FORECASTER.declared)
        with patch.object(ForecastingAnalystAgent, "forecast", return_value="forecast") as mock_forecast:
            FORECASTER.run(ctx, upstream)
        self.assertEqual(mock_forecast.call_args.kwargs["terminal_growth"], valuation_kwargs["terminal_growth_rate"])
        self.assertAlmostEqual(valuation_kwargs["terminal_growth_rate"], 0.031)
```

Remove the now-unused `date` import from that file only if nothing else in it uses `date`.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `docker exec finance_agent pytest tests/test_agent_forecasting_analyst.py tests/test_agent_valuation.py -q`
Expected: FAIL with `ImportError: cannot import name 'AGENT'`.

- [ ] **Step 4: Implement the forecaster run function, milestone and AGENT**

Append to `agent_forecasting_analyst.py` (add `from app.agents.pipeline.contract import RunContext, StatusLine, Upstream`):

```python
def forecast_cash_flows(run: RunContext, upstream: Upstream) -> ForecastOutput:
    """Pipeline entry: UFCF forecast from the audit, guided by moat and risk when they ran."""
    return ForecastingAnalystAgent().forecast(
        ticker=run.ticker,
        fiscal_year=run.fiscal_year,
        financial_audit=upstream.get("financial_audit"),
        business_moat=upstream.get("business_moat"),
        risk_audit=upstream.get("risk_audit"),
        terminal_growth=run.terminal_growth_rate,
        callbacks=run.callbacks,
    )


def summarize_forecast(fc: Optional[ForecastOutput]) -> StatusLine:
    if fc is None:
        return "Forecast unavailable", None
    source = "MD&A-informed" if fc.guidance_source == "mdna_informed" else "historical fade"
    return (
        f"Built {fc.forecast_horizon_years}-year UFCF forecast ({source}, revenue CAGR "
        f"{fc.revenue_cagr_pct:.1f}%, {len(fc.scenarios)} scenarios)",
        None,
    )
```

Add to `forecasting_analyst/__init__.py` (and `"AGENT"` to `__all__`):

```python
from app.agents.pipeline.contract import AgentSpec
from app.agents.specialized.forecasting_analyst.agent_forecasting_analyst import forecast_cash_flows, summarize_forecast

AGENT = AgentSpec(
    name="forecasting_analyst",
    label="Forecasting Analyst",
    output_key="forecast",
    run=forecast_cash_flows,
    milestone=summarize_forecast,
    needs=frozenset({"financial_audit"}),
    uses=frozenset({"business_moat", "risk_audit"}),
    skip_message="Forecast skipped: no audited financial statements were available.",
)
```

- [ ] **Step 5: Move the valuation's outside data into `context.py`**

`finance_agent_backend/app/agents/specialized/valuation_specialist/context.py`:

```python
"""Outside data the valuation needs: market quotes (with the user's overrides) and the balance-sheet date."""

import logging
from typing import Any, Mapping, Optional

from app.agents.tools.market_data_tools import fetch_market_context
from app.agents.tools.valuation_model import MarketInputs
from app.config import settings
from app.database import SessionLocal
from app.models import ANNUAL_REPORT_FILING_TYPE, Document

logger = logging.getLogger("finance_agent.agents.valuation_specialist")


def balance_sheet_date(ticker: str, fiscal_year: int) -> Optional[str]:
    """Fiscal year-end (period_end_date) of the anchor 10-K, used to roll the DCF forward to today."""
    db = SessionLocal()
    try:
        row = (
            db.query(Document.period_end_date)
            .filter(
                Document.ticker == ticker.upper(),
                Document.fiscal_year == fiscal_year,
                Document.filing_type == ANNUAL_REPORT_FILING_TYPE,
            )
            .first()
        )
        return row[0] if row and row[0] else None
    except Exception as e:
        logger.warning(f"Balance-sheet date lookup failed for {ticker} FY{fiscal_year}: {e}")
        return None
    finally:
        db.close()


def market_inputs(ticker: str, overrides: Mapping[str, Any]) -> MarketInputs:
    """Live price, market cap, raw beta and 10Y Treasury, with user price / beta taking precedence."""
    user_beta, user_share_price = overrides.get("beta"), overrides.get("share_price")
    market_data = fetch_market_context(ticker)
    live_price = market_data.get("share_price")
    return MarketInputs(
        share_price=user_share_price if user_share_price is not None else live_price,
        share_price_source=(
            "user_provided" if user_share_price is not None
            else ("yfinance_live" if live_price is not None else "unavailable")
        ),
        # A user price makes the live market cap inconsistent; weights then use price x valuation shares.
        market_cap=None if user_share_price is not None else market_data.get("market_cap"),
        beta=user_beta if user_beta is not None else market_data.get("beta"),
        beta_source="user_provided" if user_beta is not None else market_data.get("beta_source", "unavailable"),
        risk_free_rate=market_data.get("risk_free_rate", settings.DEFAULT_RISK_FREE_RATE),
        risk_free_rate_source=market_data.get("risk_free_rate_source", "house_default_config"),
        market_data_source=market_data.get("market_data_source", "fallback_offline"),
    )
```

- [ ] **Step 6: Implement the valuation run function, milestone and AGENT**

Append to `agent_valuation_specialist.py` (add imports `from app.agents.pipeline.contract import RunContext, StatusLine, Upstream` and `from app.agents.specialized.valuation_specialist.context import balance_sheet_date, market_inputs`; ensure `Optional` is imported):

```python
def value_equity(run: RunContext, upstream: Upstream) -> DCFValuationOutput:
    """Pipeline entry: CAPM WACC, DCF, scenarios and sensitivity at today's market inputs."""
    market = market_inputs(run.ticker, run.valuation_overrides)
    bs_date = balance_sheet_date(run.ticker, run.fiscal_year)
    user_wacc = run.valuation_overrides.get("wacc")
    wacc_label = f"{user_wacc * 100:.2f}%" if user_wacc is not None else "CAPM"
    logger.info(
        f"[valuation_specialist] Valuing {run.ticker} (price=${market.share_price} [{market.share_price_source}], "
        f"beta={market.beta} [{market.beta_source}], rf={market.risk_free_rate} [{market.risk_free_rate_source}], "
        f"g={run.terminal_growth_rate * 100:.1f}%, balance_sheet_date={bs_date}, user_wacc={wacc_label})"
    )
    return ValuationSpecialistAgent().value(
        ticker=run.ticker,
        fiscal_year=run.fiscal_year,
        financial_audit=upstream.get("financial_audit"),
        forecast=upstream.get("forecast"),
        market=market,
        terminal_growth_rate=run.terminal_growth_rate,
        wacc_override=user_wacc,
        valuation_date=run.valuation_date,
        balance_sheet_date=bs_date,
        callbacks=run.callbacks,
    )


def summarize_valuation(val: Optional[DCFValuationOutput]) -> StatusLine:
    if val is None:
        return "Valuation unavailable", None
    message = (
        f"Discounted at {val.wacc_audit.wacc_pct:.2f}% "
        f"({'user override' if val.wacc_audit.wacc_source == 'user_override' else 'CAPM WACC'}); "
        f"base-case DCF fair value ${val.implied_fair_value_per_share:.2f}/share"
    )
    if val.bear_fair_value is not None and val.bull_fair_value is not None:
        message += f" (bear ${val.bear_fair_value:.2f} - bull ${val.bull_fair_value:.2f})"
    return message, None
```

If `logger` is not defined in `agent_valuation_specialist.py`, add `logger = logging.getLogger("finance_agent.agents.valuation_specialist")` (and `import logging`) at module level.

Add to `valuation_specialist/__init__.py` (and `"AGENT"` to `__all__`):

```python
from app.agents.pipeline.contract import AgentSpec
from app.agents.specialized.valuation_specialist.agent_valuation_specialist import summarize_valuation, value_equity

AGENT = AgentSpec(
    name="valuation_specialist",
    label="Valuation Specialist",
    output_key="dcf_valuation",
    run=value_equity,
    milestone=summarize_valuation,
    needs=frozenset({"financial_audit"}),
    uses=frozenset({"forecast"}),
    skip_message="Valuation skipped: no audited financial statements were available.",
)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `docker exec finance_agent pytest tests/test_agent_forecasting_analyst.py tests/test_agent_valuation.py tests/pipeline -q` → all passed. Then `docker exec finance_agent pytest tests/ -q` → all passed (golden unchanged: the old orchestrator still runs).

- [ ] **Step 8: Commit**

```bash
git add finance_agent_backend/app/agents/specialized/forecasting_analyst finance_agent_backend/app/agents/specialized/valuation_specialist finance_agent_backend/tests/test_agent_forecasting_analyst.py finance_agent_backend/tests/test_agent_valuation.py
git commit -m "feat(pipeline): forecaster and valuation AGENT specs; market data moves into valuation context"
```

### Task 8: AGENT spec for the Lead Synthesizer

**Files:**
- Modify: `finance_agent_backend/app/agents/specialized/lead_synthesizer/agent_lead_synthesizer.py` and `__init__.py`
- Test: `finance_agent_backend/tests/pipeline/test_agent_specs.py` (new class)

**Interfaces:**
- Consumes: Task 3 contract.
- Produces: `write_research_report(run, upstream) -> Final10KResearchReport`, `summarize_report(report) -> StatusLine`; `AGENT` (`lead_synthesizer`, label `Lead Synthesizer`, output_key `final_report`, uses the five research outputs, needs_any `business_moat` / `financial_audit` / `risk_audit`, `writes_final=True`).

- [ ] **Step 1: Write the failing test**

Append to `tests/pipeline/test_agent_specs.py`:

```python
from app.agents.pipeline.contract import RunNote
from app.agents.specialized.lead_synthesizer import AGENT as SYNTHESIZER
from app.agents.specialized.lead_synthesizer import LeadSynthesizerAgent


class TestSynthesizerSpec(unittest.TestCase):
    def test_declarations(self):
        self.assertTrue(SYNTHESIZER.writes_final)
        self.assertEqual(SYNTHESIZER.uses, frozenset({"business_moat", "financial_audit", "forecast",
                                                       "dcf_valuation", "risk_audit"}))
        self.assertEqual(SYNTHESIZER.needs_any, frozenset({"business_moat", "financial_audit", "risk_audit"}))

    def test_passes_route_notes_and_every_upstream_output(self):
        notes = (RunNote(agent="supervisor", kind="parameter", message="WACC override 9% applied"),)
        upstream = Upstream({"financial_audit": "audit", "risk_audit": "risk"}, SYNTHESIZER.declared, notes)
        ctx = make_run_context(route="financial_audit_only", year_substituted=True, company_name="Apple Inc.")
        with patch.object(LeadSynthesizerAgent, "synthesize", return_value="report") as synthesize:
            self.assertEqual(SYNTHESIZER.run(ctx, upstream), "report")
        kwargs = synthesize.call_args.kwargs
        self.assertEqual(kwargs["query_type"], "financial_audit_only")
        self.assertEqual(kwargs["pipeline_notes"], ["WACC override 9% applied"])
        self.assertTrue(kwargs["year_substituted"])
        self.assertEqual((kwargs["financial_audit"], kwargs["risk_audit"], kwargs["business_moat"]), ("audit", "risk", None))
        self.assertEqual(kwargs["available_fiscal_years"], [2025, 2024])
        self.assertEqual(kwargs["company_name"], "Apple Inc.")

    def test_milestone(self):
        self.assertEqual(SYNTHESIZER.milestone("report"),
                         ("Synthesized 3-Pillar Thesis and compiled final publication report.", None))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec finance_agent pytest tests/pipeline/test_agent_specs.py -q`
Expected: FAIL with `ImportError: cannot import name 'AGENT' from 'app.agents.specialized.lead_synthesizer'`

- [ ] **Step 3: Implement**

Append to `agent_lead_synthesizer.py` (add `from app.agents.pipeline.contract import RunContext, StatusLine, Upstream`):

```python
def write_research_report(run: RunContext, upstream: Upstream) -> Final10KResearchReport:
    """Pipeline entry: assembles the route's report from every research output that exists."""
    return LeadSynthesizerAgent().synthesize(
        ticker=run.ticker,
        company_name=run.company_name,
        fiscal_year=run.fiscal_year,
        business_moat=upstream.get("business_moat"),
        financial_audit=upstream.get("financial_audit"),
        forecast=upstream.get("forecast"),
        dcf_valuation=upstream.get("dcf_valuation"),
        risk_audit=upstream.get("risk_audit"),
        year_substituted=run.year_substituted,
        available_fiscal_years=list(run.available_fiscal_years),
        user_query=run.user_query,
        query_type=run.route,
        pipeline_notes=[note.message for note in upstream.notes],
        callbacks=run.callbacks,
    )


def summarize_report(report: Optional[Final10KResearchReport]) -> StatusLine:
    return "Synthesized 3-Pillar Thesis and compiled final publication report.", None
```

Add to `lead_synthesizer/__init__.py` (and `"AGENT"` to `__all__`):

```python
from app.agents.pipeline.contract import AgentSpec
from app.agents.specialized.lead_synthesizer.agent_lead_synthesizer import summarize_report, write_research_report

AGENT = AgentSpec(
    name="lead_synthesizer",
    label="Lead Synthesizer",
    output_key="final_report",
    run=write_research_report,
    milestone=summarize_report,
    uses=frozenset({"business_moat", "financial_audit", "forecast", "dcf_valuation", "risk_audit"}),
    needs_any=frozenset({"business_moat", "financial_audit", "risk_audit"}),
    writes_final=True,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker exec finance_agent pytest tests/pipeline -q` → all passed; `docker exec finance_agent pytest tests/ -q` → all passed.

- [ ] **Step 5: Commit**

```bash
git add finance_agent_backend/app/agents/specialized/lead_synthesizer finance_agent_backend/tests/pipeline/test_agent_specs.py
git commit -m "feat(pipeline): lead synthesizer AGENT spec (writes final, needs any research output)"
```

### Task 9: Route table and graph builder

**Files:**
- Create: `finance_agent_backend/app/agents/pipeline/routes.py`
- Create: `finance_agent_backend/app/agents/pipeline/graph.py`
- Test: `finance_agent_backend/tests/pipeline/test_routes.py`, `finance_agent_backend/tests/pipeline/test_graph.py`

**Interfaces:**
- Consumes: every agent's `AGENT` (Tasks 6–8); `make_agent_node`, `supervisor_node` (Tasks 4–5); `ResearchRunState` (Task 4).
- Produces: `ROUTES: Dict[str, List[Step]]`, `Step = Union[AgentSpec, Sequence[AgentSpec]]`, `CONVERSATIONAL = "conversational"`, `step_specs(step) -> List[AgentSpec]`, `all_specs(routes=ROUTES) -> List[AgentSpec]`; `first_step(route, routes=ROUTES) -> List[str]`, `next_step(agent_name, route, routes=ROUTES) -> List[str]`, `build_graph(routes=ROUTES) -> CompiledStateGraph`.

- [ ] **Step 1: Write the failing tests**

`finance_agent_backend/tests/pipeline/test_routes.py`:

```python
import unittest

from langgraph.graph import END

from app.agents.pipeline.graph import first_step, next_step
from app.agents.pipeline.routes import ROUTES, all_specs, step_specs

# SSE `node` names are part of the API contract.
NODE_NAMES = {"business_strategist", "financial_auditor", "risk_analyst", "forecasting_analyst",
              "valuation_specialist", "lead_synthesizer"}


class TestRouteTable(unittest.TestCase):
    def test_routes_match_the_supervisors_report_types(self):
        self.assertEqual(set(ROUTES), {"full_10k_report", "dcf_valuation_only", "financial_audit_only",
                                       "business_moat_only", "risk_factors_only"})

    def test_every_need_is_produced_earlier_in_its_route(self):
        for route, steps in ROUTES.items():
            produced = set()
            for step in steps:
                for spec in step_specs(step):
                    self.assertLessEqual(spec.needs, produced, f"{route}: {spec.name} needs {spec.needs - produced}")
                produced |= {spec.output_key for spec in step_specs(step)}

    def test_every_route_ends_with_the_synthesizer_and_can_satisfy_it(self):
        for route, steps in ROUTES.items():
            last = step_specs(steps[-1])
            self.assertEqual([spec.name for spec in last], ["lead_synthesizer"], route)
            earlier = {spec.output_key for step in steps[:-1] for spec in step_specs(step)}
            self.assertTrue(last[0].needs_any & earlier, route)

    def test_node_names_are_unchanged(self):
        self.assertEqual({spec.name for spec in all_specs()}, NODE_NAMES)


class TestRouting(unittest.TestCase):
    def test_full_report_starts_with_three_agents_in_parallel(self):
        self.assertEqual(first_step("full_10k_report"), ["business_strategist", "financial_auditor", "risk_analyst"])

    def test_auditor_successor_depends_on_the_route(self):
        self.assertEqual(next_step("financial_auditor", "financial_audit_only"), ["lead_synthesizer"])
        self.assertEqual(next_step("financial_auditor", "dcf_valuation_only"), ["forecasting_analyst"])
        self.assertEqual(next_step("financial_auditor", "full_10k_report"), ["forecasting_analyst"])

    def test_synthesizer_ends_every_route(self):
        for route in ROUTES:
            self.assertEqual(next_step("lead_synthesizer", route), [END])

    def test_unknown_route_ends(self):
        self.assertEqual(first_step("no_such_route"), [END])
```

`finance_agent_backend/tests/pipeline/test_graph.py`:

```python
import asyncio
import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from app.agents.pipeline.contract import AgentSpec, Conversation
from app.agents.pipeline.graph import build_graph
from tests.pipeline.fixtures import make_run_context

CONVERSATION = Conversation(user_query="q", messages=None, session_state=None, ticker="", fiscal_year=0,
                            run_date=date(2026, 10, 28))


def _milestone(output):
    return None, None


def _spec(name, key, run, **fields):
    return AgentSpec(name=name, label=name, output_key=key, run=run, milestone=_milestone, **fields)


def _turn(route, run_context, notes=(), reply=None):
    return SimpleNamespace(route=route, run_context=run_context, parameter_notes=notes, conversational_response=reply)


def _invoke(routes, turn):
    with patch("app.agents.pipeline.node.resolve_turn", return_value=turn):
        return asyncio.run(build_graph(routes).ainvoke({"conversation": CONVERSATION}))


class TestGraph(unittest.TestCase):
    def test_parallel_agents_all_reach_the_final_agent(self):
        alpha = _spec("alpha", "a", lambda ctx, up: "A")
        beta = _spec("beta", "b", lambda ctx, up: "B")
        final = _spec("omega", "report", lambda ctx, up: f"{up.get('a')}+{up.get('b')}",
                      uses=frozenset({"a", "b"}), writes_final=True)
        state = _invoke({"r": [[alpha, beta], final]}, _turn("r", make_run_context(route="r")))
        self.assertEqual(state["outputs"], {"a": "A", "b": "B"})
        self.assertEqual(state["final"], "A+B")

    def test_conversational_turn_runs_no_agent(self):
        def explode(ctx, up):
            raise AssertionError("no agent may run on a conversational turn")

        routes = {"r": [_spec("omega", "report", explode, writes_final=True)]}
        state = _invoke(routes, _turn("conversational", None, notes=("Remembered beta 1.4.",), reply="Hi."))
        self.assertEqual(state["final"], "Hi.")
        self.assertEqual([note.message for note in state["notes"]], ["Remembered beta 1.4."])
        self.assertNotIn("outputs", state)

    def test_real_graph_compiles_with_todays_node_names(self):
        nodes = set(build_graph().get_graph().nodes)
        self.assertLessEqual({"supervisor", "business_strategist", "financial_auditor", "risk_analyst",
                              "forecasting_analyst", "valuation_specialist", "lead_synthesizer"}, nodes)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker exec finance_agent pytest tests/pipeline/test_routes.py tests/pipeline/test_graph.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.agents.pipeline.graph'`

- [ ] **Step 3: Write `routes.py`**

```python
"""Which agents run for each report type, in order; a nested list runs in parallel and joins before the next step."""

from typing import Dict, List, Sequence, Union

from app.agents.pipeline.contract import AgentSpec
from app.agents.specialized.business_strategist import AGENT as business_strategist
from app.agents.specialized.financial_auditor import AGENT as financial_auditor
from app.agents.specialized.forecasting_analyst import AGENT as forecasting_analyst
from app.agents.specialized.lead_synthesizer import AGENT as lead_synthesizer
from app.agents.specialized.risk_analyst import AGENT as risk_analyst
from app.agents.specialized.valuation_specialist import AGENT as valuation_specialist

Step = Union[AgentSpec, Sequence[AgentSpec]]

CONVERSATIONAL = "conversational"  # answered by the Supervisor; never routed to agents

ROUTES: Dict[str, List[Step]] = {
    "full_10k_report": [
        [business_strategist, financial_auditor, risk_analyst],
        forecasting_analyst,
        valuation_specialist,
        lead_synthesizer,
    ],
    "dcf_valuation_only": [financial_auditor, forecasting_analyst, valuation_specialist, lead_synthesizer],
    "financial_audit_only": [financial_auditor, lead_synthesizer],
    "business_moat_only": [business_strategist, lead_synthesizer],
    "risk_factors_only": [risk_analyst, lead_synthesizer],
}


def step_specs(step: Step) -> List[AgentSpec]:
    return [step] if isinstance(step, AgentSpec) else list(step)


def all_specs(routes: Dict[str, List[Step]] = ROUTES) -> List[AgentSpec]:
    """Every routed agent once, in first-seen order."""
    seen: Dict[str, AgentSpec] = {}
    for steps in routes.values():
        for step in steps:
            for spec in step_specs(step):
                seen.setdefault(spec.name, spec)
    return list(seen.values())
```

- [ ] **Step 4: Write `graph.py`**

```python
"""Builds the LangGraph state machine from the route table; the only module that knows LangGraph."""

from typing import Callable, Dict, List

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.pipeline.node import make_agent_node, supervisor_node
from app.agents.pipeline.routes import ROUTES, Step, all_specs, step_specs
from app.agents.pipeline.run_state import ResearchRunState


def first_step(route: str, routes: Dict[str, List[Step]] = ROUTES) -> List[str]:
    steps = routes.get(route)
    return [spec.name for spec in step_specs(steps[0])] if steps else [END]


def next_step(agent_name: str, route: str, routes: Dict[str, List[Step]] = ROUTES) -> List[str]:
    steps = routes.get(route) or []
    for index, step in enumerate(steps):
        if agent_name in [spec.name for spec in step_specs(step)]:
            return [spec.name for spec in step_specs(steps[index + 1])] if index + 1 < len(steps) else [END]
    return [END]


def _after_supervisor(routes: Dict[str, List[Step]]) -> Callable[[ResearchRunState], List[str]]:
    def after_supervisor(state: ResearchRunState) -> List[str]:
        turn = state["run"]
        return first_step(turn.route, routes) if turn.run_context is not None else [END]

    return after_supervisor


def _after_agent(agent_name: str, routes: Dict[str, List[Step]]) -> Callable[[ResearchRunState], List[str]]:
    def after_agent(state: ResearchRunState) -> List[str]:
        return next_step(agent_name, state["run"].route, routes)

    after_agent.__name__ = f"after_{agent_name}"
    return after_agent


def build_graph(routes: Dict[str, List[Step]] = ROUTES) -> CompiledStateGraph:
    builder = StateGraph(ResearchRunState)
    builder.add_node("supervisor", supervisor_node)
    specs = all_specs(routes)
    for spec in specs:
        builder.add_node(spec.name, make_agent_node(spec))

    builder.add_edge(START, "supervisor")
    starts = {name for route in routes for name in first_step(route, routes)} | {END}
    builder.add_conditional_edges("supervisor", _after_supervisor(routes), sorted(starts))
    for spec in specs:
        successors = {name for route in routes for name in next_step(spec.name, route, routes)}
        builder.add_conditional_edges(spec.name, _after_agent(spec.name, routes), sorted(successors))
    return builder.compile()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `docker exec finance_agent pytest tests/pipeline -q` → all passed.

- [ ] **Step 6: Commit**

```bash
git add finance_agent_backend/app/agents/pipeline/routes.py finance_agent_backend/app/agents/pipeline/graph.py finance_agent_backend/tests/pipeline/test_routes.py finance_agent_backend/tests/pipeline/test_graph.py
git commit -m "feat(pipeline): route table and graph builder"
```

### Task 10: Thin orchestrator; switch over and delete the old wiring

**Files:**
- Create: `finance_agent_backend/app/agents/pipeline/orchestrator.py`
- Modify: `finance_agent_backend/app/agents/__init__.py`
- Delete: `finance_agent_backend/app/agents/orchestrator.py`, `finance_agent_backend/app/agents/state.py`
- Modify: `finance_agent_backend/tests/test_full_pipeline.py`, `tests/test_agent_financial_auditor.py`, `tests/test_agent_forecasting_analyst.py`, `tests/test_agent_lead_synthesizer.py`, `tests/test_agent_risk_analyst.py`, `tests/test_agent_supervisor.py`, `tests/test_report_citations.py` (schema imports from agent doors)
- Modify: `finance_agent_backend/tests/golden/harness.py` (`extract_outputs`), `finance_agent_backend/tests/golden/offline.py` (`CLOCK_TARGETS`)
- Test: `finance_agent_backend/tests/pipeline/test_orchestrator.py`
- Docs: `AGENTS.md` §2.3, `finance_agent_backend/CLAUDE.md` (Layout, Conventions), `docs/superpowers/specs/2026-10-28-codebase-top-down-structure-design.md` §4.3 / §4.5

**Interfaces:**
- Consumes: `build_graph`, `all_specs` (Task 9); `Conversation` (Task 3); `supervisor_milestone` (Task 5); `Final10KResearchReport` from the synthesizer door.
- Produces: `MultiAgentOrchestrator` (registered `multi_agent`, `orchestrator`) with unchanged public methods `arun(user_query, ticker=None, fiscal_year=None, messages=None, session_state=None, callbacks=None) -> ResearchRunState`, `astream_run(...) -> AsyncIterator[dict]`, `run(messages, session_state=None, callbacks=None) -> AgentOutput`; `app.agents.build_graph`.

- [ ] **Step 1: Write the failing orchestrator test**

`finance_agent_backend/tests/pipeline/test_orchestrator.py`:

```python
import asyncio
import unittest
from unittest.mock import patch

from app.agents import MultiAgentOrchestrator
from app.agents.specialized.financial_auditor import FinancialAuditorAgent
from app.agents.specialized.supervisor import RoutingPlan, TurnResolution
from tests.pipeline.fixtures import make_run_context


def _audit_turn(**ctx_changes) -> TurnResolution:
    ctx = make_run_context(route="financial_audit_only", **ctx_changes)
    plan = RoutingPlan(ticker="AAPL", fiscal_year=2025, query_type="financial_audit_only",
                       available_fiscal_years=[2025, 2024])
    return TurnResolution(route="financial_audit_only", plan=plan, run_context=ctx, conversational_response=None,
                          session_update={"active_ticker": "AAPL"}, parameter_notes=())


async def _events(**kwargs):
    return [event async for event in MultiAgentOrchestrator().astream_run(user_query="audit AAPL", **kwargs)]


class TestOrchestratorStreaming(unittest.TestCase):
    def test_every_research_agent_failing_ends_with_an_error_event(self):
        with patch("app.agents.pipeline.node.resolve_turn", return_value=_audit_turn()), \
                patch.object(FinancialAuditorAgent, "audit", side_effect=RuntimeError("boom")):
            events = asyncio.run(_events())
        self.assertEqual(events[0], {"type": "status", "node": "supervisor",
                                     "message": "Analyzing research inquiry & context..."})
        self.assertIn({"type": "status", "node": "financial_auditor", "message": "Financial statement audit unavailable"}, events)
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(
            events[-1]["message"],
            "Research pipeline failed: No research output could be produced for AAPL FY2025. "
            "Financial Auditor failed (RuntimeError: boom); its section is omitted from this report.",
        )

    def test_conversational_turn_returns_the_reply(self):
        turn = TurnResolution(route="conversational", plan=None, run_context=None,
                              conversational_response="A DCF discounts cash flows.",
                              session_update={"active_ticker": "AAPL"}, parameter_notes=())
        with patch("app.agents.pipeline.node.resolve_turn", return_value=turn):
            events = asyncio.run(_events())
        self.assertEqual(events[-1], {"type": "result", "response": "A DCF discounts cash flows.", "sources": [],
                                      "updated_session_state": {"active_ticker": "AAPL"}})

    def test_sync_run_raises_when_no_report_exists(self):
        with patch("app.agents.pipeline.node.resolve_turn", return_value=_audit_turn()), \
                patch.object(FinancialAuditorAgent, "audit", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                MultiAgentOrchestrator().run([{"role": "user", "content": "audit AAPL"}])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec finance_agent pytest tests/pipeline/test_orchestrator.py -q`
Expected: FAIL. The old orchestrator ignores `app.agents.pipeline.node.resolve_turn`, so the patch has no effect and the supervisor calls the (unfaked) LLM path; the assertions on the event sequence fail.

- [ ] **Step 3: Write `pipeline/orchestrator.py`**

```python
"""Research run entry point: build the turn, run the graph, stream milestones, shape the result."""

import asyncio
import concurrent.futures
import logging
from datetime import date
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from langgraph.graph.state import CompiledStateGraph

from app.agents.base import AgentOutput, BaseAgent
from app.agents.pipeline.contract import Conversation, StatusLine
from app.agents.pipeline.graph import build_graph
from app.agents.pipeline.routes import all_specs
from app.agents.pipeline.run_state import ResearchRunState
from app.agents.registry import AgentRegistry
from app.agents.specialized.lead_synthesizer import Final10KResearchReport
from app.agents.specialized.supervisor import supervisor_milestone

logger = logging.getLogger("finance_agent.agents.orchestrator")

SPECS_BY_NAME = {spec.name: spec for spec in all_specs()}
START_MESSAGE = "Analyzing research inquiry & context..."
NO_REPORT_MESSAGE = "The research pipeline finished without a report."


@AgentRegistry.register("multi_agent")
@AgentRegistry.register("orchestrator")
class MultiAgentOrchestrator(BaseAgent):
    """Runs one chat turn through the research pipeline."""

    def __init__(self, recursion_limit: int = 50):
        self.recursion_limit = recursion_limit
        self._cached_graph: Optional[CompiledStateGraph] = None

    def get_graph(self) -> CompiledStateGraph:
        if self._cached_graph is None:
            self._cached_graph = build_graph()
        return self._cached_graph

    @staticmethod
    def _initial_state(user_query: str, ticker: Optional[str], fiscal_year: Optional[int],
                       messages: Optional[List[Dict[str, str]]], session_state: Optional[Dict[str, Any]],
                       callbacks: Optional[List[Any]]) -> ResearchRunState:
        conversation = Conversation(
            user_query=user_query,
            messages=messages,
            session_state=session_state,
            ticker=ticker.upper() if ticker else "",
            fiscal_year=fiscal_year or 0,
            run_date=date.today(),
            callbacks=callbacks,
        )
        return {"conversation": conversation}

    def _config(self, callbacks: Optional[List[Any]]) -> Dict[str, Any]:
        config: Dict[str, Any] = {"recursion_limit": self.recursion_limit}
        if callbacks:
            config["callbacks"] = callbacks
        return config

    async def arun(self, user_query: str, ticker: Optional[str] = None, fiscal_year: Optional[int] = None,
                   messages: Optional[List[Dict[str, str]]] = None, session_state: Optional[Dict[str, Any]] = None,
                   callbacks: Optional[List[Any]] = None) -> ResearchRunState:
        """Runs the graph and returns the final run state."""
        initial = self._initial_state(user_query, ticker, fiscal_year, messages, session_state, callbacks)
        return await self.get_graph().ainvoke(initial, config=self._config(callbacks))

    async def astream_run(self, user_query: str, ticker: Optional[str] = None, fiscal_year: Optional[int] = None,
                          messages: Optional[List[Dict[str, str]]] = None,
                          session_state: Optional[Dict[str, Any]] = None,
                          callbacks: Optional[List[Any]] = None) -> AsyncIterator[Dict[str, Any]]:
        """Runs the graph, streaming one status event per finished node, then the result or an error."""
        initial = self._initial_state(user_query, ticker, fiscal_year, messages, session_state, callbacks)
        yield {"type": "status", "node": "supervisor", "message": START_MESSAGE}
        final_state: Dict[str, Any] = {}
        try:
            async for mode, chunk in self.get_graph().astream(
                initial, config=self._config(callbacks), stream_mode=["updates", "values"]
            ):
                if mode == "values":
                    final_state = chunk
                    continue
                for node_name, update in chunk.items():
                    message, details = self._milestone(node_name, update)
                    if message:
                        event: Dict[str, Any] = {"type": "status", "node": node_name, "message": message}
                        if details:
                            event["details"] = details
                        yield event
                    for note in update.get("notes") or []:
                        yield {"type": "status", "node": node_name, "message": note.message,
                               "details": {"severity": "warning"}}
        except Exception as e:
            # Stream an error event instead of tearing the SSE connection down; the chat service persists it.
            logger.exception("astream_run: research pipeline failed")
            yield {"type": "error", "message": f"Research pipeline failed: {e}"}
            return
        yield self._result_event(final_state)

    @staticmethod
    def _milestone(node_name: str, update: Dict[str, Any]) -> StatusLine:
        if node_name == "supervisor":
            return supervisor_milestone(update.get("run"))
        spec = SPECS_BY_NAME.get(node_name)
        if spec is None:
            return None, None
        output = update.get("final") if spec.writes_final else (update.get("outputs") or {}).get(spec.output_key)
        return spec.milestone(output)

    @staticmethod
    def _result_event(state: Dict[str, Any]) -> Dict[str, Any]:
        final = state.get("final")
        turn = state.get("run")
        session_update = turn.session_update if turn is not None else None
        if isinstance(final, str) and final:
            return {"type": "result", "response": final, "sources": [], "updated_session_state": session_update}
        if isinstance(final, Final10KResearchReport):
            return {
                "type": "result",
                "response": final.full_markdown_report,
                "sources": [s.model_dump() for s in final.sources],
                "final_report": final.model_dump(),
                "updated_session_state": session_update,
            }
        return {"type": "error", "message": NO_REPORT_MESSAGE}

    def run(self, messages: List[Dict[str, str]], session_state: Optional[Dict[str, Any]] = None,
            callbacks: Optional[List[Any]] = None) -> AgentOutput:
        """Synchronous BaseAgent interface; a run without a report raises."""
        query = messages[-1]["content"] if messages else ""
        coroutine = self.arun(user_query=query, messages=messages, session_state=session_state, callbacks=callbacks)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                state = pool.submit(asyncio.run, coroutine).result()
        else:
            state = asyncio.run(coroutine)
        event = self._result_event(state)
        if event["type"] == "error":
            raise RuntimeError(event["message"])
        return AgentOutput(content=event["response"], sources=event["sources"],
                           updated_session_state=event["updated_session_state"])
```

Note: `Tuple` is imported for readers of `StatusLine`; drop it if your linter flags it unused.

- [ ] **Step 4: Switch the package over and delete the old wiring**

Replace `finance_agent_backend/app/agents/__init__.py` with:

```python
from app.agents.base import AgentOutput, BaseAgent
from app.agents.registry import AgentRegistry
import app.agents.specialized  # Triggers self-registration of specialized agents
from app.agents.pipeline.graph import build_graph
from app.agents.pipeline.orchestrator import MultiAgentOrchestrator

__all__ = [
    "BaseAgent",
    "AgentOutput",
    "AgentRegistry",
    "MultiAgentOrchestrator",
    "build_graph",
]
```

Then delete the old files:

```bash
git rm finance_agent_backend/app/agents/orchestrator.py finance_agent_backend/app/agents/state.py
```

Run `grep -rn "app.agents.orchestrator\|app.agents.state\|build_equity_research_graph\|EquityResearchState" finance_agent_backend/app finance_agent_backend/tests` and fix every hit (Steps 5–6). The only allowed remaining hits are docstrings that say "orchestrator" in prose.

- [ ] **Step 5: Point tests at the agent doors**

Replace each `from app.agents.state import …` with imports from the agent that owns the schema:

| Test file | New import(s) |
|---|---|
| `tests/test_agent_financial_auditor.py` | `from app.agents.specialized.financial_auditor import FinancialAuditOutput` (merge into the existing door import) |
| `tests/test_agent_forecasting_analyst.py` | `from app.agents.specialized.valuation_specialist import DCFValuationOutput` |
| `tests/test_agent_risk_analyst.py` | `from app.agents.specialized.risk_analyst import RiskAuditOutput, RiskEvidence, RiskItem` |
| `tests/test_agent_supervisor.py` | `from app.agents.specialized.supervisor import RoutingPlan` |
| `tests/test_report_citations.py` | `from app.agents.specialized.risk_analyst import RiskEvidence` |
| `tests/test_agent_lead_synthesizer.py` | split the block by owner: `business_strategist` → `BusinessMoatOutput, SegmentDetail`; `valuation_specialist` → `DCFValuationOutput`; `lead_synthesizer` → `Final10KResearchReport, ThreePillarThesis`; `financial_auditor` → `FinancialAuditOutput`; `forecasting_analyst` → `ForecastOutput`; `risk_analyst` → `RiskAuditOutput, RiskDataQuality, RiskDisclosureDiff, RiskEvidence, RiskItem` (these 12 names are the whole original block) |

- [ ] **Step 6: Update `tests/test_full_pipeline.py`**

1. Replace the import `from app.agents import AgentRegistry, MultiAgentOrchestrator, build_equity_research_graph` with `from app.agents import AgentRegistry, MultiAgentOrchestrator, build_graph`, and `graph = build_equity_research_graph()` with `graph = build_graph()`.
2. Replace the `from app.agents.state import (...)` block with door imports (same owner mapping as Step 5); drop `EquityResearchState`.
3. In `setUpClass`, replace the two orchestrator patches with:

```python
            patch("app.agents.specialized.valuation_specialist.context.fetch_market_context", return_value=cls.mock_market),
            patch("app.agents.specialized.valuation_specialist.agent_valuation_specialist.balance_sheet_date",
                  return_value="2025-12-31"),
```

4. Add this helper method to `TestMilestone6FullPipeline`:

```python
    @staticmethod
    def _view(state) -> dict:
        """Flat view of the run state for assertions."""
        turn, outputs = state["run"], state.get("outputs") or {}
        ctx = turn.run_context
        return {
            "ticker": ctx.ticker if ctx else None,
            "fiscal_year": ctx.fiscal_year if ctx else None,
            "query_type": turn.route,
            **outputs,
            "final_report": state.get("final"),
        }
```

5. In each of the three places that read `state: EquityResearchState = loop.run_until_complete(`, change the line to `state = self._view(loop.run_until_complete(` and add one closing `)` to the matching `)` that closes `run_until_complete(...)`. The existing `state.get(...)` / `state[...]` assertions then work unchanged.

- [ ] **Step 7: Update the golden harness for the new state shape**

In `tests/golden/offline.py`, set:

```python
CLOCK_TARGETS = ("app.agents.pipeline.orchestrator.date", "app.agents.tools.valuation_model.date")
```

In `tests/golden/harness.py`, replace `extract_outputs` with:

```python
def extract_outputs(state: Dict[str, Any]) -> Dict[str, Any]:
    """Agent outputs and turn results from the final run state.

    This is the only place that knows the state's shape; update it when the shape changes, never the snapshots.
    """
    outputs = state.get("outputs") or {}
    final = state.get("final")
    turn = state.get("run")
    return {
        **{key: _dump(outputs.get(key)) for key in OUTPUT_KEYS},
        "final_report": _dump(final) if isinstance(final, BaseModel) else None,
        "conversational_response": final if isinstance(final, str) else None,
        "updated_session_state": turn.session_update if turn is not None else None,
        "pipeline_notes": [note.message for note in state.get("notes") or []],
    }
```

- [ ] **Step 8: Run everything; the golden snapshots must not change**

Run: `docker exec finance_agent pytest tests/pipeline/test_orchestrator.py -q` → 3 passed.
Run: `docker exec finance_agent pytest tests/golden -q` → all passed, **with no snapshot regenerated**. If a scenario fails, the new wiring changed behaviour. Read the `first_difference` path, fix the wiring (not the snapshot), and rerun.
Run: `docker exec finance_agent pytest tests/ -q` → all passed.
Run: `git status --short finance_agent_backend/tests/golden/snapshots` → empty.

- [ ] **Step 9: Update the docs**

1. `AGENTS.md` §2.3: rename to "Agent Execution Graph (`app/agents/pipeline/`)". Explain that routes live in `pipeline/routes.py` (`ROUTES`, a nested list = parallel); every agent exports `AGENT = AgentSpec(...)` from its `__init__` with `needs` / `uses` / `needs_any`; `pipeline/node.py` runs every agent with skip / failure notes (`RunNote`); the Supervisor turns a chat turn into an immutable `RunContext` (incl. terminal growth and valuation date); the Synthesizer writes `final`. Keep the Failure isolation bullet, reworded for `RunNote`.
2. `finance_agent_backend/CLAUDE.md` Layout: replace the `orchestrator.py` and `state.py` lines with:

   ```
   pipeline/          contract (AgentSpec, RunContext, Upstream, RunNote), run_state, node, routes (ROUTES), graph, orchestrator
   ```

   Conventions, "Adding an agent": add "export `AGENT = AgentSpec(...)` from the agent's `__init__` and add it to `ROUTES` in `pipeline/routes.py`". Drop "re-export its schemas from `agents/state.py`".
3. `docs/superpowers/specs/2026-10-28-codebase-top-down-structure-design.md`: in §4.3, add `route` to the `RunContext` field list. In §4.5, drop the `error` row (nothing ever wrote it; the orchestrator's no-report message is a constant) and note that `conversation` holds a `Conversation`. In §4.6, note that the `AgentRegistry` trim happens in PR 11.

- [ ] **Step 10: Commit**

```bash
git add -A finance_agent_backend/app/agents finance_agent_backend/tests AGENTS.md docs/superpowers/specs/2026-10-28-codebase-top-down-structure-design.md   # + finance_agent_backend/CLAUDE.md once tracked
git commit -m "refactor(pipeline): route-table graph and thin orchestrator replace the hand-wired orchestrator"
```

- [ ] **Step 11: Open the PR**

PR description: what moved (L0 into `agents/pipeline/`, market data into valuation `context.py`, g and valuation date into `RunContext`); the gate result (full suite green, golden snapshots unchanged, `git status` clean); and the quirks kept on purpose (see "Known quirks Phase 1 preserves on purpose").
