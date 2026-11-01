"""Financial Forecasting Analyst Agent.

Deterministic-first operating forecast:

1. Code builds the audited baseline from FinancialAuditOutput (normalized margin, median tax rate,
   multi-year CapEx intensity, D&A, working-capital and SBC intensity) and screens the Item 7 MD&A
   excerpts for forward-looking and quantified guidance.
2. When MD&A carries real guidance, the LLM proposes revenue growth and margin drivers through the
   per-run `propose_forecast_drivers_tool`, which clamps them to code-owned bounds, and writes the
   rationales. Otherwise growth fades linearly from the historical trend to terminal growth with no
   LLM call.
3. Code builds the schedule, bull/base/bear scenarios and reinvestment diagnostics from the last
   accepted drivers, so no LLM-typed number reaches the output.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from langchain_core.tools import StructuredTool
from pydantic import ValidationError

from app.agents.base import StructuredAgent
from app.agents.pipeline.contract import RunContext, StatusLine, Upstream
from app.agents.registry import AgentRegistry
from app.agents.shared.filing.narrative_search import NarrativeChunkResult, retrieve_10k_narrative_tool
from app.agents.shared.filing.table_search import retrieve_10k_tables_tool
from app.agents.shared.prompts import render_prompt
from app.agents.specialized.business_strategist.state_business_strategist import BusinessMoatOutput
from app.agents.specialized.financial_auditor.state_financial_auditor import FinancialAuditOutput
from app.agents.specialized.forecasting_analyst.assemble import (
    assemble_forecast,
    fallback_rationales,
    mdna_fade_reason,
    resolve_forecast_citations,
)
from app.agents.specialized.forecasting_analyst.context import fetch_mdna_chunks
from app.agents.specialized.forecasting_analyst.modules.audited_baseline import ForecastBaseline, build_forecast_baseline, render_history_table
from app.agents.specialized.forecasting_analyst.modules.forecast_consistency import compute_diagnostics
from app.agents.specialized.forecasting_analyst.modules.free_cash_flow_schedule import run_schedule
from app.agents.specialized.forecasting_analyst.modules.management_guidance import MdnaAssessment, assess_mdna_guidance, margin_evidence_chunk_ids
from app.agents.specialized.forecasting_analyst.modules.reinvestment_path import apply_reinvestment_path
from app.agents.specialized.forecasting_analyst.modules.revenue_and_margin_drivers import (
    DriverAdjustment,
    ForecastDriverProposal,
    ForecastDrivers,
    extend_fade,
    historical_fade_drivers,
    screen_forecast_drivers,
    unsupported_margin_corridor,
)
from app.agents.specialized.forecasting_analyst.modules.scenarios import build_scenarios
from app.agents.specialized.forecasting_analyst.state_forecasting_analyst import ForecastBuild, ForecastOutput
from app.agents.specialized.risk_analyst.state_risk_analyst import RiskAuditOutput

logger = logging.getLogger("finance_agent.agents.forecasting_analyst")

PROPOSE_TOOL_NAME = "propose_forecast_drivers_tool"
EXCERPT_CHARS = 700
MAX_PROMPT_RISKS = 3
RISK_TRANSMISSIONS = {"Revenue", "Gross Margin", "Operating Expenses"}
RATIONALE_KEYS = ("growth_rationale", "margin_rationale")
NO_DRIVERS_REASON = "No screened LLM drivers were accepted, so growth fades linearly from the historical trend."
NO_ACCEPTED_DRIVERS_WARNING = (
    "The LLM did not submit accepted drivers; growth fades from the historical trend to terminal growth."
)
LLM_STEP_FAILED_WARNING = "The LLM step failed before finishing; the forecast was rebuilt in code from what it had submitted."

PROPOSE_TOOL_DESCRIPTION = """Submit 5-year revenue growth and operating margin drivers for screening and a schedule preview.

All values are WHOLE percent (6.5 means 6.5%). Code clamps year-1 growth to a band around the latest
audited growth and limits margin moves to 3 percentage points per year. A margin path more than 1 point
from the base margin must either follow the audited margin trend shown in the task or cite, in
margin_citation_chunk_id, an MD&A excerpt with forward-looking margin or cost language; otherwise it is
clamped. CapEx or working-capital overrides need a chunk_id from the MD&A excerpts. CapEx then fades in
code to a steady state consistent with terminal growth.

Args:
    revenue_growth_pct: exactly 5 YoY revenue growth rates, year 1 first.
    operating_margin_pct: exactly 5 EBIT margins, year 1 first.
    margin_citation_chunk_id: chunk_id supporting a margin path that departs from the audited trend.
    capex_pct_of_revenue: optional year-1 CapEx intensity override (whole percent of revenue).
    capex_citation_chunk_id: chunk_id supporting the CapEx override.
    nwc_pct_of_incremental_revenue: optional working-capital intensity override (whole percent of incremental revenue).
    nwc_citation_chunk_id: chunk_id supporting the working-capital override.
    evidence_chunk_ids: chunk_ids supporting the growth and margin curves.

Returns status "accepted" (applied drivers, clamps, schedule preview, diagnostics warnings) or
"rejected" (errors to fix before calling again).
"""


def build_forecast(
    baseline: ForecastBaseline,
    drivers: ForecastDrivers,
    terminal_growth: float,
    risk_profile: Optional[str] = None,
) -> ForecastBuild:
    """The forecast checklist: every number after the drivers, built in code from the screened (or faded) path."""
    growth, margins, extended = extend_fade(drivers.revenue_growth, drivers.operating_margin, terminal_growth)
    applied, reinvestment_warnings = apply_reinvestment_path(
        baseline, drivers.model_copy(update={"revenue_growth": growth, "operating_margin": margins}), terminal_growth
    )
    schedule = run_schedule(baseline, applied)
    diagnostics = compute_diagnostics(baseline, applied, schedule, terminal_growth, extended)
    diagnostics.warnings.extend(reinvestment_warnings)
    return ForecastBuild(
        drivers=applied,
        schedule=schedule,
        scenarios=build_scenarios(baseline, applied, schedule, risk_profile, terminal_growth),
        diagnostics=diagnostics,
    )


def _risk_lines(risk_audit: Optional[RiskAuditOutput]) -> List[str]:
    if risk_audit is None:
        return []
    relevant = [
        r for r in risk_audit.identified_risks
        if r.severity == "Severe" or RISK_TRANSMISSIONS & set(r.financial_transmission)
    ]
    return [
        f"{r.risk_id} [{r.severity}] {r.risk_title} (hits: {', '.join(r.financial_transmission)})"
        for r in relevant[:MAX_PROMPT_RISKS]
    ]


@AgentRegistry.register("forecasting_analyst")
class ForecastingAnalystAgent(StructuredAgent[ForecastOutput]):
    """Deterministic-first operating forecaster with LLM-proposed, code-screened drivers."""

    prompt_name = "forecasting_analyst"
    tools = [retrieve_10k_narrative_tool, retrieve_10k_tables_tool]
    output_schema = ForecastOutput
    default_recursion_limit = 20

    def __init__(self, model_name: str = "openai:gpt-4o-mini", recursion_limit: Optional[int] = None):
        super().__init__(model_name=model_name, recursion_limit=recursion_limit)
        self._run_tools: Optional[List[Any]] = None
        self._last_accepted: Optional[Tuple[ForecastDrivers, List[DriverAdjustment], List[str]]] = None

    def get_tools(self) -> List[Any]:
        return self._run_tools if self._run_tools is not None else list(self.tools)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def forecast(
        self,
        ticker: str,
        fiscal_year: int,
        financial_audit: FinancialAuditOutput,
        terminal_growth: float,
        business_moat: Optional[BusinessMoatOutput] = None,
        risk_audit: Optional[RiskAuditOutput] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> ForecastOutput:
        ticker = ticker.upper()
        baseline = build_forecast_baseline(financial_audit)
        chunks = fetch_mdna_chunks(ticker, fiscal_year)
        mdna = assess_mdna_guidance(chunks)
        risk_profile = risk_audit.overall_risk_profile if risk_audit is not None else None

        if not mdna.is_informative:
            logger.info(
                f"Forecast {ticker} FY{fiscal_year}: {mdna.forward_looking_count} forward-looking MD&A "
                f"sentence(s); using historical fade without the LLM."
            )
            drivers = historical_fade_drivers(baseline, terminal_growth)
            return assemble_forecast(
                ticker=ticker, fiscal_year=fiscal_year, baseline=baseline,
                build=build_forecast(baseline, drivers, terminal_growth, risk_profile),
                guidance_source="historical_fade", mdna=mdna, terminal_growth=terminal_growth,
                rationales=fallback_rationales(baseline, drivers, terminal_growth, mdna_fade_reason(mdna)),
            )

        self._configure_tools(
            baseline, {c.chunk_id for c in chunks}, margin_evidence_chunk_ids(chunks), terminal_growth, risk_profile
        )
        query = self._render_query(ticker, fiscal_year, baseline, mdna, chunks, business_moat, risk_audit, terminal_growth)
        context = {
            "baseline": baseline,
            "mdna": mdna,
            "chunks": chunks,
            "terminal_growth": terminal_growth,
            "risk_profile": risk_profile,
        }
        try:
            return self.execute_structured(
                query, ticker=ticker, fiscal_year=fiscal_year, callbacks=callbacks, forecast_context=context,
            )
        except Exception as e:
            # A failed LLM step (recursion limit, API error, unparseable output) must not fail the pipeline:
            # rebuild from the last drivers the tool accepted, or fade from history.
            logger.warning(f"Forecast {ticker} FY{fiscal_year}: LLM step failed ({e!r}); rebuilding in code.")
            return self._build_from_accepted(
                ticker, fiscal_year, context, self._last_accepted, rationales=None, llm_citations=None,
                messages=[], extra_warnings=[LLM_STEP_FAILED_WARNING],
            )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _render_query(
        self,
        ticker: str,
        fiscal_year: int,
        baseline: ForecastBaseline,
        mdna: MdnaAssessment,
        chunks: Sequence[NarrativeChunkResult],
        business_moat: Optional[BusinessMoatOutput],
        risk_audit: Optional[RiskAuditOutput],
        terminal_growth: float,
    ) -> str:
        return render_prompt(
            "forecasting_analyst", "prompt_forecasting_analyst_query.j2",
            ticker=ticker,
            fiscal_year=fiscal_year,
            baseline=baseline,
            history_table=render_history_table(baseline.historical_rows),
            margin_corridor=unsupported_margin_corridor(baseline),
            terminal_growth=terminal_growth,
            moat=business_moat,
            risk_profile=risk_audit.overall_risk_profile if risk_audit is not None else None,
            risk_lines=_risk_lines(risk_audit),
            mdna=mdna,
            excerpts=[
                {"chunk_id": c.chunk_id, "breadcrumb": c.breadcrumb, "text": " ".join(c.content.split())[:EXCERPT_CHARS]}
                for c in chunks
            ],
        )

    def _configure_tools(
        self,
        baseline: ForecastBaseline,
        allowed_chunk_ids: Set[str],
        margin_chunk_ids: Set[str],
        terminal_growth: float,
        risk_profile: Optional[str],
    ) -> None:
        def propose_forecast_drivers(
            revenue_growth_pct: List[float],
            operating_margin_pct: List[float],
            margin_citation_chunk_id: Optional[str] = None,
            capex_pct_of_revenue: Optional[float] = None,
            capex_citation_chunk_id: Optional[str] = None,
            nwc_pct_of_incremental_revenue: Optional[float] = None,
            nwc_citation_chunk_id: Optional[str] = None,
            evidence_chunk_ids: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            try:
                proposal = ForecastDriverProposal(
                    revenue_growth_pct=revenue_growth_pct,
                    operating_margin_pct=operating_margin_pct,
                    margin_citation_chunk_id=margin_citation_chunk_id,
                    capex_pct_of_revenue=capex_pct_of_revenue,
                    capex_citation_chunk_id=capex_citation_chunk_id,
                    nwc_pct_of_incremental_revenue=nwc_pct_of_incremental_revenue,
                    nwc_citation_chunk_id=nwc_citation_chunk_id,
                    evidence_chunk_ids=evidence_chunk_ids or [],
                )
            except ValidationError as e:
                return {"status": "rejected", "errors": [f"Invalid driver inputs: {e.errors()[0]['msg']}"]}
            screen = screen_forecast_drivers(
                proposal, baseline, allowed_chunk_ids, terminal_growth, margin_evidence_chunk_ids=margin_chunk_ids,
            )
            if not screen.accepted:
                return {"status": "rejected", "errors": screen.errors}
            try:
                build = build_forecast(baseline, screen.drivers, terminal_growth, risk_profile)
            except ValueError as e:
                return {"status": "rejected", "errors": [str(e)]}
            self._last_accepted = (screen.drivers, list(screen.adjustments), list(screen.evidence_chunk_ids))
            return {
                "status": "accepted",
                "applied_drivers": screen.drivers.model_dump(),
                "adjustments": [a.model_dump() for a in screen.adjustments],
                "evidence_chunk_ids": screen.evidence_chunk_ids,
                "margin_evidence_accepted": screen.margin_evidence_accepted,
                "forecast_horizon_years": build.schedule["forecast_horizon_years"],
                "revenue_cagr_pct": build.schedule["revenue_cagr_pct"],
                "preview": build.preview_rows(),
                "diagnostics_warnings": build.diagnostics.warnings,
            }

        self._last_accepted = None
        self._run_tools = list(self.tools) + [
            StructuredTool.from_function(
                func=propose_forecast_drivers, name=PROPOSE_TOOL_NAME, description=PROPOSE_TOOL_DESCRIPTION,
            )
        ]
        self._cached_agent = None

    # ------------------------------------------------------------------
    # Output lifecycle hooks
    # ------------------------------------------------------------------
    def _pre_validate_data(
        self,
        data: Dict[str, Any],
        result: Dict[str, Any],
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        forecast_context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if forecast_context is None:
            return data
        messages = result.get("messages", [])
        output = self._build_from_accepted(
            ticker, fiscal_year, forecast_context, self._last_accepted_proposal(messages),
            rationales={
                k: data[k].strip() for k in RATIONALE_KEYS if isinstance(data.get(k), str) and data[k].strip()
            },
            llm_citations=data.get("citations"),
            messages=messages,
        )
        return output.model_dump()

    def _build_from_accepted(
        self,
        ticker: str,
        fiscal_year: int,
        ctx: Dict[str, Any],
        accepted: Optional[Tuple[ForecastDrivers, List[DriverAdjustment], List[str]]],
        rationales: Optional[Dict[str, str]],
        llm_citations: Any,
        messages: List[Any],
        extra_warnings: Sequence[str] = (),
    ) -> ForecastOutput:
        """Assembles the output from the accepted drivers, or the historical fade when none were accepted."""
        baseline, terminal_growth = ctx["baseline"], ctx["terminal_growth"]
        common = dict(
            ticker=ticker, fiscal_year=fiscal_year, baseline=baseline, mdna=ctx["mdna"],
            terminal_growth=terminal_growth,
        )
        if accepted is None:
            logger.warning(f"{ticker} FY{fiscal_year}: {NO_ACCEPTED_DRIVERS_WARNING}")
            drivers = historical_fade_drivers(baseline, terminal_growth)
            return assemble_forecast(
                **common, build=build_forecast(baseline, drivers, terminal_growth, ctx["risk_profile"]),
                guidance_source="historical_fade",
                rationales=fallback_rationales(baseline, drivers, terminal_growth, NO_DRIVERS_REASON),
                extra_warnings=[*extra_warnings, NO_ACCEPTED_DRIVERS_WARNING],
            )
        drivers, adjustments, evidence_ids = accepted
        return assemble_forecast(
            **common, build=build_forecast(baseline, drivers, terminal_growth, ctx["risk_profile"]),
            guidance_source="mdna_informed", rationales=rationales, adjustments=adjustments,
            citations=resolve_forecast_citations(
                llm_citations, ctx["chunks"], evidence_ids, self._extract_citations_from_messages(messages),
            ),
            extra_warnings=extra_warnings,
        )

    @staticmethod
    def _last_accepted_proposal(
        messages: List[Any],
    ) -> Optional[Tuple[ForecastDrivers, List[DriverAdjustment], List[str]]]:
        for msg in reversed(messages):
            if (getattr(msg, "name", "") or "") != PROPOSE_TOOL_NAME:
                continue
            raw = getattr(msg, "content", "")
            try:
                payload = raw if isinstance(raw, dict) else json.loads(raw)
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict) and payload.get("status") == "accepted":
                return (
                    ForecastDrivers.model_validate(payload["applied_drivers"]),
                    [DriverAdjustment.model_validate(a) for a in payload.get("adjustments", [])],
                    list(payload.get("evidence_chunk_ids", [])),
                )
        return None


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
