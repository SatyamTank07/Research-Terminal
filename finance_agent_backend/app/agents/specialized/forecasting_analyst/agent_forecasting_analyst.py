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

from app.agents.base import StructuredAgent
from app.agents.registry import AgentRegistry
from app.agents.specialized.prompts import render_prompt
from app.agents.specialized.business_strategist.state_business_strategist import BusinessMoatOutput
from app.agents.specialized.financial_auditor.state_financial_auditor import FinancialAuditOutput
from app.agents.specialized.forecasting_analyst.state_forecasting_analyst import (
    ForecastOutput,
    GuidanceSource,
)
from app.agents.specialized.risk_analyst.state_risk_analyst import RiskAuditOutput
from app.agents.tools.forecast_baseline import (
    ForecastBaseline,
    build_forecast_baseline,
    render_history_table,
)
from app.agents.tools.forecast_drivers import (
    DEFAULT_TERMINAL_GROWTH,
    DriverAdjustment,
    ForecastDriverProposal,
    ForecastDrivers,
    historical_fade_drivers,
    screen_forecast_drivers,
)
from app.agents.tools.forecast_model import build_forecast
from app.agents.tools.mdna_guidance import MdnaAssessment, assess_mdna_guidance
from app.agents.tools.rag_narrative_tools import (
    NarrativeChunkResult,
    retrieve_10k_narrative,
    retrieve_10k_narrative_tool,
)
from app.agents.tools.rag_table_tools import retrieve_10k_tables_tool

logger = logging.getLogger("finance_agent.agents.forecasting_analyst")

PROPOSE_TOOL_NAME = "propose_forecast_drivers_tool"
MDNA_QUERY = "outlook expect demand growth pricing margins costs capital expenditures guidance"
MDNA_CHUNK_LIMIT = 6
EXCERPT_CHARS = 700
MAX_PROMPT_RISKS = 3
RISK_TRANSMISSIONS = {"Revenue", "Gross Margin", "Operating Expenses"}
RATIONALE_KEYS = ("growth_rationale", "margin_rationale", "reinvestment_rationale")
NO_DRIVERS_REASON = "No screened LLM drivers were accepted, so growth fades linearly from the historical trend."
NO_ACCEPTED_DRIVERS_WARNING = (
    "The LLM did not submit accepted drivers; growth fades from the historical trend to terminal growth."
)
LLM_STEP_FAILED_WARNING = "The LLM step failed before finishing; the forecast was rebuilt in code from what it had submitted."

PROPOSE_TOOL_DESCRIPTION = """Submit 5-year revenue growth and operating margin drivers for screening and a schedule preview.

All values are WHOLE percent (6.5 means 6.5%). Code clamps year-1 growth to a band around the latest
audited growth, limits margin moves to 3 percentage points per year, and only accepts CapEx or
working-capital overrides that cite a chunk_id from the MD&A excerpts in the task.

Args:
    revenue_growth_pct: exactly 5 YoY revenue growth rates, year 1 first.
    operating_margin_pct: exactly 5 EBIT margins, year 1 first.
    capex_pct_of_revenue: optional CapEx intensity override (whole percent of revenue).
    capex_citation_chunk_id: chunk_id supporting the CapEx override.
    nwc_pct_of_incremental_revenue: optional working-capital intensity override (whole percent of incremental revenue).
    nwc_citation_chunk_id: chunk_id supporting the working-capital override.
    evidence_chunk_ids: chunk_ids supporting the growth and margin curves.

Returns status "accepted" (applied drivers, clamps, schedule preview, diagnostics warnings) or
"rejected" (errors to fix before calling again).
"""


def driver_rationales(baseline: ForecastBaseline, drivers: ForecastDrivers, terminal_growth: float) -> Dict[str, str]:
    """Factual rationale text used when the LLM did not supply one."""
    g, m = drivers.revenue_growth, drivers.operating_margin
    return {
        "growth_rationale": (
            f"Revenue growth runs from {g[0] * 100:.2f}% in year 1 to {g[-1] * 100:.2f}% in year {len(g)}, "
            f"against {terminal_growth * 100:.2f}% terminal growth."
        ),
        "margin_rationale": (
            f"Operating margin moves from the {baseline.margin_source} base of "
            f"{baseline.base_operating_margin * 100:.2f}% to {m[-1] * 100:.2f}% by year {len(m)}."
        ),
        "reinvestment_rationale": (
            f"CapEx at {drivers.capex_pct_of_revenue * 100:.2f}% of revenue ({baseline.capex_years}-year average "
            f"{baseline.capex_pct_of_revenue * 100:.2f}%); D&A at {drivers.da_pct_of_revenue * 100:.2f}% "
            f"({baseline.da_source}); working capital at {drivers.nwc_pct_of_incremental_revenue * 100:.2f}% of "
            f"incremental revenue ({baseline.nwc_source}); SBC at {drivers.sbc_pct_of_revenue * 100:.2f}% of "
            f"revenue is already expensed in EBIT, so UFCF treats it as a cash cost."
        ),
    }


def mdna_fade_reason(mdna: MdnaAssessment) -> str:
    return (
        f"Item 7 gave {mdna.forward_looking_count} forward-looking statement(s), too few to anchor a guided "
        f"forecast, so growth fades linearly from the historical trend."
    )


def fallback_rationales(
    baseline: ForecastBaseline, drivers: ForecastDrivers, terminal_growth: float, reason: str
) -> Dict[str, str]:
    texts = driver_rationales(baseline, drivers, terminal_growth)
    texts["growth_rationale"] = f"{reason} {texts['growth_rationale']}"
    return texts


def build_forecast_output(
    *,
    ticker: str,
    fiscal_year: int,
    baseline: ForecastBaseline,
    drivers: ForecastDrivers,
    guidance_source: GuidanceSource,
    mdna: MdnaAssessment,
    terminal_growth: float,
    risk_profile: Optional[str] = None,
    rationales: Optional[Dict[str, str]] = None,
    adjustments: Sequence[DriverAdjustment] = (),
    citations: Sequence[Dict[str, Any]] = (),
    extra_warnings: Sequence[str] = (),
) -> ForecastOutput:
    """Assembles ForecastOutput from code-built numbers; missing rationales get factual defaults."""
    build = build_forecast(baseline, drivers, terminal_growth, risk_profile)
    sched = build.schedule
    texts = {**driver_rationales(baseline, build.drivers, terminal_growth), **(rationales or {})}
    diagnostics = build.diagnostics.model_copy(
        update={"warnings": build.diagnostics.warnings + list(extra_warnings)}
    )
    return ForecastOutput(
        ticker=ticker.upper(),
        fiscal_year=fiscal_year,
        base_revenue=sched["base_revenue"],
        forecast_horizon_years=sched["forecast_horizon_years"],
        revenue_cagr_pct=sched["revenue_cagr_pct"],
        cumulative_fcf=sched["cumulative_fcf"],
        average_annual_fcf=sched["average_annual_fcf"],
        guidance_source=guidance_source,
        tax_rate_pct=sched["tax_rate_pct"],
        projected_fcfs=sched["projected_fcfs"],
        forecast_schedule=sched["forecast_schedule"],
        forecast_table_markdown=sched["forecast_table_markdown"],
        assumptions=baseline,
        applied_drivers=build.drivers,
        driver_adjustments=list(adjustments),
        scenarios=build.scenarios,
        diagnostics=diagnostics,
        mdna_assessment=mdna,
        growth_rationale=texts["growth_rationale"],
        margin_rationale=texts["margin_rationale"],
        reinvestment_rationale=texts["reinvestment_rationale"],
        citations=list(citations),
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
        business_moat: Optional[BusinessMoatOutput] = None,
        risk_audit: Optional[RiskAuditOutput] = None,
        terminal_growth: float = DEFAULT_TERMINAL_GROWTH,
        callbacks: Optional[List[Any]] = None,
    ) -> ForecastOutput:
        ticker = ticker.upper()
        baseline = build_forecast_baseline(financial_audit)
        chunks = self._fetch_mdna(ticker, fiscal_year)
        mdna = assess_mdna_guidance(chunks)
        risk_profile = risk_audit.overall_risk_profile if risk_audit is not None else None

        if not mdna.is_informative:
            logger.info(
                f"Forecast {ticker} FY{fiscal_year}: {mdna.forward_looking_count} forward-looking MD&A "
                f"sentence(s); using historical fade without the LLM."
            )
            drivers = historical_fade_drivers(baseline, terminal_growth)
            return build_forecast_output(
                ticker=ticker, fiscal_year=fiscal_year, baseline=baseline, drivers=drivers,
                guidance_source="historical_fade", mdna=mdna, terminal_growth=terminal_growth,
                risk_profile=risk_profile,
                rationales=fallback_rationales(baseline, drivers, terminal_growth, mdna_fade_reason(mdna)),
            )

        self._configure_tools(baseline, {c.chunk_id for c in chunks}, terminal_growth, risk_profile)
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
    @staticmethod
    def _fetch_mdna(ticker: str, fiscal_year: int) -> List[NarrativeChunkResult]:
        try:
            return retrieve_10k_narrative(
                ticker=ticker, fiscal_year=fiscal_year, query=MDNA_QUERY,
                section_item="Item 7", limit=MDNA_CHUNK_LIMIT,
            )
        except Exception as e:
            logger.warning(f"MD&A retrieval failed for {ticker} FY{fiscal_year}: {e}")
            return []

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
            "prompt_forecasting_analyst_query.j2",
            ticker=ticker,
            fiscal_year=fiscal_year,
            baseline=baseline,
            history_table=render_history_table(baseline.historical_rows),
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
        terminal_growth: float,
        risk_profile: Optional[str],
    ) -> None:
        def propose_forecast_drivers(
            revenue_growth_pct: List[float],
            operating_margin_pct: List[float],
            capex_pct_of_revenue: Optional[float] = None,
            capex_citation_chunk_id: Optional[str] = None,
            nwc_pct_of_incremental_revenue: Optional[float] = None,
            nwc_citation_chunk_id: Optional[str] = None,
            evidence_chunk_ids: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            proposal = ForecastDriverProposal(
                revenue_growth_pct=revenue_growth_pct,
                operating_margin_pct=operating_margin_pct,
                capex_pct_of_revenue=capex_pct_of_revenue,
                capex_citation_chunk_id=capex_citation_chunk_id,
                nwc_pct_of_incremental_revenue=nwc_pct_of_incremental_revenue,
                nwc_citation_chunk_id=nwc_citation_chunk_id,
                evidence_chunk_ids=evidence_chunk_ids or [],
            )
            screen = screen_forecast_drivers(proposal, baseline, allowed_chunk_ids, terminal_growth)
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
        common = dict(
            ticker=ticker, fiscal_year=fiscal_year, baseline=ctx["baseline"], mdna=ctx["mdna"],
            terminal_growth=ctx["terminal_growth"], risk_profile=ctx["risk_profile"],
        )
        if accepted is None:
            logger.warning(f"{ticker} FY{fiscal_year}: {NO_ACCEPTED_DRIVERS_WARNING}")
            drivers = historical_fade_drivers(ctx["baseline"], ctx["terminal_growth"])
            return build_forecast_output(
                **common, drivers=drivers, guidance_source="historical_fade",
                rationales=fallback_rationales(ctx["baseline"], drivers, ctx["terminal_growth"], NO_DRIVERS_REASON),
                extra_warnings=[*extra_warnings, NO_ACCEPTED_DRIVERS_WARNING],
            )
        drivers, adjustments, evidence_ids = accepted
        return build_forecast_output(
            **common, drivers=drivers, guidance_source="mdna_informed", rationales=rationales,
            adjustments=adjustments,
            citations=self._resolve_citations(llm_citations, ctx["chunks"], evidence_ids, messages),
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

    def _resolve_citations(
        self,
        llm_citations: Any,
        chunks: Sequence[NarrativeChunkResult],
        evidence_ids: Sequence[str],
        messages: List[Any],
    ) -> List[Dict[str, Any]]:
        """Keeps only citations that point at a chunk the agent actually saw."""
        known: Dict[str, Dict[str, Any]] = {
            c.chunk_id: {"chunk_id": c.chunk_id, "item": c.item, "breadcrumb": c.breadcrumb} for c in chunks
        }
        for cit in self._extract_citations_from_messages(messages):
            known.setdefault(cit["chunk_id"], cit)
        ordered = [c.get("chunk_id") for c in (llm_citations or []) if isinstance(c, dict)] + list(evidence_ids)
        seen: Set[str] = set()
        out: List[Dict[str, Any]] = []
        for cid in ordered:
            if cid in known and cid not in seen:
                seen.add(cid)
                out.append(known[cid])
        return out
