"""Assembles ForecastOutput from the code-built forecast: rationale templates, citations and warnings."""

from typing import Any, Dict, List, Optional, Sequence, Set

from app.agents.shared.filing.narrative_search import NarrativeChunkResult
from app.agents.specialized.forecasting_analyst.state_forecasting_analyst import ForecastOutput, GuidanceSource
from app.agents.specialized.forecasting_analyst.modules.audited_baseline import ForecastBaseline
from app.agents.tools.forecast_drivers import DriverAdjustment, ForecastDrivers
from app.agents.tools.forecast_model import ForecastBuild
from app.agents.tools.mdna_guidance import MdnaAssessment


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
        "reinvestment_rationale": _reinvestment_rationale(baseline, drivers, terminal_growth),
    }


def _reinvestment_rationale(baseline: ForecastBaseline, drivers: ForecastDrivers, terminal_growth: float) -> str:
    start = drivers.capex_path[0] if drivers.capex_path else drivers.capex_pct_of_revenue
    steady = drivers.steady_state_capex_pct_of_revenue
    n = len(drivers.capex_path) or len(drivers.revenue_growth)
    if drivers.capex_pct_of_revenue == baseline.capex_pct_of_revenue:
        origin = (
            f"the recency-weighted {baseline.capex_years}-year audited level (latest year "
            f"{baseline.latest_capex_pct_of_revenue * 100:.2f}%, simple average "
            f"{baseline.capex_average_pct_of_revenue * 100:.2f}%)"
        )
    else:
        origin = "a cited MD&A CapEx plan"
    if steady is None:
        fade = ""
    elif drivers.reinvestment_method == "maintenance_fade":
        fade = (
            f" and fades to {steady * 100:.2f}% by year {n}, maintenance level (equal to D&A), because "
            f"final-year NOPAT is not positive"
        )
    else:
        fade = (
            f" and fades to a {steady * 100:.2f}% steady state by year {n}, where net reinvestment equals "
            f"{terminal_growth * 100:.2f}% terminal growth / {baseline.steady_state_ronic * 100:.1f}% return on new "
            f"capital ({baseline.steady_state_ronic_source.replace('_', ' ')}) = "
            f"{terminal_growth / baseline.steady_state_ronic * 100:.1f}% of NOPAT"
        )
    return (
        f"CapEx starts at {start * 100:.2f}% of revenue from {origin}{fade}; D&A is held at "
        f"{drivers.da_pct_of_revenue * 100:.2f}% ({baseline.da_source}); working capital at "
        f"{drivers.nwc_pct_of_incremental_revenue * 100:.2f}% of incremental revenue ({baseline.nwc_source}); "
        f"SBC at {drivers.sbc_pct_of_revenue * 100:.2f}% of revenue is already expensed in EBIT, so UFCF treats "
        f"it as a cash cost."
    )


def mdna_fade_reason(mdna: MdnaAssessment) -> str:
    return (
        f"Item 7 gave {mdna.forward_looking_count} forward-looking statement(s), too few to anchor a guided "
        f"forecast, so growth fades linearly from the historical trend."
    )


def fallback_rationales(
    baseline: ForecastBaseline, drivers: ForecastDrivers, terminal_growth: float, reason: str
) -> Dict[str, str]:
    """Growth rationale prefixed with why no guided forecast was built (the rest stays code-written)."""
    return {"growth_rationale": f"{reason} {driver_rationales(baseline, drivers, terminal_growth)['growth_rationale']}"}


def resolve_forecast_citations(
    llm_citations: Any,
    chunks: Sequence[NarrativeChunkResult],
    evidence_ids: Sequence[str],
    message_citations: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keeps only citations that point at a chunk the agent actually saw (MD&A excerpts or tool results)."""
    known: Dict[str, Dict[str, Any]] = {
        c.chunk_id: {"chunk_id": c.chunk_id, "item": c.item, "breadcrumb": c.breadcrumb} for c in chunks
    }
    for cit in message_citations:
        known.setdefault(cit["chunk_id"], cit)
    ordered = [c.get("chunk_id") for c in (llm_citations or []) if isinstance(c, dict)] + list(evidence_ids)
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    for cid in ordered:
        if cid in known and cid not in seen:
            seen.add(cid)
            out.append(known[cid])
    return out


def assemble_forecast(
    *,
    ticker: str,
    fiscal_year: int,
    baseline: ForecastBaseline,
    build: ForecastBuild,
    guidance_source: GuidanceSource,
    mdna: MdnaAssessment,
    terminal_growth: float,
    rationales: Optional[Dict[str, str]] = None,
    adjustments: Sequence[DriverAdjustment] = (),
    citations: Sequence[Dict[str, Any]] = (),
    extra_warnings: Sequence[str] = (),
) -> ForecastOutput:
    """ForecastOutput from the code-built forecast; missing rationales get factual defaults.

    The reinvestment rationale is always code-written: code owns the whole CapEx/D&A/working-capital path.
    """
    sched = build.schedule
    code_texts = driver_rationales(baseline, build.drivers, terminal_growth)
    texts = {**code_texts, **(rationales or {}), "reinvestment_rationale": code_texts["reinvestment_rationale"]}
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
