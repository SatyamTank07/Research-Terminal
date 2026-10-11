"""Deterministic valuation assembly: WACC, base DCF, bull/base/bear scenarios and data-quality provenance.

Consumes the Financial Auditor and Forecasting Analyst artifacts plus resolved market inputs and returns
every number the Valuation Specialist reports. The LLM only writes commentary on top of this build.
All `_pct` fields from upstream agents are whole percents and are converted to decimals here exactly once.
"""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import TYPE_CHECKING, Callable, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from app.agents.tools.dcf_tools import calculate_dcf_with_sensitivity, dcf_fair_value_per_share
from app.agents.tools.forecast_baseline import MARGIN_FLOOR
from app.agents.tools.forecast_model import SCENARIO_PROBABILITIES, project_with_overrides
from app.agents.tools.wacc_tools import WACCCalculationResult, calculate_wacc
from app.config import settings

if TYPE_CHECKING:  # runtime import would cycle through app.agents.specialized
    from app.agents.specialized.financial_auditor.state_financial_auditor import FinancialAuditOutput
    from app.agents.specialized.forecasting_analyst.state_forecasting_analyst import ForecastOutput

logger = logging.getLogger("finance_agent.tools.valuation_model")

STANCE_BAND_PCT = 10.0
RONIC_CAP = 0.30
NAIVE_FALLBACK_GROWTH = 0.05
NAIVE_FALLBACK_YEARS = 5
DAYS_PER_YEAR = 365.25

# Reverse DCF: constant explicit-period revenue growth that makes the base DCF equal the share price, with
# margins, tax, the CapEx/D&A/working-capital logic, WACC, terminal growth, terminal RONIC and the equity
# bridge held fixed. Scanned on a grid, then bisected inside the sign-change bracket nearest the modeled
# CAGR: bounded and always terminates. The range spans steep decline to hyper-growth for 5-10 years.
REVERSE_DCF_GROWTH_BOUNDS = (-0.20, 0.60)
# Companion solve: uniform operating-margin shift (decimal pp) at modeled growth, margins kept in [-200%, 100%].
REVERSE_DCF_MARGIN_SHIFT_BOUNDS = (-0.30, 0.50)
REVERSE_DCF_SCAN_POINTS = 33
REVERSE_DCF_MAX_ITERATIONS = 60
REVERSE_DCF_TOLERANCE = 1e-5
REVERSE_DCF_GAP_WARNING_PP = 5.0

ValuationStance = Literal["Undervalued", "Fairly Valued", "Overvalued"]
ReverseDCFStatus = Literal[
    "solved", "price_above_range", "price_below_range", "no_solution", "price_unavailable", "forecast_unavailable"
]


class MarketInputs(BaseModel):
    """Market data resolved by the orchestrator (live yfinance and/or user overrides) with provenance."""

    share_price: Optional[float] = None
    share_price_source: str = "unavailable"
    market_cap: Optional[float] = Field(None, description="$ Millions")
    beta: Optional[float] = None
    beta_source: str = Field("unavailable", description="'user_provided' | 'yfinance_raw' | 'unavailable'")
    risk_free_rate: float = settings.DEFAULT_RISK_FREE_RATE
    risk_free_rate_source: str = "house_default_config"
    market_data_source: str = "fallback_offline"


class ScenarioValuation(BaseModel):
    name: Literal["bull", "base", "bear"]
    probability: float
    revenue_cagr_pct: Optional[float] = None
    cumulative_fcf: float
    enterprise_value: float
    fair_value_per_share: float
    upside_downside_pct: Optional[float] = None


class ValuationDataQuality(BaseModel):
    forecast_source: Literal["forecasting_analyst", "naive_5pct_fallback"]
    market_data_source: str
    share_price_source: str
    warnings: List[str] = Field(default_factory=list)


class ValuationBuild(BaseModel):
    """Every deterministic figure of the DCF valuation ($ Millions unless stated)."""

    ticker: str
    fiscal_year: int
    valuation_date: str = Field(..., description="ISO date the value is stated at")
    balance_sheet_date: Optional[str] = Field(None, description="ISO fiscal year-end of the anchor 10-K")
    roll_forward_years: float = 0.0

    wacc_audit: WACCCalculationResult
    terminal_growth_rate: float
    discounting_convention: str
    terminal_method: str
    terminal_fcf: float
    ronic_used_pct: Optional[float] = None
    implied_terminal_ev_ebitda: Optional[float] = Field(
        None, description="Terminal value / final-year EBITDA; warned outside 5x-30x"
    )
    implied_terminal_ev_nopat: Optional[float] = Field(
        None, description="Terminal value / final-year NOPAT; warned outside 6x-35x"
    )

    projected_fcfs: List[float]
    pv_explicit_fcfs: float
    pv_terminal_value: float
    terminal_value_pct_of_ev: float
    enterprise_value: float
    net_debt: float = Field(..., description="Negative indicates a net cash surplus")
    noncontrolling_interests: float = 0.0
    equity_value: float = Field(..., description="Equity value at the valuation date")
    diluted_shares: float

    implied_fair_value_per_share: float = Field(..., description="Base-case fair value per share ($)")
    current_share_price: Optional[float] = None
    upside_downside_pct: Optional[float] = None
    valuation_stance: Optional[ValuationStance] = None
    implied_ev_ebitda: Optional[float] = Field(None, description="EV / audited base-year EBITDA")

    scenario_valuations: List[ScenarioValuation] = Field(default_factory=list)
    probability_weighted_fair_value: Optional[float] = None
    bear_fair_value: Optional[float] = None
    bull_fair_value: Optional[float] = None

    market_implied_growth_pct: Optional[float] = Field(
        None,
        description="Reverse DCF: constant explicit-period revenue growth (whole %) at which the base DCF equals "
        "the share price, all other inputs held; None unless reverse_dcf_status == 'solved'",
    )
    modeled_revenue_cagr_pct: Optional[float] = Field(
        None, description="Base-case revenue CAGR over the same explicit horizon (whole %), for comparison"
    )
    reverse_dcf_horizon_years: Optional[int] = Field(None, description="Explicit years the implied growth applies to")
    market_implied_margin_shift_pp: Optional[float] = Field(
        None,
        description="Reverse DCF companion: uniform operating-margin shift (percentage points, every explicit year) "
        "at modeled growth that equates the base DCF to the price; None without a price or a root in -30..+50 pp",
    )
    modeled_final_margin_pct: Optional[float] = Field(
        None, description="Base-case final explicit-year operating margin (whole %), for the margin comparison"
    )
    reverse_dcf_status: ReverseDCFStatus = Field(
        "price_unavailable",
        description="'solved'; 'price_above_range' / 'price_below_range' (no growth in -20%..60% reaches the price); "
        "'no_solution' (no finite DCF values); 'price_unavailable'; 'forecast_unavailable'",
    )
    reverse_dcf_summary: str = Field("", description="One-line reverse-DCF result or the reason there is none")

    sensitivity_matrix_markdown: str
    ronic_sensitivity_matrix_markdown: str = Field(
        "", description="WACC x terminal RONIC grid of fair value per share ('' unless the TV is value-driver)"
    )
    equity_bridge_markdown: str
    scenario_table_markdown: str = ""
    data_quality: ValuationDataQuality


# ==============================================================================
# Helpers
# ==============================================================================
def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _upside(fair: float, price: Optional[float]) -> Optional[float]:
    if price is None or price <= 0:
        return None
    return round((fair - price) / price * 100.0, 2)


def _stance(upside: Optional[float]) -> Optional[ValuationStance]:
    if upside is None:
        return None
    if upside > STANCE_BAND_PCT:
        return "Undervalued"
    if upside < -STANCE_BAND_PCT:
        return "Overvalued"
    return "Fairly Valued"


def _resolve_ronic(forecast: Optional["ForecastOutput"], wacc: float, warnings: List[str]) -> Optional[float]:
    """RONIC for the value-driver TV, clamped to [WACC, 30%].

    Prefers the forecast's steady-state RONIC, which already sized the final explicit year's reinvestment,
    so the terminal value continues the same reinvestment identity; falls back to the horizon RONIC, then
    base ROIC.
    """
    if forecast is None:
        return None
    diag = forecast.diagnostics
    raw_pct = next(
        (p for p in (diag.steady_state_ronic_pct, diag.ronic_pct, diag.base_roic_pct) if p is not None), None
    )
    if raw_pct is None:
        warnings.append("No RONIC or base ROIC available; terminal value grows the final UFCF instead.")
        return None
    raw = raw_pct / 100.0
    clamped = min(max(raw, wacc), RONIC_CAP)
    if clamped != raw:
        if raw < wacc:
            warnings.append(
                f"Terminal RONIC {raw * 100:.1f}% clamped to {clamped * 100:.1f}% (WACC: growth beyond the horizon "
                f"creates no value). The final explicit year still reinvests at the {raw * 100:.1f}% steady-state "
                f"return, so the explicit period is more conservative than the terminal value."
            )
        else:
            warnings.append(f"Terminal RONIC {raw * 100:.1f}% clamped to {clamped * 100:.1f}% ({RONIC_CAP:.0%} cap).")
    return clamped


def solve_reverse_dcf(
    value_fn: Callable[[float], Optional[float]],
    target: float,
    bounds: Tuple[float, float] = REVERSE_DCF_GROWTH_BOUNDS,
    anchor: Optional[float] = None,
) -> Tuple[Optional[float], str]:
    """Finds x in `bounds` with value_fn(x) == target. Never raises; at most
    REVERSE_DCF_SCAN_POINTS + REVERSE_DCF_MAX_ITERATIONS evaluations.

    Returns (root, 'solved'), or (None, 'price_above_range' | 'price_below_range' | 'no_solution').
    `value_fn` may return None (no finite value); such points are skipped. With several brackets (a
    non-monotonic value curve) the one nearest `anchor` wins.
    """
    def f(x: float) -> Optional[float]:
        try:
            v = value_fn(x)
        except Exception:  # noqa: BLE001 - the solver must never raise
            return None
        return v - target if v is not None and math.isfinite(v) else None

    lo, hi = bounds
    xs = [lo + (hi - lo) * i / (REVERSE_DCF_SCAN_POINTS - 1) for i in range(REVERSE_DCF_SCAN_POINTS)]
    points = [(x, fx) for x, fx in ((x, f(x)) for x in xs) if fx is not None]
    if not points:
        return None, "no_solution"
    exact = [x for x, fx in points if fx == 0.0]
    brackets = [
        (a, fa, b, fb) for (a, fa), (b, fb) in zip(points, points[1:]) if fa != 0.0 and fb != 0.0 and (fa < 0) != (fb < 0)
    ]
    if not exact and not brackets:
        return None, ("price_above_range" if all(fx < 0 for _, fx in points) else "price_below_range")
    ref = anchor if anchor is not None else (lo + hi) / 2.0
    candidates = [(abs(x - ref), x, None) for x in exact] + [
        (abs((br[0] + br[2]) / 2.0 - ref), None, br) for br in brackets
    ]
    _, exact_root, bracket = min(candidates, key=lambda c: c[0])
    if bracket is None:
        return exact_root, "solved"
    a, fa, b, fb = bracket
    for _ in range(REVERSE_DCF_MAX_ITERATIONS):
        mid = (a + b) / 2.0
        fm = f(mid)
        if fm is None:  # undefined inside the bracket: settle on the bracket midpoint
            break
        if fm == 0.0 or (b - a) / 2.0 < REVERSE_DCF_TOLERANCE:
            return mid, "solved"
        if (fa < 0) != (fm < 0):
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return (a + b) / 2.0, "solved"


def _reverse_dcf(
    forecast: Optional["ForecastOutput"],
    price: Optional[float],
    terminal_growth_rate: float,
    dcf_kwargs: dict,
    warnings: List[str],
) -> dict:
    """Market-implied constant explicit-period revenue growth, plus the uniform operating-margin shift that
    reaches the price at modeled growth; never raises."""
    empty = dict(
        market_implied_growth_pct=None, modeled_revenue_cagr_pct=None, reverse_dcf_horizon_years=None,
        market_implied_margin_shift_pp=None, modeled_final_margin_pct=None,
    )
    if forecast is None or not forecast.applied_drivers.revenue_growth:
        return {**empty, "reverse_dcf_status": "forecast_unavailable",
                "reverse_dcf_summary": "No forecast schedule, so no market-implied growth was solved."}
    horizon = len(forecast.applied_drivers.revenue_growth)
    modeled = forecast.revenue_cagr_pct
    baseline, drivers = forecast.assumptions, forecast.applied_drivers
    final_margin = drivers.operating_margin[-1]
    base = {
        **empty, "modeled_revenue_cagr_pct": modeled, "reverse_dcf_horizon_years": horizon,
        "modeled_final_margin_pct": round(final_margin * 100.0, 2),
    }
    if price is None:
        return {**base, "reverse_dcf_status": "price_unavailable",
                "reverse_dcf_summary": "No share price, so no market-implied growth was solved."}

    def value_at_growth(growth: float) -> Optional[float]:
        fcfs, nopat = project_with_overrides(baseline, drivers, terminal_growth_rate, growth=[growth] * horizon)
        return dcf_fair_value_per_share(fcfs, terminal_nopat=nopat, **dcf_kwargs)

    def value_at_margin_shift(shift: float) -> Optional[float]:
        margins = [min(max(m + shift, MARGIN_FLOOR), 1.0) for m in drivers.operating_margin]
        fcfs, nopat = project_with_overrides(baseline, drivers, terminal_growth_rate, margins=margins)
        return dcf_fair_value_per_share(fcfs, terminal_nopat=nopat, **dcf_kwargs)

    root, status = solve_reverse_dcf(value_at_growth, price, anchor=modeled / 100.0)
    shift, shift_status = solve_reverse_dcf(value_at_margin_shift, price, REVERSE_DCF_MARGIN_SHIFT_BOUNDS, anchor=0.0)
    shift_pp = round(shift * 100.0, 2) if shift_status == "solved" else None
    if shift_pp is not None:
        margin_text = (
            f" At modeled growth the price instead needs a {shift_pp:+.1f} pp operating-margin shift in every year "
            f"(final-year margin {min(max(final_margin + shift, MARGIN_FLOOR), 1.0) * 100:.1f}% versus "
            f"{final_margin * 100:.1f}% modeled)."
        )
    else:
        lo_m, hi_m = REVERSE_DCF_MARGIN_SHIFT_BOUNDS
        margin_text = (
            f" No uniform operating-margin shift between {lo_m * 100:+.0f} and {hi_m * 100:+.0f} pp reaches the price "
            f"at modeled growth."
        )
    base["market_implied_margin_shift_pp"] = shift_pp

    lo, hi = REVERSE_DCF_GROWTH_BOUNDS
    if status == "solved":
        implied = round(root * 100.0, 2)
        gap = implied - modeled
        summary = (
            f"At ${price:.2f} the market implies {implied:.1f}% constant revenue growth for {horizon} years (then "
            f"{terminal_growth_rate * 100:.1f}% terminal), holding modeled margins, reinvestment, WACC and terminal "
            f"RONIC, versus a {modeled:.1f}% modeled revenue CAGR." + margin_text
        )
        if abs(gap) > REVERSE_DCF_GAP_WARNING_PP:
            warnings.append(
                f"Reverse DCF: the share price implies {implied:.1f}% annual revenue growth for {horizon} years versus "
                f"{modeled:.1f}% modeled ({gap:+.1f} pp); the market is pricing materially "
                f"{'more' if gap > 0 else 'less'} growth, or different margins and returns, than the base case."
            )
        return {**base, "market_implied_growth_pct": implied, "reverse_dcf_status": "solved",
                "reverse_dcf_summary": summary}
    if status == "price_above_range":
        summary = (
            f"The ${price:.2f} share price exceeds the DCF value even at {hi * 100:.0f}% constant revenue growth for "
            f"{horizon} years at modeled margins."
        )
    elif status == "price_below_range":
        summary = (
            f"The ${price:.2f} share price is below the DCF value even at {lo * 100:.0f}% constant revenue growth for "
            f"{horizon} years at modeled margins."
        )
    else:
        summary = "The reverse DCF found no finite valuation across the growth range."
    summary += margin_text
    warnings.append(f"Reverse DCF: {summary}")
    return {**base, "reverse_dcf_status": status, "reverse_dcf_summary": summary}


def _scenario_terminal_nopat(forecast: "ForecastOutput", scenario, tax: float) -> Optional[float]:
    """Final-year NOPAT of a scenario rebuilt from its growth and margin path (same formula as the schedule)."""
    if not scenario.revenue_growth_pct or not scenario.operating_margin_pct:
        return None
    revenue = forecast.base_revenue
    for g in scenario.revenue_growth_pct:
        revenue *= 1.0 + g / 100.0
    return revenue * scenario.operating_margin_pct[-1] / 100.0 * (1.0 - tax)


def _scenario_table(scenarios: List[ScenarioValuation], weighted: Optional[float], price: Optional[float]) -> str:
    lines = [
        "| Scenario | Probability | Revenue CAGR | Cumulative UFCF | Fair Value / Share | vs. Price |",
        "|:---|:---:|:---:|---:|---:|:---:|",
    ]
    for s in scenarios:
        cagr = f"{s.revenue_cagr_pct:.1f}%" if s.revenue_cagr_pct is not None else "n/a"
        vs = f"{s.upside_downside_pct:+.1f}%" if s.upside_downside_pct is not None else "n/a"
        lines.append(
            f"| {s.name.title()} | {s.probability:.0%} | {cagr} | ${s.cumulative_fcf:,.0f}M | "
            f"${s.fair_value_per_share:.2f} | {vs} |"
        )
    if weighted is not None:
        vs = _upside(weighted, price)
        lines.append(
            f"| **Probability-weighted** | 100% | | | **${weighted:.2f}** | "
            f"{f'{vs:+.1f}%' if vs is not None else 'n/a'} |"
        )
    return "\n".join(lines)


# ==============================================================================
# Main Assembly
# ==============================================================================
def build_valuation(
    ticker: str,
    fiscal_year: int,
    financial_audit: "FinancialAuditOutput",
    forecast: Optional["ForecastOutput"],
    market: MarketInputs,
    *,
    terminal_growth_rate: float,
    wacc_override: Optional[float] = None,
    valuation_date: Optional[date] = None,
    balance_sheet_date: Optional[str] = None,
) -> ValuationBuild:
    warnings: List[str] = []
    bs = financial_audit.balance_sheet
    latest = financial_audit.multi_year_history[-1]
    shares = bs.valuation_shares_outstanding
    valuation_date = valuation_date or date.today()

    # 1. Cash flows and tax rate
    if forecast is not None and forecast.projected_fcfs:
        projected_fcfs = list(forecast.projected_fcfs)
        tax_rate = forecast.tax_rate_pct / 100.0
        forecast_source = "forecasting_analyst"
    else:
        projected_fcfs = [
            round(latest.free_cash_flow * (1.0 + NAIVE_FALLBACK_GROWTH) ** i, 2)
            for i in range(1, NAIVE_FALLBACK_YEARS + 1)
        ]
        tax_rate = financial_audit.profitability_and_return_ratios.effective_tax_rate_pct / 100.0
        forecast_source = "naive_5pct_fallback"
        warnings.append("No forecast available: UFCF is the last audited FCF grown 5% a year (placeholder).")

    # 2. Market inputs and capital-structure equity value
    beta, beta_source, blume = market.beta, market.beta_source, market.beta_source == "yfinance_raw"
    if beta is None:
        beta, beta_source, blume = 1.0, "default_market_beta", False
        warnings.append("No beta available; CAPM uses a market beta of 1.0.")
    if market.risk_free_rate_source != "us_10y_treasury_live":
        warnings.append(f"Risk-free rate {market.risk_free_rate * 100:.2f}% is the configured default, not a live yield.")

    equity_value_for_weights = market.market_cap
    if equity_value_for_weights is None and market.share_price is not None:
        equity_value_for_weights = round(market.share_price * shares, 2)
    equity_value_source = "market_value"
    if equity_value_for_weights is None:
        if bs.stockholders_equity > 0:
            equity_value_for_weights, equity_value_source = bs.stockholders_equity, "book_equity_fallback"
            warnings.append("No market price: WACC weights use book equity, which understates the equity weight.")
        else:
            warnings.append("No market price and non-positive book equity: WACC weighted 100% equity.")

    wacc_audit = WACCCalculationResult.model_validate(calculate_wacc(
        beta=beta,
        total_debt=bs.total_debt,
        prior_total_debt=bs.prior_total_debt,
        market_cap=equity_value_for_weights,
        risk_free_rate=market.risk_free_rate,
        equity_risk_premium=settings.DEFAULT_EQUITY_RISK_PREMIUM,
        tax_rate=min(max(tax_rate, 0.0), 0.5),
        interest_expense=latest.interest_expense,
        apply_blume_adjustment=blume,
        beta_source=beta_source,
        risk_free_rate_source=market.risk_free_rate_source,
        equity_risk_premium_source="house_assumption_config",
        equity_value_source=equity_value_source,
        wacc_override=wacc_override,
    ))
    if wacc_audit.cost_of_debt_note:
        warnings.append(wacc_audit.cost_of_debt_note)
    wacc = wacc_audit.wacc

    # 3. Valuation date roll-forward
    bs_date = _parse_date(balance_sheet_date)
    roll_years = 0.0
    if bs_date is None:
        warnings.append("Balance-sheet date unknown; value is stated at the fiscal year-end (no roll-forward).")
    else:
        roll_years = max(0.0, (valuation_date - bs_date).days / DAYS_PER_YEAR)

    # 4. Base DCF
    ronic = _resolve_ronic(forecast, wacc, warnings)
    final_year = forecast.forecast_schedule[-1] if forecast is not None and forecast.forecast_schedule else None
    common = dict(
        wacc=wacc,
        terminal_growth_rate=terminal_growth_rate,
        net_debt=bs.net_debt,
        diluted_shares=shares,
        mid_year_convention=True,
        ronic=ronic,
        noncontrolling_interests=bs.noncontrolling_interests or 0.0,
        roll_forward_years=roll_years,
        cost_of_equity=wacc_audit.cost_of_equity,
        risk_free_rate=market.risk_free_rate,
    )
    base = calculate_dcf_with_sensitivity(
        projected_fcfs=projected_fcfs,
        terminal_nopat=final_year.projected_nopat if final_year else None,
        terminal_ebitda=(final_year.projected_ebit + final_year.projected_depreciation) if final_year else None,
        **common,
    )
    warnings.extend(base["warnings"])
    fair = base["fair_value_per_share"]
    price = market.share_price if market.share_price and market.share_price > 0 else None
    upside = _upside(fair, price)

    # 4b. Reverse DCF (market-implied explicit-period growth), same WACC / TV / bridge as the headline
    reverse = _reverse_dcf(
        forecast if forecast_source == "forecasting_analyst" else None, price, terminal_growth_rate,
        dict(
            wacc=wacc, terminal_growth_rate=terminal_growth_rate, net_debt=bs.net_debt, diluted_shares=shares,
            ronic=ronic, noncontrolling_interests=bs.noncontrolling_interests or 0.0,
            roll_forward_years=roll_years, cost_of_equity=wacc_audit.cost_of_equity, mid_year_convention=True,
        ),
        warnings,
    )

    # 5. Scenario valuations (same WACC, growth and bridge; base reuses the headline DCF)
    scenario_vals: List[ScenarioValuation] = []
    if forecast is not None:
        for sc in forecast.scenarios:
            if sc.name == "base":
                ev, per_share = base["enterprise_value"], fair
            else:
                res = calculate_dcf_with_sensitivity(
                    projected_fcfs=sc.projected_fcfs,
                    terminal_nopat=_scenario_terminal_nopat(forecast, sc, tax_rate),
                    **common,
                )
                ev, per_share = res["enterprise_value"], res["fair_value_per_share"]
            scenario_vals.append(ScenarioValuation(
                name=sc.name,
                probability=SCENARIO_PROBABILITIES.get(sc.name, sc.probability),
                revenue_cagr_pct=sc.revenue_cagr_pct,
                cumulative_fcf=sc.cumulative_fcf,
                enterprise_value=ev,
                fair_value_per_share=per_share,
                upside_downside_pct=_upside(per_share, price),
            ))
    by_name = {s.name: s for s in scenario_vals}
    weighted = (
        round(sum(s.probability * s.fair_value_per_share for s in scenario_vals), 2)
        if {"bull", "base", "bear"} <= set(by_name) else None
    )

    base_ebitda = financial_audit.solvency_and_liquidity_ratios.ebitda
    implied_ev_ebitda = (
        round(base["enterprise_value"] / base_ebitda, 2) if base_ebitda is not None and base_ebitda > 0 else None
    )

    build = ValuationBuild(
        ticker=ticker.upper(),
        fiscal_year=fiscal_year,
        valuation_date=valuation_date.isoformat(),
        balance_sheet_date=bs_date.isoformat() if bs_date else None,
        roll_forward_years=base["roll_forward_years"],
        wacc_audit=wacc_audit,
        terminal_growth_rate=terminal_growth_rate,
        discounting_convention=base["discounting_convention"],
        terminal_method=base["terminal_method"],
        terminal_fcf=base["terminal_fcf"],
        ronic_used_pct=round(base["ronic"] * 100.0, 2) if base["ronic"] is not None else None,
        implied_terminal_ev_ebitda=base["implied_terminal_ev_ebitda"],
        implied_terminal_ev_nopat=base["implied_terminal_ev_nopat"],
        projected_fcfs=projected_fcfs,
        pv_explicit_fcfs=base["pv_explicit_fcfs"],
        pv_terminal_value=base["pv_terminal_value"],
        terminal_value_pct_of_ev=base["terminal_value_pct_of_ev"],
        enterprise_value=base["enterprise_value"],
        net_debt=base["net_debt"],
        noncontrolling_interests=base["noncontrolling_interests"],
        equity_value=base["equity_value"],
        diluted_shares=base["diluted_shares"],
        implied_fair_value_per_share=fair,
        current_share_price=price,
        upside_downside_pct=upside,
        valuation_stance=_stance(upside),
        implied_ev_ebitda=implied_ev_ebitda,
        scenario_valuations=scenario_vals,
        probability_weighted_fair_value=weighted,
        bear_fair_value=by_name["bear"].fair_value_per_share if "bear" in by_name else None,
        bull_fair_value=by_name["bull"].fair_value_per_share if "bull" in by_name else None,
        **reverse,
        sensitivity_matrix_markdown=base["sensitivity_matrix_markdown"],
        ronic_sensitivity_matrix_markdown=base["ronic_sensitivity_matrix_markdown"],
        equity_bridge_markdown=base["equity_bridge_markdown"],
        scenario_table_markdown=_scenario_table(scenario_vals, weighted, price) if scenario_vals else "",
        data_quality=ValuationDataQuality(
            forecast_source=forecast_source,
            market_data_source=market.market_data_source,
            share_price_source=market.share_price_source if price is not None else "unavailable",
            warnings=warnings,
        ),
    )
    logger.info(
        f"build_valuation {build.ticker} FY{fiscal_year}: WACC {wacc_audit.wacc_pct:.2f}% ({wacc_audit.wacc_source}), "
        f"base ${fair:.2f}, bear ${build.bear_fair_value}, bull ${build.bull_fair_value}, "
        f"weighted ${weighted}, roll {roll_years:.2f}y, {len(warnings)} warnings"
    )
    return build
