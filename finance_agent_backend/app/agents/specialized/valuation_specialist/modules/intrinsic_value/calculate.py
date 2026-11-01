"""What is the business worth today? Mid-year two-stage DCF (Gordon growth, TV at N-0.5), optional value-driver
terminal cash flow NOPAT x (1 + g) x (1 - g / RONIC), the EV-to-equity bridge (net debt, noncontrolling interests)
and the roll-forward from the balance-sheet date to the valuation date (operating value at the discount rate,
net debt and minority claims at the risk-free rate).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from app.agents.specialized.valuation_specialist.modules.cost_of_capital.schema import CostOfCapital
from app.agents.specialized.valuation_specialist.modules.intrinsic_value.render import render_equity_bridge
from app.agents.specialized.valuation_specialist.modules.intrinsic_value.schema import (
    DCFCase,
    DCFPoint,
    DCFTerms,
    DCFValue,
    IntrinsicValue,
    IntrinsicValueInputs,
)
from app.agents.specialized.valuation_specialist.modules.intrinsic_value.thresholds import (
    DAYS_PER_YEAR,
    MAX_TERMINAL_GROWTH_RATE,
    MAX_WACC_RATE,
    MIN_TERMINAL_GROWTH_RATE,
    RONIC_CAP,
    STALE_ROLL_FORWARD_YEARS,
    TERMINAL_EV_EBITDA_BAND,
    TERMINAL_EV_NOPAT_BAND,
    TERMINAL_GROWTH_WARNING_RATE,
    TV_CONCENTRATION_WARNING_PCT,
)

if TYPE_CHECKING:
    from app.agents.specialized.forecasting_analyst.state_forecasting_analyst import ForecastDiagnostics

logger = logging.getLogger("finance_agent.agents.valuation_specialist")

def _validate_finite_float(val: Any, name: str) -> float:
    """Validates that a numeric argument is a real, finite float (rejects NaN and Infinity)."""
    if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
        raise ValueError(f"{name} must be a valid finite number, got: {val}")
    return float(val)


def _optional_finite(val: Any, name: str) -> Optional[float]:
    return None if val is None else _validate_finite_float(val, name)


def value_point(case: DCFCase, rate: float, g: float) -> Optional[DCFPoint]:
    """One DCF evaluation at `rate` and `g`; None when the inputs have no finite value (rate <= g, RONIC <= g).

    Value at the valuation date: the operating value compounds at the discount rate it was computed with,
    while net debt and noncontrolling interests (cash and debt balances) compound at the risk-free rate.
    """
    fcfs, ronic, terminal_nopat, offset = case.fcfs, case.ronic, case.terminal_nopat, case.offset
    net_debt, noncontrolling_interests = case.net_debt, case.noncontrolling_interests
    if case.terminal_method == "value_driver":
        if ronic is None or terminal_nopat is None or ronic <= g:
            return None
        tv_fcf = terminal_nopat * (1.0 + g) * (1.0 - g / ronic)
    else:
        tv_fcf = fcfs[-1] * (1.0 + g)
    if rate <= g or rate <= 0:
        return None
    pv_fcfs = sum(fcf / ((1.0 + rate) ** (idx + 1 - offset)) for idx, fcf in enumerate(fcfs))
    tv = tv_fcf / (rate - g)
    pv_tv = tv / ((1.0 + rate) ** (len(fcfs) - offset))
    ev = pv_fcfs + pv_tv
    equity_bs = ev - net_debt - noncontrolling_interests
    equity = ev * (1.0 + rate) ** case.roll_forward_years - (net_debt + noncontrolling_interests) * case.claims_roll_factor
    return DCFPoint(
        tv_fcf=tv_fcf, pv_fcfs=pv_fcfs, tv=tv, pv_tv=pv_tv, ev=ev,
        equity_bs=equity_bs, equity=equity, per_share=equity / case.diluted_shares,
    )


def _terminal_method(terminal_nopat: Optional[float], ronic: Optional[float], g: float) -> str:
    if terminal_nopat is not None and ronic is not None and terminal_nopat > 0 and ronic > g:
        return "value_driver"
    return "last_fcf_growth"


def dcf_fair_value_per_share(
    projected_fcfs: List[float],
    wacc: float,
    terminal_growth_rate: float,
    net_debt: float,
    diluted_shares: float,
    *,
    terminal_nopat: Optional[float] = None,
    ronic: Optional[float] = None,
    noncontrolling_interests: float = 0.0,
    roll_forward_years: float = 0.0,
    risk_free_rate: Optional[float] = None,
    mid_year_convention: bool = True,
) -> Optional[float]:
    """Fair value per share with exactly the headline engine's math, without grids, tables or logging.

    Never raises: returns None for invalid or non-convergent inputs. Used by the reverse DCF solver.
    """
    try:
        fcfs = [_validate_finite_float(f, "fcf") for f in projected_fcfs]
        if not fcfs or diluted_shares <= 0 or not 0 < wacc < 1.0 or terminal_growth_rate >= wacc:
            return None
        roll_years = max(0.0, roll_forward_years or 0.0)
        case = DCFCase(
            fcfs=tuple(fcfs),
            wacc=wacc,
            terminal_growth_rate=terminal_growth_rate,
            terminal_method=_terminal_method(terminal_nopat, ronic, terminal_growth_rate),
            terminal_nopat=terminal_nopat,
            ronic=ronic,
            offset=0.5 if mid_year_convention else 0.0,
            net_debt=net_debt,
            noncontrolling_interests=noncontrolling_interests or 0.0,
            roll_forward_years=roll_years,
            claims_roll_factor=(1.0 + (risk_free_rate or 0.0)) ** roll_years,
            diluted_shares=diluted_shares,
        )
        point = value_point(case, wacc, terminal_growth_rate)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None
    if point is None or not math.isfinite(point.per_share):
        return None
    return point.per_share


def _multiple_warning(name: str, value: Optional[float], band: Tuple[float, float]) -> Optional[str]:
    if value is None:
        return None
    lo, hi = band
    if value < lo:
        return (
            f"Implied terminal {name} of {value:.1f}x is below the {lo:.0f}x-{hi:.0f}x plausibility band: the "
            f"terminal cash flow is small relative to earnings (heavy final-year reinvestment, a low RONIC or a "
            f"high discount rate), so the terminal value may be understated."
        )
    if value > hi:
        return (
            f"Implied terminal {name} of {value:.1f}x is above the {lo:.0f}x-{hi:.0f}x plausibility band: the "
            f"perpetuity assumes more than a mature company typically sustains (high RONIC, low discount rate or "
            f"depressed final-year earnings), so the terminal value may be overstated."
        )
    return None


def calculate_dcf(
    projected_fcfs: List[float],
    wacc: float,
    terminal_growth_rate: float,
    net_debt: float,
    diluted_shares: float,
    mid_year_convention: bool = True,
    *,
    terminal_nopat: Optional[float] = None,
    ronic: Optional[float] = None,
    terminal_ebitda: Optional[float] = None,
    noncontrolling_interests: float = 0.0,
    roll_forward_years: float = 0.0,
    risk_free_rate: Optional[float] = None,
) -> DCFValue:
    """
    Calculates intrinsic Enterprise Value, Equity Value, and Fair Value per share
    using a 2-stage Perpetual Gordon Growth DCF model.

    All dollar quantities (cash flows, net debt, NCI) and share counts must be on the same scale
    (standardized to $ millions and millions of shares).

    DISCOUNTING CONVENTION:
        With mid_year_convention=True, explicit cash flows are discounted at (t - 0.5) and the
        perpetuity-growth terminal value at (N - 0.5): the Gordon value is built from cash flows
        that also arrive through the year (Rosenbaum & Pearl; McKinsey applies the same half-year
        uplift to the whole value). Year-end discounting uses t and N.

    TERMINAL CASH FLOW:
        When terminal_nopat and ronic are supplied, the capitalized cash flow is the value-driver
        figure NOPAT_N x (1 + g) x (1 - g / RONIC), which keeps reinvestment consistent with growth.
        Otherwise the final UFCF is grown by (1 + g).

    EQUITY BRIDGE:
        Equity = EV - net debt - noncontrolling interests at the balance-sheet date. To state it at the
        valuation date, EV rolls forward at the discount rate and net debt / NCI at the risk-free rate
        (cash earns, and debt accrues, roughly the risk-free rate; zero when Rf is not given).
        net_debt is NEGATIVE for a net cash surplus.

    Raises:
        ValueError: If inputs violate finite number checks or economic boundary conditions.
    """
    # --------------------------------------------------------------------------
    # Step 0: Strict Finite Float & Range Validation
    # --------------------------------------------------------------------------
    if not projected_fcfs:
        raise ValueError("projected_fcfs must contain at least one forecast period.")

    cleaned_fcfs: List[float] = []
    for idx, fcf in enumerate(projected_fcfs):
        cleaned_fcfs.append(_validate_finite_float(fcf, f"projected_fcfs[{idx}]"))

    wacc = _validate_finite_float(wacc, "wacc")
    terminal_growth_rate = _validate_finite_float(terminal_growth_rate, "terminal_growth_rate")
    net_debt = _validate_finite_float(net_debt, "net_debt")
    diluted_shares = _validate_finite_float(diluted_shares, "diluted_shares")
    noncontrolling_interests = _validate_finite_float(noncontrolling_interests or 0.0, "noncontrolling_interests")
    roll_forward_years = max(0.0, _validate_finite_float(roll_forward_years or 0.0, "roll_forward_years"))
    terminal_nopat = _optional_finite(terminal_nopat, "terminal_nopat")
    ronic = _optional_finite(ronic, "ronic")
    terminal_ebitda = _optional_finite(terminal_ebitda, "terminal_ebitda")
    risk_free_rate = _optional_finite(risk_free_rate, "risk_free_rate")

    if wacc <= 0:
        raise ValueError(f"WACC must be strictly positive (> 0), got: {wacc}")

    # Catch LLM percentage formatting handoff error (e.g. 8.5 instead of 0.085)
    if wacc >= 1.0:
        raise ValueError(
            f"WACC ({wacc}) appears to be passed as a whole percentage rather than a decimal. "
            f"Please pass WACC as a decimal (e.g. 0.085 for 8.5%, max 0.50)."
        )
    if wacc > MAX_WACC_RATE:
        raise ValueError(f"WACC ({wacc * 100:.1f}%) is unrealistically high (maximum allowed is 50.0%).")

    if diluted_shares <= 0:
        raise ValueError(f"diluted_shares must be strictly positive (> 0), got: {diluted_shares}")

    if terminal_growth_rate >= wacc:
        raise ValueError(
            f"terminal_growth_rate ({terminal_growth_rate * 100:.2f}%) must be strictly less "
            f"than WACC ({wacc * 100:.2f}%) for Gordon Growth convergence."
        )

    # Economic reasonableness boundary checks on perpetual growth
    if terminal_growth_rate > MAX_TERMINAL_GROWTH_RATE:
        raise ValueError(
            f"terminal_growth_rate ({terminal_growth_rate * 100:.2f}%) exceeds economically plausible "
            f"perpetual GDP growth rate (maximum allowed is 5.0%)."
        )
    if terminal_growth_rate < MIN_TERMINAL_GROWTH_RATE:
        raise ValueError(
            f"terminal_growth_rate ({terminal_growth_rate * 100:.2f}%) is unrealistically negative "
            f"(minimum allowed is -2.0%)."
        )

    warnings: List[str] = []
    if terminal_growth_rate > TERMINAL_GROWTH_WARNING_RATE:
        warnings.append(
            f"Terminal growth {terminal_growth_rate * 100:.2f}% exceeds typical long-term developed-market "
            f"GDP growth (2.0%-3.0%)."
        )
    if risk_free_rate is not None and terminal_growth_rate > risk_free_rate:
        warnings.append(
            f"Terminal growth {terminal_growth_rate * 100:.2f}% exceeds the risk-free rate "
            f"{risk_free_rate * 100:.2f}%; perpetual growth above Rf implies the firm outgrows the economy."
        )

    terminal_method = "last_fcf_growth"
    if terminal_nopat is not None and ronic is not None:
        if terminal_nopat <= 0:
            warnings.append("Final-year NOPAT is not positive; value-driver terminal value skipped (final UFCF grown).")
        elif ronic <= terminal_growth_rate:
            warnings.append(
                f"RONIC {ronic * 100:.2f}% does not exceed terminal growth; value-driver terminal value skipped."
            )
        else:
            terminal_method = "value_driver"

    num_years = len(cleaned_fcfs)
    convention_label = "mid_year" if mid_year_convention else "year_end"
    offset = 0.5 if mid_year_convention else 0.0
    roll_factor = (1.0 + wacc) ** roll_forward_years
    claims_rate = risk_free_rate if risk_free_rate is not None else 0.0
    claims_roll_factor = (1.0 + claims_rate) ** roll_forward_years

    if roll_forward_years > STALE_ROLL_FORWARD_YEARS:
        warnings.append(
            f"Valuation is rolled forward {roll_forward_years:.2f} years from the balance-sheet date; "
            f"the anchor filing is stale and cash generated since then is not reflected in net debt."
        )

    logger.info(
        f"calculate_dcf: net_debt passed: ${net_debt:,.2f}M "
        f"({'Net Cash Surplus' if net_debt < 0 else 'Net Indebtedness'}) | Convention: {convention_label} | "
        f"TV method: {terminal_method}"
    )

    case = DCFCase(
        fcfs=tuple(cleaned_fcfs),
        wacc=wacc,
        terminal_growth_rate=terminal_growth_rate,
        terminal_method=terminal_method,
        terminal_nopat=terminal_nopat,
        ronic=ronic,
        offset=offset,
        net_debt=net_debt,
        noncontrolling_interests=noncontrolling_interests,
        roll_forward_years=roll_forward_years,
        claims_roll_factor=claims_roll_factor,
        diluted_shares=diluted_shares,
    )

    # --------------------------------------------------------------------------
    # Steps 1-3: Explicit PV, Terminal Value, Valuation Bridge
    # --------------------------------------------------------------------------
    base = value_point(case, wacc, terminal_growth_rate)
    enterprise_value = base.ev
    tv_pct_of_ev = (base.pv_tv / enterprise_value * 100.0) if enterprise_value != 0 else 0.0

    if base.tv <= 0:
        warnings.append(
            "Terminal value is not positive (negative final-year cash flow); the DCF fair value is not meaningful."
        )
    elif tv_pct_of_ev > TV_CONCENTRATION_WARNING_PCT:
        warnings.append(
            f"Terminal value is {tv_pct_of_ev:.1f}% of enterprise value (>{TV_CONCENTRATION_WARNING_PCT:.0f}%); "
            f"the valuation rests almost entirely on perpetuity assumptions."
        )

    implied_tv_multiple = None
    if terminal_ebitda is not None and terminal_ebitda > 0 and base.tv > 0:
        implied_tv_multiple = round(base.tv / terminal_ebitda, 2)
    implied_tv_nopat = None
    if terminal_nopat is not None and terminal_nopat > 0 and base.tv > 0:
        implied_tv_nopat = round(base.tv / terminal_nopat, 2)
    for name, value, band in (
        ("EV/EBITDA", implied_tv_multiple, TERMINAL_EV_EBITDA_BAND),
        ("EV/NOPAT", implied_tv_nopat, TERMINAL_EV_NOPAT_BAND),
    ):
        message = _multiple_warning(name, value, band)
        if message:
            warnings.append(message)

    # --------------------------------------------------------------------------
    # Step 5: Equity Bridge Table
    # --------------------------------------------------------------------------
    equity_bridge = render_equity_bridge(
        case, base, discounting_convention=convention_label, roll_forward_factor=roll_factor, claims_rate=claims_rate
    )

    result = DCFValue(
        enterprise_value=round(enterprise_value, 2),
        pv_explicit_fcfs=round(base.pv_fcfs, 2),
        terminal_value=round(base.tv, 2),
        pv_terminal_value=round(base.pv_tv, 2),
        terminal_value_pct_of_ev=round(tv_pct_of_ev, 1),
        terminal_method=terminal_method,
        terminal_fcf=round(base.tv_fcf, 2),
        ronic=round(ronic, 6) if terminal_method == "value_driver" else None,
        implied_terminal_ev_ebitda=implied_tv_multiple,
        implied_terminal_ev_nopat=implied_tv_nopat,
        net_debt=round(net_debt, 2),
        noncontrolling_interests=round(noncontrolling_interests, 2),
        equity_value_at_balance_sheet_date=round(base.equity_bs, 2),
        roll_forward_years=round(roll_forward_years, 4),
        roll_forward_factor=round(roll_factor, 6),
        claims_roll_forward_factor=round(claims_roll_factor, 6),
        equity_value=round(base.equity, 2),
        diluted_shares=round(diluted_shares, 2),
        fair_value_per_share=round(base.per_share, 2),
        wacc=round(wacc, 4),
        terminal_growth_rate=round(terminal_growth_rate, 4),
        forecast_years_count=num_years,
        discounting_convention=convention_label,
        equity_bridge_markdown=equity_bridge,
        warnings=tuple(warnings),
        case=case,
        point=base,
    )

    logger.info(
        f"calculate_dcf: EV=${result.enterprise_value:,.2f}M | "
        f"EqVal=${result.equity_value:,.2f}M | FairValue=${result.fair_value_per_share:.2f}/share "
        f"(WACC={wacc*100:.1f}%, g={terminal_growth_rate*100:.1f}%, TV%={result.terminal_value_pct_of_ev}%, {convention_label})"
    )

    return result


def resolve_terminal_ronic(
    diagnostics: Optional["ForecastDiagnostics"], wacc: float
) -> Tuple[Optional[float], List[str]]:
    """RONIC for the value-driver TV, clamped to [WACC, 30%].

    Prefers the forecast's steady-state RONIC, which already sized the final explicit year's reinvestment,
    so the terminal value continues the same reinvestment identity; falls back to the horizon RONIC, then
    base ROIC. None (and no warning) without a forecast.
    """
    warnings: List[str] = []
    if diagnostics is None:
        return None, warnings
    diag = diagnostics
    raw_pct = next(
        (p for p in (diag.steady_state_ronic_pct, diag.ronic_pct, diag.base_roic_pct) if p is not None), None
    )
    if raw_pct is None:
        warnings.append("No RONIC or base ROIC available; terminal value grows the final UFCF instead.")
        return None, warnings
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
    return clamped, warnings


def _terminal_nopat(final_year, terminal_tax: float) -> Optional[float]:
    """Final-year NOPAT re-taxed at the marginal rate; a loss year keeps its scheduled NOPAT."""
    if final_year is None:
        return None
    if final_year.projected_ebit <= 0:
        return final_year.projected_nopat
    return round(final_year.projected_ebit * (1.0 - terminal_tax), 2)


def estimate_intrinsic_value(
    inputs: IntrinsicValueInputs, capital: CostOfCapital, terminal_growth_rate: float
) -> IntrinsicValue:
    """Base-case DCF at the cost of capital, rolled forward from the balance-sheet date to the valuation date."""
    warnings: List[str] = []
    roll_years = 0.0
    if inputs.balance_sheet_date is None:
        warnings.append("Balance-sheet date unknown; value is stated at the fiscal year-end (no roll-forward).")
    else:
        roll_years = max(0.0, (inputs.valuation_date - inputs.balance_sheet_date).days / DAYS_PER_YEAR)
    ronic, ronic_warnings = resolve_terminal_ronic(inputs.forecast_diagnostics, capital.wacc)
    warnings.extend(ronic_warnings)
    final_year = inputs.final_year
    terminal_nopat = _terminal_nopat(final_year, capital.marginal_tax_rate)
    terms = DCFTerms(
        wacc=capital.wacc,
        terminal_growth_rate=terminal_growth_rate,
        net_debt=inputs.net_debt,
        diluted_shares=inputs.diluted_shares,
        ronic=ronic,
        noncontrolling_interests=inputs.noncontrolling_interests,
        roll_forward_years=roll_years,
        risk_free_rate=inputs.risk_free_rate,
    )
    dcf = calculate_dcf(
        projected_fcfs=inputs.projected_fcfs,
        terminal_nopat=terminal_nopat,
        terminal_ebitda=(final_year.projected_ebit + final_year.projected_depreciation) if final_year else None,
        **terms.as_kwargs(),
    )
    warnings.extend(dcf.warnings)
    ebitda = inputs.base_ebitda
    return IntrinsicValue(
        dcf=dcf,
        terms=terms,
        projected_fcfs=inputs.projected_fcfs,
        forecast_source=inputs.forecast_source,
        terminal_nopat=terminal_nopat,
        balance_sheet_date=inputs.balance_sheet_date,
        implied_ev_ebitda=round(dcf.enterprise_value / ebitda, 2) if ebitda is not None and ebitda > 0 else None,
        cash_flow_warnings=inputs.cash_flow_warnings,
        warnings=tuple(warnings),
    )
