"""Deterministic Discounted Cash Flow (DCF) & Sensitivity Engine.

Enforces Architectural Principle #2: Zero Mathematical Hallucination.
Offloads valuation calculations and sensitivity grids to pure, deterministic Python functions.
Uses the 2-Stage Free Cash Flow to Firm (FCFF) Perpetual Gordon Growth Model with
support for both institutional Mid-Year Discounting and standard Year-End conventions,
an optional value-driver terminal cash flow (NOPAT x (1 - g / RONIC)), a full
enterprise-to-equity bridge (net debt and noncontrolling interests) and a roll-forward
from the balance-sheet date to the valuation date.
"""

import logging
import math
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger("finance_agent.tools.dcf")

TV_CONCENTRATION_WARNING_PCT = 85.0
STALE_ROLL_FORWARD_YEARS = 1.5


# ==============================================================================
# 1. Structured Output Data Schema
# ==============================================================================
class DCFCalculationResult(BaseModel):
    """Structured Pydantic model for DCF valuation output and sensitivity matrix."""
    enterprise_value: float = Field(..., description="Implied enterprise value in $ millions")
    pv_explicit_fcfs: float = Field(..., description="Present value of explicit forecast period cash flows in $ millions")
    terminal_value: float = Field(..., description="Undiscounted perpetual terminal value at year N in $ millions")
    pv_terminal_value: float = Field(..., description="Present value of perpetual terminal value in $ millions")
    terminal_value_pct_of_ev: float = Field(..., description="Terminal value as a % of enterprise value (sanity metric)")
    terminal_method: str = Field(
        ...,
        description="'value_driver' (NOPAT x (1+g) x (1 - g/RONIC)) or 'last_fcf_growth' (final UFCF x (1+g))",
    )
    terminal_fcf: float = Field(..., description="Year N+1 cash flow capitalized in the terminal value ($M)")
    ronic: Optional[float] = Field(None, description="Return on new invested capital used by the value-driver TV")
    implied_terminal_ev_ebitda: Optional[float] = Field(
        None, description="Terminal value / final-year EBITDA (exit-multiple sanity check)"
    )
    net_debt: float = Field(
        ...,
        description="Net debt in $ millions (Total Debt - Liquid Cash); negative indicates net cash surplus"
    )
    noncontrolling_interests: float = Field(0.0, description="Noncontrolling interests deducted in the bridge ($M)")
    equity_value_at_balance_sheet_date: float = Field(..., description="EV - net debt - NCI ($M)")
    roll_forward_years: float = Field(0.0, description="Years rolled forward from the balance-sheet date")
    roll_forward_factor: float = Field(1.0, description="(1 + cost of equity) ^ roll_forward_years")
    equity_value: float = Field(..., description="Implied equity value at the valuation date in $ millions")
    diluted_shares: float = Field(..., description="Diluted shares outstanding in millions")
    fair_value_per_share: float = Field(..., description="Implied intrinsic fair value per share in dollars")
    wacc: float = Field(..., description="Base WACC discount rate used (e.g. 0.085 for 8.5%)")
    terminal_growth_rate: float = Field(..., description="Base perpetual growth rate used (e.g. 0.025 for 2.5%)")
    forecast_years_count: int = Field(..., description="Number of explicit forecast years")
    discounting_convention: str = Field(
        ...,
        description="'mid_year' (explicit FCFs and perpetuity TV at t-0.5) or 'year_end'"
    )
    sensitivity_matrix_markdown: str = Field(..., description="5x5 Markdown sensitivity table flexing WACC against Growth")
    equity_bridge_markdown: str = Field(..., description="Markdown EV-to-equity-to-per-share bridge")
    warnings: List[str] = Field(default_factory=list, description="Valuation sanity warnings")


# ==============================================================================
# 2. Input Validation Helper
# ==============================================================================
def _validate_finite_float(val: Any, name: str) -> float:
    """Validates that a numeric argument is a real, finite float (rejects NaN and Infinity)."""
    if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
        raise ValueError(f"{name} must be a valid finite number, got: {val}")
    return float(val)


def _optional_finite(val: Any, name: str) -> Optional[float]:
    return None if val is None else _validate_finite_float(val, name)


# ==============================================================================
# 3. Core Deterministic Calculation Engine
# ==============================================================================
def calculate_dcf_with_sensitivity(
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
    cost_of_equity: Optional[float] = None,
    risk_free_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculates intrinsic Enterprise Value, Equity Value, and Fair Value per share
    using a 2-stage Perpetual Gordon Growth DCF model, and constructs a 5x5 sensitivity matrix.

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
        Equity = EV - net debt - noncontrolling interests, valued at the balance-sheet date, then
        rolled forward to the valuation date at the cost of equity (or WACC when Ke is not given).
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
    cost_of_equity = _optional_finite(cost_of_equity, "cost_of_equity")
    risk_free_rate = _optional_finite(risk_free_rate, "risk_free_rate")

    if wacc <= 0:
        raise ValueError(f"WACC must be strictly positive (> 0), got: {wacc}")

    # Catch LLM percentage formatting handoff error (e.g. 8.5 instead of 0.085)
    if wacc >= 1.0:
        raise ValueError(
            f"WACC ({wacc}) appears to be passed as a whole percentage rather than a decimal. "
            f"Please pass WACC as a decimal (e.g. 0.085 for 8.5%, max 0.50)."
        )
    if wacc > 0.50:
        raise ValueError(f"WACC ({wacc * 100:.1f}%) is unrealistically high (maximum allowed is 50.0%).")

    if diluted_shares <= 0:
        raise ValueError(f"diluted_shares must be strictly positive (> 0), got: {diluted_shares}")

    if terminal_growth_rate >= wacc:
        raise ValueError(
            f"terminal_growth_rate ({terminal_growth_rate * 100:.2f}%) must be strictly less "
            f"than WACC ({wacc * 100:.2f}%) for Gordon Growth convergence."
        )

    # Economic reasonableness boundary checks on perpetual growth
    if terminal_growth_rate > 0.05:
        raise ValueError(
            f"terminal_growth_rate ({terminal_growth_rate * 100:.2f}%) exceeds economically plausible "
            f"perpetual GDP growth rate (maximum allowed is 5.0%)."
        )
    if terminal_growth_rate < -0.02:
        raise ValueError(
            f"terminal_growth_rate ({terminal_growth_rate * 100:.2f}%) is unrealistically negative "
            f"(minimum allowed is -2.0%)."
        )

    warnings: List[str] = []
    if terminal_growth_rate > 0.035:
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
    roll_rate = cost_of_equity if cost_of_equity is not None else wacc
    roll_factor = (1.0 + roll_rate) ** roll_forward_years

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

    def _terminal_fcf(g: float) -> Optional[float]:
        if terminal_method == "value_driver":
            if ronic <= g:
                return None
            return terminal_nopat * (1.0 + g) * (1.0 - g / ronic)
        return cleaned_fcfs[-1] * (1.0 + g)

    def _value_at(rate: float, g: float) -> Optional[Dict[str, float]]:
        tv_fcf = _terminal_fcf(g)
        if tv_fcf is None or rate <= g or rate <= 0:
            return None
        pv_fcfs = sum(fcf / ((1.0 + rate) ** (idx + 1 - offset)) for idx, fcf in enumerate(cleaned_fcfs))
        tv = tv_fcf / (rate - g)
        pv_tv = tv / ((1.0 + rate) ** (num_years - offset))
        ev = pv_fcfs + pv_tv
        equity_bs = ev - net_debt - noncontrolling_interests
        equity = equity_bs * roll_factor
        return {
            "tv_fcf": tv_fcf, "pv_fcfs": pv_fcfs, "tv": tv, "pv_tv": pv_tv, "ev": ev,
            "equity_bs": equity_bs, "equity": equity, "per_share": equity / diluted_shares,
        }

    # --------------------------------------------------------------------------
    # Steps 1-3: Explicit PV, Terminal Value, Valuation Bridge
    # --------------------------------------------------------------------------
    base = _value_at(wacc, terminal_growth_rate)
    enterprise_value = base["ev"]
    tv_pct_of_ev = (base["pv_tv"] / enterprise_value * 100.0) if enterprise_value != 0 else 0.0

    if base["tv"] <= 0:
        warnings.append(
            "Terminal value is not positive (negative final-year cash flow); the DCF fair value is not meaningful."
        )
    elif tv_pct_of_ev > TV_CONCENTRATION_WARNING_PCT:
        warnings.append(
            f"Terminal value is {tv_pct_of_ev:.1f}% of enterprise value (>{TV_CONCENTRATION_WARNING_PCT:.0f}%); "
            f"the valuation rests almost entirely on perpetuity assumptions."
        )

    implied_tv_multiple = None
    if terminal_ebitda is not None and terminal_ebitda > 0 and base["tv"] > 0:
        implied_tv_multiple = round(base["tv"] / terminal_ebitda, 2)

    # --------------------------------------------------------------------------
    # Step 4: 5x5 Two-Way Sensitivity Matrix (WACC vs. Perpetual Growth)
    # --------------------------------------------------------------------------
    wacc_steps = [round(wacc + d, 4) for d in (-0.010, -0.005, 0.0, 0.005, 0.010)]
    growth_steps = [round(terminal_growth_rate + d, 4) for d in (-0.006, -0.003, 0.0, 0.003, 0.006)]

    header_cols = [f"{g * 100:.1f}%" for g in growth_steps]
    table_lines = [
        "| WACC \\ Growth | " + " | ".join(header_cols) + " |",
        "|:---|" + "|".join([":---:"] * len(growth_steps)) + "|",
    ]
    for i, w in enumerate(wacc_steps):
        row_label = f"**{w * 100:.1f}%**" + (" *(Base)*" if i == 2 else "")
        cells = [row_label]
        for j, g in enumerate(growth_steps):
            if i == 2 and j == 2:
                cells.append(f"**${base['per_share']:.2f}**")
                continue
            cell = _value_at(w, g)
            cells.append("N/A" if cell is None else f"${cell['per_share']:.2f}")
        table_lines.append("| " + " | ".join(cells) + " |")
    sensitivity_markdown = "\n".join(table_lines)

    # --------------------------------------------------------------------------
    # Step 5: Equity Bridge Table
    # --------------------------------------------------------------------------
    bridge_lines = [
        "| Valuation Bridge | Value |",
        "|:---|---:|",
        f"| PV of explicit UFCF ({num_years} yrs, {convention_label}) | ${base['pv_fcfs']:,.1f}M |",
        f"| PV of terminal value ({terminal_method}) | ${base['pv_tv']:,.1f}M |",
        f"| **Enterprise value** | **${enterprise_value:,.1f}M** |",
        f"| {'Plus: net cash' if net_debt < 0 else 'Less: net debt'} | ${abs(net_debt):,.1f}M |",
        f"| Less: noncontrolling interests | ${noncontrolling_interests:,.1f}M |",
        f"| Equity value at balance-sheet date | ${base['equity_bs']:,.1f}M |",
    ]
    if roll_forward_years > 0:
        bridge_lines.append(
            f"| Roll-forward {roll_forward_years:.2f} yrs @ {roll_rate * 100:.2f}% | x{roll_factor:.4f} |"
        )
    bridge_lines += [
        f"| **Equity value at valuation date** | **${base['equity']:,.1f}M** |",
        f"| Diluted shares | {diluted_shares:,.2f}M |",
        f"| **Fair value per share** | **${base['per_share']:.2f}** |",
    ]

    result = DCFCalculationResult(
        enterprise_value=round(enterprise_value, 2),
        pv_explicit_fcfs=round(base["pv_fcfs"], 2),
        terminal_value=round(base["tv"], 2),
        pv_terminal_value=round(base["pv_tv"], 2),
        terminal_value_pct_of_ev=round(tv_pct_of_ev, 1),
        terminal_method=terminal_method,
        terminal_fcf=round(base["tv_fcf"], 2),
        ronic=round(ronic, 6) if terminal_method == "value_driver" else None,
        implied_terminal_ev_ebitda=implied_tv_multiple,
        net_debt=round(net_debt, 2),
        noncontrolling_interests=round(noncontrolling_interests, 2),
        equity_value_at_balance_sheet_date=round(base["equity_bs"], 2),
        roll_forward_years=round(roll_forward_years, 4),
        roll_forward_factor=round(roll_factor, 6),
        equity_value=round(base["equity"], 2),
        diluted_shares=round(diluted_shares, 2),
        fair_value_per_share=round(base["per_share"], 2),
        wacc=round(wacc, 4),
        terminal_growth_rate=round(terminal_growth_rate, 4),
        forecast_years_count=num_years,
        discounting_convention=convention_label,
        sensitivity_matrix_markdown=sensitivity_markdown,
        equity_bridge_markdown="\n".join(bridge_lines),
        warnings=warnings,
    )

    logger.info(
        f"calculate_dcf_with_sensitivity: EV=${result.enterprise_value:,.2f}M | "
        f"EqVal=${result.equity_value:,.2f}M | FairValue=${result.fair_value_per_share:.2f}/share "
        f"(WACC={wacc*100:.1f}%, g={terminal_growth_rate*100:.1f}%, TV%={result.terminal_value_pct_of_ev}%, {convention_label})"
    )

    return result.model_dump()


# ==============================================================================
# 4. LangChain Agent Tool Decorator
# ==============================================================================
@tool
def calculate_dcf_tool(
    projected_fcfs: List[float],
    wacc: float,
    terminal_growth_rate: float,
    net_debt: float,
    diluted_shares: float,
    mid_year_convention: bool = True,
    noncontrolling_interests: float = 0.0,
) -> Dict[str, Any]:
    """
    Calculate the intrinsic fair value per share of a company using a deterministic DCF model.

    Executes a 2-stage Perpetual Gordon Growth valuation and builds a 5x5 sensitivity matrix
    flexing WACC against Perpetual Growth.

    All dollar quantities and share counts MUST be on the same scale (millions).

    Args:
        projected_fcfs: Forward-year Unlevered Free Cash Flows (UFCF) in $ millions.
        wacc: Weighted Average Cost of Capital as a decimal (e.g. 0.085 for 8.5%, max 0.50).
        terminal_growth_rate: Perpetual terminal growth rate as a decimal (e.g. 0.025 for 2.5%, max 0.05).
        net_debt: Total Debt minus Liquid Cash in $ millions.
                  IMPORTANT: If the firm has a cash surplus (Cash > Debt), pass as a NEGATIVE number
                  (e.g. -$33763.0 for Apple).
        diluted_shares: Diluted common shares outstanding in millions (e.g. 15004.7 for Apple).
        mid_year_convention: True (default, institutional standard) or False (year-end).
        noncontrolling_interests: Noncontrolling interests in $ millions, deducted from equity value.

    Returns:
        Dictionary containing enterprise_value, equity_value, fair_value_per_share,
        terminal_value_pct_of_ev, equity_bridge_markdown and sensitivity_matrix_markdown.
    """
    try:
        return calculate_dcf_with_sensitivity(
            projected_fcfs=projected_fcfs,
            wacc=wacc,
            terminal_growth_rate=terminal_growth_rate,
            net_debt=net_debt,
            diluted_shares=diluted_shares,
            mid_year_convention=mid_year_convention,
            noncontrolling_interests=noncontrolling_interests,
        )
    except Exception as e:
        return {"error": f"calculate_dcf_tool failed: {e}"}
