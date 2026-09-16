"""Deterministic Discounted Cash Flow (DCF) & Sensitivity Engine.

Enforces Architectural Principle #2: Zero Mathematical Hallucination.
Offloads valuation calculations and sensitivity grids to pure, deterministic Python functions.
Uses the 2-Stage Free Cash Flow to Firm (FCFF) Perpetual Gordon Growth Model with
support for both institutional Mid-Year Discounting and standard Year-End conventions.
"""

import logging
import math
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool

logger = logging.getLogger("finance_agent.tools.dcf")


# ==============================================================================
# 1. Structured Output Data Schema
# ==============================================================================
class DCFCalculationResult(BaseModel):
    """Structured Pydantic model for DCF valuation output and sensitivity matrix."""
    enterprise_value: float = Field(..., description="Implied enterprise value in $ millions")
    pv_explicit_fcfs: float = Field(..., description="Present value of explicit forecast period cash flows in $ millions")
    pv_terminal_value: float = Field(..., description="Present value of perpetual terminal value in $ millions")
    terminal_value_pct_of_ev: float = Field(..., description="Terminal value as a % of enterprise value (sanity metric)")
    net_debt: float = Field(
        ...,
        description="Net debt in $ millions (Total Debt - Liquid Cash); negative indicates net cash surplus"
    )
    equity_value: float = Field(..., description="Implied equity value in $ millions")
    diluted_shares: float = Field(..., description="Diluted shares outstanding in millions")
    fair_value_per_share: float = Field(..., description="Implied intrinsic fair value per share in dollars")
    wacc: float = Field(..., description="Base WACC discount rate used (e.g. 0.085 for 8.5%)")
    terminal_growth_rate: float = Field(..., description="Base perpetual growth rate used (e.g. 0.025 for 2.5%)")
    forecast_years_count: int = Field(..., description="Number of explicit forecast years")
    discounting_convention: str = Field(
        ...,
        description="Discounting convention: 'mid_year' (explicit FCFs at t-0.5, TV at t=N per McKinsey) or 'year_end'"
    )
    sensitivity_matrix_markdown: str = Field(..., description="5x5 Markdown sensitivity table flexing WACC against Growth")


# ==============================================================================
# 2. Input Validation Helper
# ==============================================================================
def _validate_finite_float(val: Any, name: str) -> float:
    """Validates that a numeric argument is a real, finite float (rejects NaN and Infinity)."""
    if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
        raise ValueError(f"{name} must be a valid finite number, got: {val}")
    return float(val)


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
) -> Dict[str, Any]:
    """
    Calculates intrinsic Enterprise Value, Equity Value, and Fair Value per share
    using a 2-stage Perpetual Gordon Growth DCF model, and constructs a 5x5 sensitivity matrix.

    All dollar quantities (cash flows, net debt) and share counts must be on the same scale
    (standardized to $ millions and millions of shares).

    NOTE ON DISCOUNTING CONVENTION:
        When mid_year_convention=True (institutional standard), explicit cash flows are discounted
        at (t - 0.5) assuming cash is generated evenly throughout the fiscal year.
        Terminal Value is evaluated as of period N and discounted at (1 + WACC)^N
        following standard McKinsey / Rosenbaum & Pearl institutional conventions.

    CRITICAL NET DEBT SIGN CONVENTION:
        net_debt = Total Debt (short + long term) minus Cash & Liquid Marketable Securities.
        If the company holds a net cash surplus (Cash > Debt), net_debt MUST be passed as a
        NEGATIVE float (e.g. -$33,763.0M for Apple).
        Equity Value is derived as: (Enterprise Value - Net Debt).

    Args:
        projected_fcfs: List of projected Unlevered Free Cash Flows (UFCF) in $ millions (e.g. 5 forward years).
        wacc: Weighted Average Cost of Capital as a decimal (e.g. 0.085 for 8.5%, max 0.50).
        terminal_growth_rate: Perpetual terminal growth rate as a decimal (e.g. 0.025 for 2.5%, max 0.05).
        net_debt: Total Debt minus Liquid Cash in $ millions (negative for net cash surplus).
        diluted_shares: Diluted shares outstanding in millions.
        mid_year_convention: True (default, institutional standard) or False (year-end).

    Returns:
        Dict conforming to DCFCalculationResult containing valuation bridge and Markdown sensitivity table.

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
    if terminal_growth_rate > 0.035:
        logger.warning(
            f"terminal_growth_rate ({terminal_growth_rate * 100:.2f}%) exceeds typical long-term "
            f"developed market GDP growth (2.0% - 3.0%)."
        )

    num_years = len(cleaned_fcfs)
    convention_label = "mid_year" if mid_year_convention else "year_end"

    logger.info(
        f"calculate_dcf: net_debt passed: ${net_debt:,.2f}M "
        f"({'Net Cash Surplus' if net_debt < 0 else 'Net Indebtedness'}) | Convention: {convention_label}"
    )

    # Reusable helper to eliminate code duplication between base calc and sensitivity loop
    def _calc_pv_explicit(rate: float) -> float:
        exponent_offset = 0.5 if mid_year_convention else 1.0
        return sum(
            fcf / ((1.0 + rate) ** (idx + exponent_offset))
            for idx, fcf in enumerate(cleaned_fcfs)
        )

    # --------------------------------------------------------------------------
    # Step 1: Present Value of Explicit Forecast Period Cash Flows
    # --------------------------------------------------------------------------
    pv_fcfs = _calc_pv_explicit(wacc)

    # --------------------------------------------------------------------------
    # Step 2: Perpetual Terminal Value (Gordon Growth Model)
    # --------------------------------------------------------------------------
    final_fcf = cleaned_fcfs[-1]
    terminal_value = (final_fcf * (1.0 + terminal_growth_rate)) / (wacc - terminal_growth_rate)
    pv_terminal_value = terminal_value / ((1.0 + wacc) ** num_years)

    # --------------------------------------------------------------------------
    # Step 3: Valuation Bridge (Enterprise Value -> Equity Value -> Per Share)
    # --------------------------------------------------------------------------
    enterprise_value = pv_fcfs + pv_terminal_value
    equity_value = enterprise_value - net_debt
    fair_value_per_share = equity_value / diluted_shares

    tv_pct_of_ev = (pv_terminal_value / enterprise_value * 100.0) if enterprise_value != 0 else 0.0

    # --------------------------------------------------------------------------
    # Step 4: 5x5 Two-Way Sensitivity Matrix (WACC vs. Perpetual Growth)
    # --------------------------------------------------------------------------
    wacc_steps = [
        round(wacc - 0.010, 4),
        round(wacc - 0.005, 4),
        round(wacc, 4),
        round(wacc + 0.005, 4),
        round(wacc + 0.010, 4),
    ]
    growth_steps = [
        round(terminal_growth_rate - 0.006, 4),
        round(terminal_growth_rate - 0.003, 4),
        round(terminal_growth_rate, 4),
        round(terminal_growth_rate + 0.003, 4),
        round(terminal_growth_rate + 0.006, 4),
    ]

    header_cols = [f"{g * 100:.1f}%" for g in growth_steps]
    table_lines = [
        "| WACC \\ Growth | " + " | ".join(header_cols) + " |",
        "|:---|" + "|".join([":---:"] * len(growth_steps)) + "|",
    ]

    for w in wacc_steps:
        row_label = f"**{w * 100:.1f}%**"
        if abs(w - wacc) < 1e-6:
            row_label += " *(Base)*"
        cells = [row_label]

        for g in growth_steps:
            if w <= g or w <= 0:
                cells.append("N/A")
                continue

            cell_pv_fcf = _calc_pv_explicit(w)
            cell_tv = (final_fcf * (1.0 + g)) / (w - g)
            cell_pv_tv = cell_tv / ((1.0 + w) ** num_years)
            cell_ev = cell_pv_fcf + cell_pv_tv
            cell_equity = cell_ev - net_debt
            cell_share_price = cell_equity / diluted_shares

            if abs(w - wacc) < 1e-6 and abs(g - terminal_growth_rate) < 1e-6:
                cells.append(f"**${cell_share_price:.2f}**")
            else:
                cells.append(f"${cell_share_price:.2f}")

        table_lines.append("| " + " | ".join(cells) + " |")

    sensitivity_markdown = "\n".join(table_lines)

    result = DCFCalculationResult(
        enterprise_value=round(enterprise_value, 2),
        pv_explicit_fcfs=round(pv_fcfs, 2),
        pv_terminal_value=round(pv_terminal_value, 2),
        terminal_value_pct_of_ev=round(tv_pct_of_ev, 1),
        net_debt=round(net_debt, 2),
        equity_value=round(equity_value, 2),
        diluted_shares=round(diluted_shares, 2),
        fair_value_per_share=round(fair_value_per_share, 2),
        wacc=round(wacc, 4),
        terminal_growth_rate=round(terminal_growth_rate, 4),
        forecast_years_count=num_years,
        discounting_convention=convention_label,
        sensitivity_matrix_markdown=sensitivity_markdown,
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
) -> Dict[str, Any]:
    """
    Calculate the intrinsic fair value per share of a company using a deterministic DCF model.

    Executes a 2-stage Perpetual Gordon Growth valuation and builds a 5x5 sensitivity matrix
    flexing WACC against Perpetual Growth.

    All dollar quantities and share counts MUST be on the same scale (millions).

    Args:
        projected_fcfs: List of 5 forward-year Unlevered Free Cash Flows (UFCF) in $ millions.
        wacc: Weighted Average Cost of Capital as a decimal (e.g. 0.085 for 8.5%, max 0.50).
        terminal_growth_rate: Perpetual terminal growth rate as a decimal (e.g. 0.025 for 2.5%, max 0.05).
        net_debt: Total Debt minus Liquid Cash in $ millions.
                  IMPORTANT: If the firm has a cash surplus (Cash > Debt), pass as a NEGATIVE number
                  (e.g. -$33763.0 for Apple).
        diluted_shares: Diluted common shares outstanding in millions (e.g. 15004.7 for Apple).
        mid_year_convention: True (default, institutional standard) or False (year-end).

    Returns:
        Dictionary containing enterprise_value, equity_value, fair_value_per_share,
        terminal_value_pct_of_ev, discounting_convention, and sensitivity_matrix_markdown.
    """
    return calculate_dcf_with_sensitivity(
        projected_fcfs=projected_fcfs,
        wacc=wacc,
        terminal_growth_rate=terminal_growth_rate,
        net_debt=net_debt,
        diluted_shares=diluted_shares,
        mid_year_convention=mid_year_convention,
    )
