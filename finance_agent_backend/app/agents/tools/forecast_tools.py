"""Deterministic 5-Year Financial Forecasting Schedule Engine.

Enforces Architectural Principle #2: Zero Mathematical Hallucination.
Offloads forward financial projections, compounding growth curves, and cash-flow bridges
to pure, deterministic Python functions.

Supports Flexible Dual-Mode UFCF Calculations (Option C):
  1. 'comprehensive_line_item': UFCF = NOPAT + D&A - CapEx - ΔNWC
  2. 'simplified_nopat_less_capex': UFCF = NOPAT - CapEx - ΔNWC
"""

import logging
import math
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
from langchain_core.tools import tool

logger = logging.getLogger("finance_agent.tools.forecast")


# ==============================================================================
# 1. Validation & Normalization Helpers
# ==============================================================================
def _validate_finite_float(val: Any, name: str) -> float:
    """Strictly validates that a numeric input is a real, finite number (not None, NaN, or Inf)."""
    if val is None:
        raise ValueError(f"'{name}' is required and cannot be None.")
    try:
        f_val = float(val)
    except (ValueError, TypeError):
        raise ValueError(f"'{name}' must be a valid numeric float, got: {val!r}")

    if math.isnan(f_val):
        raise ValueError(f"'{name}' cannot be NaN.")
    if math.isinf(f_val):
        raise ValueError(f"'{name}' cannot be infinite.")
    return f_val


def _normalize_ratio(
    val: Any,
    name: str,
    required: bool = True,
    default: Optional[float] = None,
    max_bound: float = 1.0,
    allow_negative: bool = False,
) -> Optional[float]:
    """
    Normalizes standard capital/statutory ratios (e.g. tax rate, CapEx % of rev).
    Uses threshold = 1.0 (e.g. 21.0 -> 0.21, 3.06 -> 0.0306).
    Rejects None for required fields and enforces post-normalization bounds.
    """
    if val is None:
        if required:
            raise ValueError(f"'{name}' is required and cannot be None.")
        return default

    f_val = _validate_finite_float(val, name)
    if not allow_negative and f_val < 0.0:
        raise ValueError(f"'{name}' cannot be negative, got: {f_val}")

    # Standard percentage-to-decimal auto-normalization
    if abs(f_val) > 1.0:
        normalized = f_val / 100.0
        logger.info(f"Auto-normalizing '{name}': {f_val} -> {normalized:.4f}")
        f_val = normalized

    if f_val > max_bound:
        raise ValueError(
            f"'{name}' ({f_val*100:.1f}%) exceeds maximum allowable threshold ({max_bound*100:.1f}%)."
        )

    return f_val


def _normalize_growth_rates(
    vals: Any,
    name: str,
    expected_length: int,
) -> List[float]:
    """
    Validates, auto-normalizes, and bounds revenue growth rates.
    Supports legitimate hyper-growth (e.g. 1.40 for +140% YoY).
    Auto-normalizes whole-percentage series (e.g. [7.0, 6.5, 6.0]) if any magnitude > 5.0.
    Enforces floor g > -0.95 to protect going-concern and CAGR calculation.
    """
    if vals is None:
        raise ValueError(f"'{name}' is required and cannot be None.")
    if not isinstance(vals, (list, tuple)):
        raise ValueError(f"'{name}' must be a list of numbers, got: {type(vals).__name__}")
    if len(vals) != expected_length:
        raise ValueError(f"'{name}' must contain exactly {expected_length} entries (got {len(vals)}).")

    raw_floats = [_validate_finite_float(v, f"{name}[{i}]") for i, v in enumerate(vals)]

    # If any value has abs > 5.0 (500%), user passed whole percentages (e.g. 7.0%, 40.0%)
    if any(abs(v) > 5.0 for v in raw_floats):
        logger.info(f"Auto-normalizing '{name}' from whole percentages to decimals (values > 5.0 detected).")
        cleaned = [v / 100.0 for v in raw_floats]
    else:
        cleaned = raw_floats

    # Strict post-normalization sanity bounds: -95% < g <= 500%
    for i, g in enumerate(cleaned):
        if g <= -0.95:
            raise ValueError(
                f"'{name}[{i}]' ({g*100:.1f}%) cannot be <= -95.0%. "
                f"Extreme revenue contraction invalidates going-concern valuation."
            )
        if g > 5.0:
            raise ValueError(
                f"'{name}[{i}]' ({g*100:.1f}%) exceeds maximum allowable threshold (+500.0%)."
            )

    return cleaned


def _normalize_margins(
    vals: Any,
    name: str,
    expected_length: int,
) -> List[float]:
    """
    Validates, auto-normalizes, and bounds operating margins.
    Supports distressed margins (e.g. -1.50 for -150% EBIT margin).
    Auto-normalizes whole-percentage series (e.g. [32.0, 32.5]) if any magnitude > 5.0.
    Enforces sanity bounds: -200% <= margin <= +100%.
    """
    if vals is None:
        raise ValueError(f"'{name}' is required and cannot be None.")
    if not isinstance(vals, (list, tuple)):
        raise ValueError(f"'{name}' must be a list of numbers, got: {type(vals).__name__}")
    if len(vals) != expected_length:
        raise ValueError(f"'{name}' must contain exactly {expected_length} entries (got {len(vals)}).")

    raw_floats = [_validate_finite_float(v, f"{name}[{i}]") for i, v in enumerate(vals)]

    # If any value has abs > 5.0 (500%), user passed whole percentages (e.g. 32.0%, 60.0%)
    if any(abs(v) > 5.0 for v in raw_floats):
        logger.info(f"Auto-normalizing '{name}' from whole percentages to decimals (values > 5.0 detected).")
        cleaned = [v / 100.0 for v in raw_floats]
    else:
        cleaned = raw_floats

    # Strict post-normalization sanity bounds: -200% <= m <= +100%
    for i, m in enumerate(cleaned):
        if m < -2.0:
            raise ValueError(
                f"'{name}[{i}]' ({m*100:.1f}%) is below minimum allowable margin (-200.0%)."
            )
        if m > 1.0:
            raise ValueError(
                f"'{name}[{i}]' ({m*100:.1f}%) exceeds maximum allowable operating margin (+100.0%)."
            )

    return cleaned


# ==============================================================================
# 2. Structured Output Data Schemas
# ==============================================================================
ProvenanceMode = Literal["comprehensive_line_item", "simplified_nopat_less_capex"]


class ForecastYearResult(BaseModel):
    """Structured container for single forward year projected metrics."""

    projected_year: int = Field(..., description="Projected fiscal year (e.g. 2026)")
    projected_revenue: float = Field(..., description="Projected revenue in $ Millions")
    projected_revenue_growth_pct: float = Field(..., description="YoY revenue growth rate as %")
    projected_ebit: float = Field(..., description="Projected Operating Income (EBIT) in $ Millions")
    projected_ebit_margin_pct: float = Field(..., description="Operating margin as %")
    projected_nopat: float = Field(..., description="Net Operating Profit After Tax in $ Millions")
    projected_capex: float = Field(..., description="Projected Capital Expenditures in $ Millions")
    projected_depreciation: Optional[float] = Field(
        None, description="Projected D&A in $ Millions (present in comprehensive mode)"
    )
    projected_nwc_change: float = Field(
        0.0, description="Projected change in non-cash Working Capital in $ Millions"
    )
    projected_unlevered_fcf: float = Field(
        ..., description="Projected Unlevered Free Cash Flow (UFCF) in $ Millions"
    )


class ForecastScheduleResult(BaseModel):
    """Structured container for complete 5-year forecast schedule and bridge."""

    base_revenue: float = Field(..., description="Base year audited revenue in $ Millions")
    base_year: int = Field(..., description="Base fiscal year audited (e.g. 2025)")
    forecast_horizon_years: int = Field(..., description="Number of explicit forecast years (e.g. 5)")
    revenue_cagr_pct: float = Field(..., description="Compound annual growth rate of revenue over forecast period %")
    cumulative_5yr_fcf: float = Field(..., description="Total cumulative UFCF over explicit forecast period in $ Millions")
    average_annual_fcf: float = Field(..., description="Average annual UFCF in $ Millions")
    provenance_mode: ProvenanceMode = Field(
        ...,
        description="Accounting bridge used: 'comprehensive_line_item' or 'simplified_nopat_less_capex'",
    )
    tax_rate_pct: float = Field(..., description="Effective tax rate percentage used")
    projected_fcfs: List[float] = Field(
        ..., description="Direct 5-year UFCF list for valuation engine handoff in $ Millions"
    )
    forecast_schedule: List[ForecastYearResult] = Field(
        ..., description="Year-by-year projected line items"
    )
    forecast_table_markdown: str = Field(
        ..., description="Formatted Markdown comparison table comparing Base year to projected years"
    )


# ==============================================================================
# 3. Core Deterministic Calculation Engine
# ==============================================================================
def calculate_forecast_schedule(
    base_revenue: float,
    base_year: int,
    revenue_growth_rates: List[float],
    operating_margins: List[float],
    tax_rate: float,
    capex_pct_of_revenue: Union[float, List[float]],
    nwc_change_pct_of_revenue: Union[float, List[float]] = 0.0,
    depreciation_pct_of_revenue: Optional[Union[float, List[float]]] = None,
) -> Dict[str, Any]:
    """
    Calculates a deterministic 5-year financial forecast schedule, derives Unlevered Free Cash Flows (UFCF),
    and constructs a publication-ready Markdown table.

    Option C Dual-Mode Provenance:
      - If depreciation_pct_of_revenue is provided:
          provenance_mode = 'comprehensive_line_item'
          UFCF = NOPAT + D&A - CapEx - ΔNWC
      - If depreciation_pct_of_revenue is None:
          provenance_mode = 'simplified_nopat_less_capex'
          UFCF = NOPAT - CapEx - ΔNWC

    Args:
        base_revenue: Base year audited revenue in $ Millions (e.g. 416161.0 for AAPL). Required.
        base_year: Base fiscal year (e.g. 2025). Required.
        revenue_growth_rates: Forward YoY revenue growth rates (list of floats/percentages). Required.
        operating_margins: Forward EBIT operating margins (list of floats/percentages). Required.
        tax_rate: Effective tax rate (decimal e.g. 0.1561 or whole % e.g. 15.61). Required.
        capex_pct_of_revenue: CapEx as % of revenue (single float or list matching horizon). Required.
        nwc_change_pct_of_revenue: ΔNWC as % of incremental revenue (default 0.0).
        depreciation_pct_of_revenue: Optional D&A as % of revenue (single float or list matching horizon).

    Returns:
        Dict conforming to ForecastScheduleResult schema.
    """
    # --------------------------------------------------------------------------
    # Step 0: Input Validation & Horizon Normalization
    # --------------------------------------------------------------------------
    base_rev = _validate_finite_float(base_revenue, "base_revenue")
    if base_rev <= 0.0:
        raise ValueError(f"'base_revenue' must be strictly positive (> 0), got: {base_rev}")

    if base_year is None:
        raise ValueError("'base_year' is required and cannot be None.")
    if not isinstance(base_year, int) or base_year < 1900 or base_year > 2100:
        raise ValueError(f"'base_year' must be a valid 4-digit year (e.g. 2025), got: {base_year}")

    if not revenue_growth_rates:
        raise ValueError("'revenue_growth_rates' cannot be empty or None.")

    horizon = len(revenue_growth_rates)
    if horizon < 1 or horizon > 10:
        raise ValueError(f"Forecast horizon must be between 1 and 10 years, got: {horizon}")

    # Validate & Normalize Growth Rates and Margins with domain-specific thresholds
    clean_growths = _normalize_growth_rates(revenue_growth_rates, "revenue_growth_rates", horizon)
    clean_margins = _normalize_margins(operating_margins, "operating_margins", horizon)

    # Validate & Normalize Tax Rate (strictly required, max 50%)
    clean_tax = _normalize_ratio(tax_rate, "tax_rate", required=True, max_bound=0.50, allow_negative=False)

    # Validate & Normalize CapEx Rate(s) (strictly required, max 100%)
    if capex_pct_of_revenue is None:
        raise ValueError("'capex_pct_of_revenue' is required and cannot be None.")

    if isinstance(capex_pct_of_revenue, (list, tuple)):
        if len(capex_pct_of_revenue) != horizon:
            raise ValueError(
                f"'capex_pct_of_revenue' list must contain exactly {horizon} entries (got {len(capex_pct_of_revenue)})."
            )
        clean_capex = [
            _normalize_ratio(c, f"capex_pct_of_revenue[{i}]", required=True, max_bound=1.0, allow_negative=False)
            for i, c in enumerate(capex_pct_of_revenue)
        ]
    else:
        single_capex = _normalize_ratio(
            capex_pct_of_revenue, "capex_pct_of_revenue", required=True, max_bound=1.0, allow_negative=False
        )
        clean_capex = [single_capex] * horizon

    # Validate & Normalize NWC Rate(s) (optional, default 0.0)
    if isinstance(nwc_change_pct_of_revenue, (list, tuple)):
        if len(nwc_change_pct_of_revenue) != horizon:
            raise ValueError(
                f"'nwc_change_pct_of_revenue' list must contain exactly {horizon} entries (got {len(nwc_change_pct_of_revenue)})."
            )
        clean_nwc = [
            _normalize_ratio(w, f"nwc_change_pct_of_revenue[{i}]", required=False, default=0.0, max_bound=0.50, allow_negative=True)
            for i, w in enumerate(nwc_change_pct_of_revenue)
        ]
    else:
        single_nwc = _normalize_ratio(
            nwc_change_pct_of_revenue, "nwc_change_pct_of_revenue", required=False, default=0.0, max_bound=0.50, allow_negative=True
        )
        clean_nwc = [single_nwc] * horizon

    # Determine Provenance Mode based on Depreciation (optional, default None)
    provenance_mode: ProvenanceMode
    clean_depr: Optional[List[float]] = None
    if depreciation_pct_of_revenue is not None:
        provenance_mode = "comprehensive_line_item"
        if isinstance(depreciation_pct_of_revenue, (list, tuple)):
            if len(depreciation_pct_of_revenue) != horizon:
                raise ValueError(
                    f"'depreciation_pct_of_revenue' list must contain exactly {horizon} entries (got {len(depreciation_pct_of_revenue)})."
                )
            clean_depr = [
                _normalize_ratio(d, f"depreciation_pct_of_revenue[{i}]", required=True, max_bound=0.50, allow_negative=False)
                for i, d in enumerate(depreciation_pct_of_revenue)
            ]
        else:
            single_depr = _normalize_ratio(
                depreciation_pct_of_revenue, "depreciation_pct_of_revenue", required=True, max_bound=0.50, allow_negative=False
            )
            clean_depr = [single_depr] * horizon
    else:
        provenance_mode = "simplified_nopat_less_capex"

    logger.info(
        f"calculate_forecast_schedule: BaseRev=${base_rev:,.2f}M | BaseYear={base_year} | "
        f"Horizon={horizon}Y | Mode={provenance_mode} | Tax={clean_tax*100:.2f}%"
    )

    # --------------------------------------------------------------------------
    # Step 1: Compounding Year-by-Year Financial Bridge
    # --------------------------------------------------------------------------
    schedule: List[ForecastYearResult] = []
    projected_fcfs: List[float] = []
    prev_rev = base_rev

    for idx in range(horizon):
        proj_year = base_year + idx + 1
        g = clean_growths[idx]
        m = clean_margins[idx]
        c = clean_capex[idx]
        w = clean_nwc[idx]

        # Compounding revenue with non-positive collapse guard
        proj_rev = prev_rev * (1.0 + g)
        if proj_rev <= 0.0:
            raise ValueError(
                f"Projected revenue in FY{proj_year} (${proj_rev:,.2f}M) collapsed to non-positive. "
                f"Going-concern DCF is invalid."
            )
        delta_rev = proj_rev - prev_rev

        # Earnings
        proj_ebit = proj_rev * m
        proj_nopat = proj_ebit * (1.0 - clean_tax)

        # Reinvestment
        proj_capex = proj_rev * c
        proj_nwc_change = delta_rev * w

        # Free Cash Flow derivation according to Mode
        if provenance_mode == "comprehensive_line_item" and clean_depr is not None:
            d = clean_depr[idx]
            proj_depr = proj_rev * d
            proj_ufcf = proj_nopat + proj_depr - proj_capex - proj_nwc_change
        else:
            proj_depr = None
            proj_ufcf = proj_nopat - proj_capex - proj_nwc_change

        year_res = ForecastYearResult(
            projected_year=proj_year,
            projected_revenue=round(proj_rev, 2),
            projected_revenue_growth_pct=round(g * 100.0, 2),
            projected_ebit=round(proj_ebit, 2),
            projected_ebit_margin_pct=round(m * 100.0, 2),
            projected_nopat=round(proj_nopat, 2),
            projected_capex=round(proj_capex, 2),
            projected_depreciation=round(proj_depr, 2) if proj_depr is not None else None,
            projected_nwc_change=round(proj_nwc_change, 2),
            projected_unlevered_fcf=round(proj_ufcf, 2),
        )
        schedule.append(year_res)
        projected_fcfs.append(round(proj_ufcf, 2))
        prev_rev = proj_rev

    # --------------------------------------------------------------------------
    # Step 2: Cumulative Metrics & CAGR (Defensive against complex numbers)
    # --------------------------------------------------------------------------
    final_rev = schedule[-1].projected_revenue
    if final_rev <= 0.0 or base_rev <= 0.0:
        raise ValueError(
            f"Cannot compute Revenue CAGR with non-positive revenue "
            f"(base: ${base_rev:,.2f}M, final: ${final_rev:,.2f}M)."
        )
    # math.pow raises ValueError rather than returning a complex number
    revenue_cagr = (math.pow(final_rev / base_rev, 1.0 / horizon) - 1.0) * 100.0
    cumulative_fcf = sum(projected_fcfs)
    average_fcf = cumulative_fcf / horizon

    # --------------------------------------------------------------------------
    # Step 3: Markdown Comparison Table Construction
    # --------------------------------------------------------------------------
    col_headers = [f"Base (FY{base_year})"] + [f"FY{s.projected_year}E" for s in schedule]
    header_line = "| Metric ($ Millions) | " + " | ".join(col_headers) + " |"
    divider_line = "|:---|" + "|".join([":---:"] * len(col_headers)) + "|"

    rows = [header_line, divider_line]

    # Row 1: Revenue
    rev_cells = [f"${base_rev:,.0f}"] + [f"${s.projected_revenue:,.0f}" for s in schedule]
    rows.append("| Revenue | " + " | ".join(rev_cells) + " |")

    # Row 2: YoY Revenue Growth
    growth_cells = ["—"] + [f"{s.projected_revenue_growth_pct:+.1f}%" for s in schedule]
    rows.append("| Revenue Growth YoY | " + " | ".join(growth_cells) + " |")

    # Row 3: Operating Income (EBIT)
    ebit_cells = ["—"] + [f"${s.projected_ebit:,.0f}" for s in schedule]
    rows.append("| Operating Income (EBIT) | " + " | ".join(ebit_cells) + " |")

    # Row 4: Operating Margin
    margin_cells = ["—"] + [f"{s.projected_ebit_margin_pct:.1f}%" for s in schedule]
    rows.append("| Operating Margin | " + " | ".join(margin_cells) + " |")

    # Row 5: NOPAT
    nopat_cells = ["—"] + [f"${s.projected_nopat:,.0f}" for s in schedule]
    rows.append(f"| NOPAT (Effective Tax {clean_tax*100:.1f}%) | " + " | ".join(nopat_cells) + " |")

    # Row 6: CapEx
    capex_cells = ["—"] + [f"${s.projected_capex:,.0f}" for s in schedule]
    rows.append("| Capital Expenditures | " + " | ".join(capex_cells) + " |")

    # Row 7: D&A (if comprehensive mode)
    if provenance_mode == "comprehensive_line_item":
        depr_cells = ["—"] + [f"${s.projected_depreciation:,.0f}" for s in schedule]
        rows.append("| Depreciation & Amortization | " + " | ".join(depr_cells) + " |")

    # Row 8: Change in NWC (if non-zero)
    if any(s.projected_nwc_change != 0.0 for s in schedule):
        nwc_cells = ["—"] + [f"${s.projected_nwc_change:,.0f}" for s in schedule]
        rows.append("| Change in Working Capital (ΔNWC) | " + " | ".join(nwc_cells) + " |")

    # Row 9: Unlevered FCF
    fcf_cells = ["—"] + [f"**${s.projected_unlevered_fcf:,.0f}**" for s in schedule]
    rows.append("| **Unlevered Free Cash Flow (UFCF)** | " + " | ".join(fcf_cells) + " |")

    forecast_table_markdown = "\n".join(rows)

    result = ForecastScheduleResult(
        base_revenue=round(base_rev, 2),
        base_year=base_year,
        forecast_horizon_years=horizon,
        revenue_cagr_pct=round(revenue_cagr, 2),
        cumulative_5yr_fcf=round(cumulative_fcf, 2),
        average_annual_fcf=round(average_fcf, 2),
        provenance_mode=provenance_mode,
        tax_rate_pct=round(clean_tax * 100.0, 2),
        projected_fcfs=projected_fcfs,
        forecast_schedule=schedule,
        forecast_table_markdown=forecast_table_markdown,
    )

    logger.info(
        f"calculate_forecast_schedule complete: 5Y CAGR={result.revenue_cagr_pct:.2f}% | "
        f"Cumulative FCF=${result.cumulative_5yr_fcf:,.2f}M | Mode={provenance_mode}"
    )

    return result.model_dump()


# ==============================================================================
# 4. LangChain Agent Tool Decorator
# ==============================================================================
@tool
def calculate_forecast_schedule_tool(
    base_revenue: float,
    base_year: int,
    revenue_growth_rates: List[float],
    operating_margins: List[float],
    tax_rate: float,
    capex_pct_of_revenue: Union[float, List[float]],
    nwc_change_pct_of_revenue: Union[float, List[float]] = 0.0,
    depreciation_pct_of_revenue: Optional[Union[float, List[float]]] = None,
) -> Dict[str, Any]:
    """
    Calculate a deterministic 5-year financial forecast schedule and Unlevered Free Cash Flows (UFCF).

    Enforces Zero Mathematical Hallucination. Uses dual-mode accounting:
      - 'comprehensive_line_item' when depreciation_pct_of_revenue is provided.
      - 'simplified_nopat_less_capex' when depreciation_pct_of_revenue is omitted.

    Args:
        base_revenue: Base year audited revenue in $ Millions (e.g. 416161.0 for Apple FY2025). Required.
        base_year: Base fiscal year audited (e.g. 2025). Required.
        revenue_growth_rates: 5 forward-year YoY revenue growth rates (as decimals e.g. [0.07, 0.065, 0.06, 0.05, 0.045]
                              or percentages e.g. [7.0, 6.5, 6.0, 5.0, 4.5]). Required.
        operating_margins: 5 forward-year EBIT margins (e.g. [0.32, 0.325, 0.33, 0.33, 0.33] or [32.0, 32.5...]). Required.
        tax_rate: Effective tax rate (e.g. 0.1561 for Apple, or 15.61). Required.
        capex_pct_of_revenue: CapEx as a percentage of revenue (single number e.g. 0.0306 or list of 5 forward years). Required.
        nwc_change_pct_of_revenue: Reinvestment in working capital as % of incremental revenue (default 0.0).
        depreciation_pct_of_revenue: Optional D&A as % of revenue (single number or list of 5 forward years).
                                     If provided, enables 'comprehensive_line_item' mode.

    Returns:
        Dictionary containing projected_fcfs (list of 5 floats for DCF tool handoff),
        revenue_cagr_pct, cumulative_5yr_fcf, provenance_mode, forecast_schedule, and forecast_table_markdown.
    """
    try:
        return calculate_forecast_schedule(
            base_revenue=base_revenue,
            base_year=base_year,
            revenue_growth_rates=revenue_growth_rates,
            operating_margins=operating_margins,
            tax_rate=tax_rate,
            capex_pct_of_revenue=capex_pct_of_revenue,
            nwc_change_pct_of_revenue=nwc_change_pct_of_revenue,
            depreciation_pct_of_revenue=depreciation_pct_of_revenue,
        )
    except Exception as e:
        logger.error(f"calculate_forecast_schedule_tool failed: {e}", exc_info=True)
        return {"error": f"calculate_forecast_schedule_tool failed: {e}"}
