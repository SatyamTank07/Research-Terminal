"""State and Output Schemas for DCF Valuation Specialist Agent."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class WACCAudit(BaseModel):
    """Detailed parameters and provenance of the discount rate derivation."""

    wacc: float = Field(..., description="Blended WACC as decimal (e.g. 0.0845)")
    wacc_pct: float = Field(..., description="Blended WACC as percentage (e.g. 8.45)")
    cost_of_equity_pct: float = Field(..., description="CAPM Cost of Equity Ke %")
    cost_of_debt_pre_tax_pct: float = Field(..., description="Pre-tax Kd %")
    cost_of_debt_after_tax_pct: float = Field(..., description="After-tax Kd %")
    cost_of_debt_source: str = Field(
        ...,
        description="Explicit provenance: explicit_provided, derived_from_10k_interest_expense, institutional_credit_spread_fallback, zero_debt_exemption",
    )
    equity_weight_pct: float = Field(..., description="Equity proportion We %")
    debt_weight_pct: float = Field(..., description="Debt proportion Wd %")
    formula_breakdown_markdown: str = Field(..., description="Markdown table of WACC components")


class DCFValuationOutput(BaseModel):
    """Structured artifact emitted by the DCF Valuation Specialist."""

    ticker: str = Field(..., description="Stock ticker symbol (e.g. AAPL)")
    fiscal_year: int = Field(..., description="Base fiscal year audited")

    # 1. Discount Rate & Core Assumptions
    wacc_audit: WACCAudit
    terminal_growth_rate: float = Field(
        default=0.025,
        description="Perpetual terminal growth rate as decimal (default 0.025 for 2.5%)",
    )
    discounting_convention: str = Field(
        default="mid_year",
        description="Institutional 'mid_year' (default) or 'year_end'",
    )

    # 2. Valuation Bridge ($ Millions)
    projected_fcfs: List[float] = Field(..., description="Explicit forecast period UFCFs ($M)")
    pv_explicit_fcfs: float = Field(..., description="Present value of explicit cash flows ($M)")
    pv_terminal_value: float = Field(..., description="Present value of terminal value ($M)")
    terminal_value_pct_of_ev: float = Field(..., description="Terminal value as % of Enterprise Value")
    enterprise_value: float = Field(..., description="Implied Enterprise Value ($M)")
    net_debt: float = Field(..., description="Net debt ($M); negative indicates cash surplus")
    equity_value: float = Field(..., description="Implied Equity Value ($M)")
    diluted_shares: float = Field(..., description="Diluted shares outstanding in Millions")

    # 3. Share Price, Multiple Sanity Check & Valuation Stance
    implied_fair_value_per_share: float = Field(..., description="Intrinsic fair value per share in $")
    current_share_price: Optional[float] = Field(None, description="Current market share price in $")
    upside_downside_pct: Optional[float] = Field(
        None, description="Implied upside/downside % relative to current market price"
    )
    valuation_stance: Optional[Literal["Undervalued", "Fairly Valued", "Overvalued"]] = Field(
        None,
        description="Valuation stance: Undervalued (> +10%), Overvalued (< -10%), Fairly Valued",
    )
    implied_ev_ebitda: Optional[float] = Field(
        None,
        description="Cross-check multiple: Enterprise Value / Base Year EBITDA (None if D&A unavailable)",
    )
    ev_ebitda_source: Optional[str] = Field(
        None,
        description="Provenance tag for EV/EBITDA multiple: 'derived_from_10k_ebit_plus_depreciation' or None",
    )

    # 4. Sensitivity Table & Qualitative Commentary
    sensitivity_matrix_markdown: str = Field(
        ..., description="5x5 Markdown matrix flexing WACC vs Perpetual Growth"
    )
    valuation_summary: str = Field(
        ...,
        description="Institutional commentary: value drivers, hurdle rate profile, terminal concentration, sensitivity bounds",
    )
