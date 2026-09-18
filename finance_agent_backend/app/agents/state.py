"""Shared State & Structured Output Schemas for Multi-Agent Equity Research.

Defines Pydantic models for individual sub-agent contracts (Auditor, Forecaster,
Valuation Specialist, Moat Strategist, Risk Analyst, Synthesizer) and the centralized
LangGraph TypedDict state graph.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict
from pydantic import BaseModel, Field

from app.agents.tools.financial_math_tools import (
    BalanceSheetResult,
    ProfitabilityRatiosResult,
    SolvencyRatiosResult,
    YearFinancialsResult,
)

# Ergonomic aliases mapping architectural specification names to calculation engine results
YearFinancials = YearFinancialsResult
BalanceSheetSnapshot = BalanceSheetResult


# ==============================================================================
# 1. Financial Auditor & Statement Analyst Output Schema (Milestone 2)
# ==============================================================================
class FinancialAuditOutput(BaseModel):
    """Structured artifact emitted by the Financial Auditor Agent."""

    ticker: str = Field(..., description="Stock ticker symbol (e.g. AAPL)")
    fiscal_year: int = Field(..., description="Target 10-K fiscal year audited")

    # 1. Deterministic Math & Multi-Year Ratios (Calculated via tool)
    multi_year_history: List[YearFinancialsResult] = Field(
        ...,
        description="3-year contiguous history (Rev, GP, EBIT, NI, OCF, CapEx, FCF, Margins, YoY Growth)",
    )
    balance_sheet: BalanceSheetResult = Field(
        ...,
        description="Audited liquidity bridge (Cash, Securities, Short/Long Debt, Net Debt, Shares, Equity)",
    )
    profitability_and_return_ratios: ProfitabilityRatiosResult = Field(
        ...,
        description="ROIC, ROE, NOPAT, Invested Capital, and derived effective tax rate",
    )
    solvency_and_liquidity_ratios: SolvencyRatiosResult = Field(
        ...,
        description="Net Debt/EBITDA, Current Ratio, Debt/Equity",
    )
    forensic_red_flags: List[str] = Field(
        default_factory=list,
        description="Algorithmic red flags (accrual divergence, weak FCF conversion, AR/Inventory lag)",
    )
    restatement_notes: List[str] = Field(
        default_factory=list,
        description="Audit notes on cross-filing comparisons, restatements, or reclassifications across filings",
    )

    # 2. Raw Markdown Tables (Preserved for UI presentation & Report Synthesizer)
    income_statement_markdown_table: str = Field(
        default="", description="Audited Statement of Operations Markdown table"
    )
    balance_sheet_markdown_table: str = Field(
        default="", description="Audited Consolidated Balance Sheets Markdown table"
    )
    cash_flow_markdown_table: str = Field(
        default="", description="Audited Statement of Cash Flows Markdown table"
    )


    # 3. Auditor Synthesis & Audit Trail
    auditor_summary: str = Field(
        ...,
        description="Qualitative synthesis explaining earnings quality, working capital dynamics, and red flag context",
    )
    citations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of chunk_ids, breadcrumbs, and table titles used in the audit",
    )


# ==============================================================================
# 2. DCF Valuation Specialist Output Schema (Milestone 3)
# ==============================================================================
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


# ==============================================================================
# 3. Financial Forecasting Analyst Output Schema (Milestone 4)
# ==============================================================================
from app.agents.tools.forecast_tools import (
    ForecastYearResult,
    ForecastScheduleResult,
    ProvenanceMode,
)

ForecastYear = ForecastYearResult


class ForecastOutput(BaseModel):
    """Structured artifact emitted by the Financial Forecasting Analyst Agent."""

    ticker: str = Field(..., description="Stock ticker symbol (e.g. AAPL)")
    fiscal_year: int = Field(..., description="Base 10-K fiscal year audited (e.g. 2025)")
    base_revenue: float = Field(..., description="Audited base year revenue in $ Millions")
    forecast_horizon_years: int = Field(default=5, description="Number of forward explicit projection years")
    revenue_cagr_pct: float = Field(..., description="5-year Compound Annual Growth Rate of Revenue as %")
    cumulative_5yr_fcf: float = Field(..., description="Total cumulative explicit UFCF generated over 5 years in $ Millions")
    average_annual_fcf: float = Field(..., description="Average annual UFCF in $ Millions")
    provenance_mode: ProvenanceMode = Field(
        ..., description="Accounting bridge used: 'comprehensive_line_item' or 'simplified_nopat_less_capex'"
    )
    tax_rate_pct: float = Field(..., description="Effective tax rate percentage used")

    # Direct handoff into DCF Valuation Specialist
    projected_fcfs: List[float] = Field(
        ..., description="Explicit forecast period UFCFs ($M) feeding directly into calculate_dcf_tool"
    )
    forecast_schedule: List[ForecastYearResult] = Field(
        ..., description="Year-by-year projected line items (Revenue, EBIT, NOPAT, CapEx, UFCF)"
    )
    forecast_table_markdown: str = Field(
        ..., description="Publication-ready Markdown forecast comparison table"
    )

    # Forecaster Synthesis & Rationale
    growth_rationale: str = Field(
        ..., description="Evidence-backed defense of revenue growth curve linked to Item 7 MD&A disclosures"
    )
    margin_expansion_rationale: str = Field(
        ..., description="Rationale for EBIT margin progression (operating leverage, product mix, cost discipline)"
    )
    reinvestment_rationale: Optional[str] = Field(
        None, description="Analysis of CapEx and working capital intensity relative to historical trend"
    )
    citations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Citations to Item 7 tables and MD&A disclosures"
    )


# ==============================================================================
# 4. Central LangGraph State Graph Definition (Multi-Agent Coordinator)
# ==============================================================================
class EquityResearchState(TypedDict, total=False):
    """Centralized TypedDict tracking the state across all 6 specialized agents."""

    # Routing & Session Context
    user_query: str
    ticker: str
    company_name: str
    fiscal_year: int
    document_id: str
    query_type: str

    # Sub-agent structured payloads
    business_moat: Optional[Dict[str, Any]]
    financial_audit: Optional[FinancialAuditOutput]
    forecast: Optional[ForecastOutput]
    dcf_valuation: Optional[DCFValuationOutput]
    risk_audit: Optional[Dict[str, Any]]

    # Final compiled output
    final_report: Optional[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    error_message: Optional[str]

